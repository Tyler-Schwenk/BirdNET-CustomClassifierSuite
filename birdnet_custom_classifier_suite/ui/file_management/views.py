from __future__ import annotations

from pathlib import Path
from typing import List
import time

import pandas as pd
import streamlit as st

from birdnet_custom_classifier_suite.ui.file_management import engine
from birdnet_custom_classifier_suite.ui.common import folder_picker
import tempfile
import os
import shutil
import zipfile


def panel(container=None):
    st_ = container if container is not None else st
    st_.header("File Management — Audio splitting")
    st_.write("Split audio files into fixed-length segments and export them to a folder.")

    default_in = Path('AudioData') / 'curated'  # placeholder default
    default_out = Path('AudioData') / 'curated_splits'

    # Primary input: drag & drop uploaded audio files. (You can still process a local folder by leaving this empty.)
    uploaded_files = st_.file_uploader(
        "Drag & drop audio files here (these files will be uploaded and split). Leave empty to split files from a local folder path.",
        type=["wav", "flac", "mp3", "ogg", "m4a"],
        accept_multiple_files=True,
        key='fm_uploader',
    )

    # Output choice: either save segments to a server-side folder (specify path) or produce a ZIP for download
    output_mode = st_.radio("Output mode", options=["save_to_path", "download_zip"], index=0, format_func=lambda x: {"save_to_path":"Save to server path","download_zip":"Download ZIP"}[x])
    out_folder = None
    if output_mode == 'save_to_path':
        out_folder = folder_picker(
            label="Output root folder (segments saved under <out>/<orig_stem>/)",
            key="fm_out_folder",
            initial_dir=default_out.parent if default_out.parent.exists() else Path.cwd(),
            relative_to=None,  # Keep absolute paths for output
            help_text="Select folder where split audio segments will be saved",
            text_input=True
        ) or str(default_out)
    
    # Optional local folder input (used when not uploading files)
    in_folder = folder_picker(
        label="Input folder (local path; used if no files are uploaded)",
        key="fm_in_folder",
        initial_dir=default_in if default_in.exists() else Path.cwd(),
        relative_to=None,  # Keep absolute paths
        help_text="Select folder containing audio files to split",
        text_input=True
    ) or str(default_in)


    seg_len = st_.number_input("Segment length (seconds)", min_value=0.1, value=3.0, step=0.1)
    keep_trailing = st_.radio("Trailing segment", options=["keep", "drop"], index=0)
    keep_trailing_bool = True if keep_trailing == 'keep' else False

    st_.markdown("---")
    st_.subheader("Naming / file template")
    name_mode = st_.selectbox("Name mode", options=["concat", "append", "custom"], index=0)
    sep = st_.text_input("Separator (used for concat mode)", value="_")
    index_width = st_.number_input("Index zero-pad width", min_value=1, max_value=6, value=3)
    custom_template = "{orig}{sep}{i}{ext}"
    if name_mode == 'custom':
        custom_template = st_.text_input("Custom template (use {orig}, {i}, {ext}, {sep})", value=custom_template)

    st_.markdown("---")
    run_btn = st_.button("Split files")

    if run_btn:
        # Determine output destination
        tmp_out_dir = None
        if output_mode == 'save_to_path':
            if not out_folder:
                st_.error("Output folder must be provided when 'Save to server path' is selected.")
                return
            dest_base = Path(out_folder)
        else:
            # create a temporary output dir to collect segments which will be zipped for download
            tmp_out_dir = Path(tempfile.mkdtemp(prefix='split_out_'))
            dest_base = tmp_out_dir

        # If uploaded files are present, split them; otherwise operate on the input folder
        if uploaded_files:
            st_.info(f"Splitting {len(uploaded_files)} uploaded files into {dest_base} — segment length: {seg_len}s")
        else:
            src = Path(in_folder)
            if not src.exists() or not src.is_dir():
                st_.error(f"Input folder not found: {src}")
                # cleanup tmp dir if created
                if tmp_out_dir is not None:
                    try:
                        shutil.rmtree(tmp_out_dir)
                    except Exception:
                        pass
                return
            st_.info(f"Splitting audio files from {src} into {dest_base} — segment length: {seg_len}s")
        try:
            mode_template = name_mode
            if name_mode == 'custom':
                mode_template = custom_template
            all_paths = []
            files_proc = 0
            segs_created = 0
            with st_.spinner("Splitting files..."):
                if uploaded_files:
                    # handle uploaded files one by one: write to temp and call split_file
                    for up in uploaded_files:
                        files_proc += 1
                        # write uploaded content to a temp file
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
                        try:
                            tmp.write(up.getvalue())
                            tmp.flush()
                            tmp.close()
                            out_dir = dest_base / Path(up.name).stem
                            written = engine.split_file(
                                Path(tmp.name),
                                out_dir,
                                segment_length_s=float(seg_len),
                                keep_trailing=keep_trailing_bool,
                                name_template=mode_template,
                                index_width=int(index_width),
                                sep=sep,
                            )
                            segs_created += len(written)
                            all_paths.extend(written)
                        finally:
                            try:
                                os.unlink(tmp.name)
                            except Exception:
                                pass
                else:
                    files_proc, segs_created, all_paths = engine.split_folder(
                        src_root=src,
                        dest_root=dest_base,
                        segment_length_s=float(seg_len),
                        keep_trailing=keep_trailing_bool,
                        name_template=mode_template,
                        index_width=int(index_width),
                        sep=sep,
                    )

            st_.success(f"Processed {files_proc} files -> created {segs_created} segments")
            if all_paths:
                # show a small preview table
                df = pd.DataFrame({"path": [str(p) for p in all_paths]})
                st_.dataframe(df.head(200), use_container_width=True)
                # provide a small download of a CSV manifest
                csvb = df.to_csv(index=False).encode('utf-8')
                fname = f"split_manifest_{int(time.time())}.csv"
                st_.download_button("Download manifest CSV", data=csvb, file_name=fname, mime='text/csv')

            # If we used a temporary out dir for zip download, create the zip and offer it
            if tmp_out_dir is not None:
                try:
                    zip_base = str(tmp_out_dir) + '_archive'
                    zip_path = shutil.make_archive(zip_base, 'zip', root_dir=str(tmp_out_dir))
                    with open(zip_path, 'rb') as fh:
                        zip_bytes = fh.read()
                    zip_name = f"split_segments_{int(time.time())}.zip"
                    st_.download_button("Download ZIP of segments", data=zip_bytes, file_name=zip_name, mime='application/zip')
                finally:
                    # cleanup temp dirs and zip file
                    try:
                        shutil.rmtree(tmp_out_dir)
                    except Exception:
                        pass
                    try:
                        os.remove(zip_path)
                    except Exception:
                        pass
        except Exception as e:
            st_.error(f"Splitting failed: {e}")
