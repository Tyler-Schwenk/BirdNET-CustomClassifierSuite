from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import math
import soundfile as sf


def _format_name(orig: str, idx: int, index_width: int, ext: str, template: str, sep: str) -> str:
    i_str = str(idx).zfill(index_width)
    if template == 'concat':
        return f"{orig}{sep}{i_str}{ext}"
    if template == 'append':
        return f"{orig}_{i_str}{ext}"
    # custom: allow {orig}, {i}, {ext}
    try:
        return template.format(orig=orig, i=i_str, ext=ext, sep=sep)
    except Exception:
        return f"{orig}{sep}{i_str}{ext}"


def split_file(
    src: Path,
    dest_dir: Path,
    segment_length_s: float = 3.0,
    keep_trailing: bool = True,
    name_template: str = 'concat',
    index_width: int = 3,
    sep: str = '_',
) -> List[Path]:
    """Split a single audio file into segments.

    Returns a list of written segment paths.
    """
    src = Path(src)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    with sf.SoundFile(src, 'r') as fh:
        sr = fh.samplerate
        frames_per_seg = int(math.floor(segment_length_s * sr))
        if frames_per_seg <= 0:
            raise ValueError("segment_length_s must be > 0")

        orig = src.stem
        ext = src.suffix or '.wav'
        idx = 1
        while True:
            data = fh.read(frames_per_seg, dtype='float32')
            if data is None:
                break
            if len(data) == 0:
                break
            # If last chunk is shorter than segment and we do not keep trailing, stop
            if len(data) < frames_per_seg and not keep_trailing:
                break

            out_name = _format_name(orig, idx, index_width, ext, name_template, sep)
            out_path = dest_dir / out_name
            # soundfile.write will infer format from extension
            sf.write(str(out_path), data, sr)
            written.append(out_path)
            idx += 1

    return written


def split_folder(
    src_root: Path,
    dest_root: Path,
    exts: Optional[List[str]] = None,
    segment_length_s: float = 3.0,
    keep_trailing: bool = True,
    name_template: str = 'concat',
    index_width: int = 3,
    sep: str = '_',
) -> Tuple[int, int, List[Path]]:
    """Split all audio files under src_root (non-recursive) and place segments under dest_root.

    Returns (files_processed, segments_created, list_of_paths)
    """
    src_root = Path(src_root)
    dest_root = Path(dest_root)
    if exts is None:
        exts = ['wav', 'flac', 'mp3', 'ogg', 'm4a']
    exts = [e.lower().lstrip('.') for e in exts]

    files = [p for p in sorted(src_root.iterdir()) if p.is_file() and p.suffix.lstrip('.').lower() in exts]
    total_files = 0
    total_segs = 0
    all_paths: List[Path] = []
    for f in files:
        total_files += 1
        out_dir = dest_root / f.stem
        written = split_file(
            f,
            out_dir,
            segment_length_s=segment_length_s,
            keep_trailing=keep_trailing,
            name_template=name_template,
            index_width=index_width,
            sep=sep,
        )
        total_segs += len(written)
        all_paths.extend(written)

    return total_files, total_segs, all_paths
