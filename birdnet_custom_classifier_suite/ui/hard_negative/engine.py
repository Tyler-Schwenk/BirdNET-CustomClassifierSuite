"""
Inference engine helpers for the hard-negative UI.

This module provides a small wrapper that will attempt to run the
BirdNET-Analyzer CLI (if available in the environment) to perform
inference over a folder of audio files. It aggregates detection CSVs
into a per-file maximum confidence DataFrame which the UI consumes.

The wrapper is defensive: if the analyzer CLI is not installed or the
invocation fails, it raises a RuntimeError with the captured stdout/stderr
so the UI can show the user what went wrong and how to proceed.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import csv
from typing import Optional

import pandas as pd
from birdnet_custom_classifier_suite.pipeline.pipeline import build_inference_cmd
from birdnet_custom_classifier_suite.utils.config import load_config
import time


def _find_confidence_column(cols):
    # returns the first column that looks like a confidence column
    for c in cols:
        lc = c.lower()
        if lc.endswith('confidence') or lc.endswith('conf') or 'radr' in lc:
            return c
    # fallback: any numeric column besides file/time
    for c in cols:
        lc = c.lower()
        if lc in ('confidence', 'score'):
            return c
    return None


def run_analyzer_cli(input_dir: Path, out_dir: Path, model_path: Optional[Path] = None, extra_args: Optional[list] = None, timeout: Optional[int] = None) -> Path:
    """Attempt to run the BirdNET-Analyzer CLI as a subprocess.

    Returns the path to the output directory where the analyzer wrote CSVs.

    Raises RuntimeError if the analyzer invocation fails or is not present.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the same CLI invocation used elsewhere in the repo (scripts/run_low_quality_inference)
    # which uses '-o' for outdir, '-c' to pass model path, '--rtype csv' and '--combine_results'.
    cmd = [sys.executable, '-m', 'birdnet_analyzer.analyze', str(input_dir), '-o', str(out_dir)]
    if model_path:
        # The analyzer historically expects '-c' (classifier/model) to pass the model file
        cmd += ['-c', str(model_path)]
    # Request CSV outputs and a combined table where supported
    cmd += ['--rtype', 'csv', '--combine_results']
    if extra_args:
        cmd += list(extra_args)

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to invoke analyzer CLI: {e}") from e

    if proc.returncode != 0:
        # include stdout/stderr to help the user debug
        raise RuntimeError(f"Analyzer CLI failed (rc={proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    return out_dir


def run_analyzer_cli_stream(input_dir: Path, out_dir: Path, model_path: Optional[Path] = None, extra_args: Optional[list] = None, timeout: Optional[int] = None):
    """Run the analyzer CLI as a subprocess and return (proc, out_dir).

    The caller may iterate over proc.stdout to stream logs. The out_dir is created
    before starting the process so the caller knows where outputs will be written.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, '-m', 'birdnet_analyzer.analyze', str(input_dir), '-o', str(out_dir)]
    if model_path:
        cmd += ['-c', str(model_path)]
    cmd += ['--rtype', 'csv', '--combine_results']
    if extra_args:
        cmd += list(extra_args)

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to invoke analyzer CLI: {e}") from e

    return proc, out_dir, cmd


def collect_per_file_max(out_dir: Path) -> pd.DataFrame:
    """Scan the output directory for CSV detection files and compute per-file max confidence.

    Returns a DataFrame with columns: File (absolute path string), radr_max_confidence (float)
    """
    out_dir = Path(out_dir)
    csv_files = list(out_dir.glob('**/*.csv'))
    if not csv_files:
        raise RuntimeError(f"No CSV outputs found in analyzer output folder: {out_dir}")

    rows = []
    for csvp in csv_files:
        try:
            df = pd.read_csv(csvp)
        except Exception:
            # skip unreadable CSVs
            continue
        if df.empty:
            continue
        conf_col = _find_confidence_column(list(df.columns))
        file_col = next((c for c in df.columns if c.lower() == 'file' or c.lower().endswith('file')), None)
        if conf_col is None:
            # try to infer a numeric column
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            conf_col = numeric_cols[0] if numeric_cols else None
        if file_col is None:
            # try first column as file
            file_col = df.columns[0]
        if conf_col is None:
            # no confidence-like column; skip
            continue
        for _, r in df.iterrows():
            try:
                v = float(r[conf_col])
            except Exception:
                continue
            fname = str(r[file_col])
            rows.append((fname, v))

    if not rows:
        raise RuntimeError("No confidence values found in analyzer outputs.")

    per_file = {}
    for fname, v in rows:
        # use basename matching later in the UI; keep original fname
        per_file.setdefault(fname, []).append(v)

    out_rows = []
    for fname, vals in per_file.items():
        out_rows.append({'File': fname, 'radr_max_confidence': float(max(vals))})

    df_out = pd.DataFrame(out_rows)
    return df_out


def run_inference_and_collect(input_dir: Path, model_path: Optional[Path] = None, extra_args: Optional[list] = None, timeout: Optional[int] = None) -> pd.DataFrame:
    """High-level helper: run analyzer CLI and return per-file max confidence DataFrame.

    This is intentionally simple: it creates a temporary output folder, runs the analyzer CLI
    there, and aggregates CSV outputs. If the analyzer is not available or fails, a
    RuntimeError is raised with captured details.
    """
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / 'analyzer_out'
        out_dir.mkdir(parents=True, exist_ok=True)
        run_analyzer_cli(input_dir=input_dir, out_dir=out_dir, model_path=model_path, extra_args=extra_args, timeout=timeout)
        df = collect_per_file_max(out_dir)
        return df


def run_inference_and_collect_stream(input_dir: Path, model_path: Optional[Path] = None, extra_args: Optional[list] = None, timeout: Optional[int] = None):
    """Run analyzer in a temp out dir and return (proc, out_dir).

    The caller should stream proc.stdout and, after proc.wait(), call
    collect_per_file_max(out_dir) to obtain the DataFrame.
    """
    td = tempfile.mkdtemp()
    out_dir = Path(td) / 'analyzer_out'
    out_dir.mkdir(parents=True, exist_ok=True)
    proc, out_dir, cmd = run_analyzer_cli_stream(input_dir=input_dir, out_dir=out_dir, model_path=model_path, extra_args=extra_args, timeout=timeout)
    return proc, out_dir, cmd


def run_inference_for_experiment(exp_name: str, input_dir: Path, experiments_root: Path = Path("experiments"), split_template: str = "test_ood", timeout: Optional[int] = None) -> pd.DataFrame:
    """Run inference using the canonical command built by pipeline.build_inference_cmd.

    This loads the experiment config (expects `experiments/<exp_name>/config_used.yaml` or
    `config.yaml`), calls build_inference_cmd to get the canonical analyzer invocation, then
    patches the built command to use `input_dir` as the dataset path and writes outputs into
    an experiment-local inference folder (unique per-run). Returns the per-file max-confidence
    DataFrame produced by collect_per_file_max.

    Raises RuntimeError on failures; the raised exception message contains stdout/stderr for
    debugging.
    """
    exp_dir = Path(experiments_root) / exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Prefer config_used.yaml if present (pipeline writes this), else fall back to config.yaml
    cfg_candidates = [exp_dir / 'config_used.yaml', exp_dir / 'config.yaml']
    cfg_path = next((p for p in cfg_candidates if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"No experiment config found in {exp_dir}. Looked for: {cfg_candidates}")

    cfg = load_config(str(cfg_path))

    # Ask the pipeline to build a canonical command (uses model from the experiment dir)
    cmd = build_inference_cmd(cfg, exp_dir, split_template)

    # Replace the analyzer's dataset argument (the positional split_dir) with our input_dir.
    # The canonical builder places the split_dir as the 4th element: [python, -m, module, split_dir, '-o', outdir, ...]
    if len(cmd) < 4:
        raise RuntimeError(f"Unexpected command shape from build_inference_cmd: {cmd}")
    cmd[3] = str(Path(input_dir).resolve())

    # Create a unique output folder under the experiment's inference area so we don't clobber
    # existing results. Use a timestamp suffix.
    stamp = int(time.time())
    out_root = exp_dir / 'inference' / f'ui_{stamp}'
    out_root.mkdir(parents=True, exist_ok=True)

    # Find the '-o' flag in the command and replace the following value with our out_root
    try:
        o_idx = cmd.index('-o')
        cmd[o_idx + 1] = str(out_root)
    except ValueError:
        # If no -o present, append it
        cmd += ['-o', str(out_root)]

    # Run the command
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to invoke analyzer command: {e}") from e

    if proc.returncode != 0:
        raise RuntimeError(f"Analyzer (experiment {exp_name}) failed (rc={proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    # Aggregate analyzer outputs
    df = collect_per_file_max(out_root)

    return df


def run_inference_for_experiment_stream(exp_name: str, input_dir: Path, experiments_root: Path = Path("experiments"), split_template: str = "test_ood", timeout: Optional[int] = None):
    """Build the canonical experiment command and start it with Popen.

    Returns (proc, out_root). Caller should stream proc.stdout and call
    collect_per_file_max(out_root) after the process completes.
    """
    exp_dir = Path(experiments_root) / exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    cfg_candidates = [exp_dir / 'config_used.yaml', exp_dir / 'config.yaml']
    cfg_path = next((p for p in cfg_candidates if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"No experiment config found in {exp_dir}. Looked for: {cfg_candidates}")

    cfg = load_config(str(cfg_path))

    cmd = build_inference_cmd(cfg, exp_dir, split_template)

    if len(cmd) < 4:
        raise RuntimeError(f"Unexpected command shape from build_inference_cmd: {cmd}")
    cmd[3] = str(Path(input_dir).resolve())

    stamp = int(time.time())
    out_root = exp_dir / 'inference' / f'ui_{stamp}'
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        o_idx = cmd.index('-o')
        cmd[o_idx + 1] = str(out_root)
    except ValueError:
        cmd += ['-o', str(out_root)]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to invoke analyzer command: {e}") from e

    return proc, out_root, cmd
