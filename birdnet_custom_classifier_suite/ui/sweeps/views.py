"""View components for sweep parameter input and action handling."""

from __future__ import annotations

from pathlib import Path
import re, time, subprocess, sys
from typing import Any

import streamlit as st
import yaml

from .types import SweepState
from .utils import parse_num_list, parse_float, validate_quality, calculate_config_count


def sweep_form() -> SweepState:
    """
    Render sweep parameter input form and return populated SweepState.
    
    Returns:
        SweepState with all user-provided parameters
    """
    state = SweepState()
    
    colA, colB = st.columns(2)
    
    # Left column: Sweep metadata and base parameters
    with colA:
        state.stage = st.number_input("Stage number", min_value=0, max_value=99, value=1, step=1)
        state.sweep_name = st.text_input("Sweep name", value=f"stage{state.stage}_spec")
        # Default the sweeps folder to stage{stage}_sweep regardless of spec name
        state.out_dir = st.text_input("Output folder for configs", value=f"config/sweeps/stage{state.stage}_sweep")
        
        st.subheader("Base parameters")
        state.epochs = st.number_input("epochs", min_value=1, value=50, step=1)
        state.batch_size = st.number_input("batch_size", min_value=1, value=32, step=1)
        
        learning_rate_str = st.text_input("learning_rate (single value)", value="0.0005")
        state.learning_rate = parse_float(learning_rate_str, 0.0005)
        
        dropout_str = st.text_input("dropout (single value)", value="0.25")
        state.dropout = parse_float(dropout_str, 0.25)
        
        upsampling_ratio_str = st.text_input("upsampling_ratio (single value)", value="0.0")
        state.upsampling_ratio = parse_float(upsampling_ratio_str, 0.0)
        
        state.mixup = st.checkbox("mixup", value=False)
        state.label_smoothing = st.checkbox("label_smoothing", value=False)
        state.focal_loss = st.checkbox("focal_loss", value=False)
        
        # Conditional focal loss params
        if state.focal_loss:
            state.focal_loss_gamma = st.number_input(
                "focal_loss_gamma", 
                min_value=0.0, 
                value=2.0, 
                step=0.1, 
                help="Focusing parameter for focal loss"
            )
            state.focal_loss_alpha = st.number_input(
                "focal_loss_alpha", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.25, 
                step=0.05, 
                help="Class balance parameter for focal loss"
            )
        
        # Advanced frequency/overlap params
        with st.expander("Advanced audio parameters", expanded=False):
            state.fmin = st.number_input(
                "fmin (Hz)", 
                min_value=0, 
                value=0, 
                step=100, 
                help="Minimum frequency for bandpass filter"
            )
            state.fmax = st.number_input(
                "fmax (Hz)", 
                min_value=0, 
                value=15000, 
                step=100, 
                help="Maximum frequency for bandpass filter"
            )
            state.overlap = st.number_input(
                "overlap", 
                min_value=0.0, 
                max_value=2.9, 
                value=0.0, 
                step=0.1, 
                help="Overlap of prediction segments"
            )
    
    # Right column: Axes (values to sweep)
    with colB:
        st.subheader("Axes (values to sweep)")
        
        seeds_str = st.text_input("seed values (comma-separated)", value="123", key="sweep_seeds")
        state.seeds = parse_num_list(seeds_str, int)
        
        hidden_units_str = st.text_input(
            "hidden_units (comma-separated)", 
            value="0,128,512", 
            key="sweep_hidden_units",
            help="Use 0 for no hidden layer"
        )
        state.hidden_units = parse_num_list(hidden_units_str, int)
        
        dropout_axis_str = st.text_input(
            "dropout values (comma-separated)", 
            value="0.0,0.25", 
            key="sweep_dropout"
        )
        state.dropout_axis = parse_num_list(dropout_axis_str, float)
        
        lr_axis_str = st.text_input(
            "learning_rate values (comma-separated)", 
            value="0.0001,0.0005,0.001", 
            key="sweep_lr"
        )
        state.learning_rate_axis = parse_num_list(lr_axis_str, float)
        
        st.markdown("**Quality sweep options**")
        quality_combos_raw = st.multiselect(
            "Select quality combinations",
            options=["high", "medium", "low", "high,medium", "high,low", "medium,low", "high,medium,low"],
            default=["high", "medium", "low", "high,medium"],
            key="sweep_quality_combos",
            help="Valid quality values: high, medium, low. Select one or more combinations to sweep over."
        )
        
        # Parse quality combinations from multiselect (already comma-separated strings)
        quality_axes = []
        for combo_str in quality_combos_raw:
            combo = [q.strip() for q in combo_str.split(",") if q.strip()]
            if combo:
                quality_axes.append(combo)
        state.quality_combos = quality_axes
        
        state.balance_opts = st.multiselect(
            "balance options", 
            options=[True, False], 
            default=[True], 
            key="sweep_balance"
        )
        state.upsampling_mode_opts = st.multiselect(
            "upsampling_mode options",
            options=["repeat", "linear", "mean", "smote"],
            default=["repeat"],
            key="sweep_upmode",
            help="BirdNET-Analyzer valid modes: repeat, linear, mean, smote. Set upsampling_ratio=0.0 to effectively disable upsampling."
        )
        
        up_ratio_axis_str = st.text_input(
            "upsampling_ratio values (comma-separated)", 
            value="0.0,0.25", 
            key="sweep_upratio"
        )
        state.upsampling_ratio_axis = parse_num_list(up_ratio_axis_str, float)
    
    # Validate quality values
    is_valid, invalid_qualities = validate_quality(state.quality_combos)
    if not is_valid:
        st.warning(
            f"⚠️ Invalid quality values detected: {', '.join(set(invalid_qualities))}. "
            "Valid values are: high, medium, low"
        )
    
    # Calculate and display config count
    axes = state.get_axes_dict()
    config_count = calculate_config_count(axes)
    
    st.markdown("### Preview spec")
    st.info(f"📊 This sweep will generate **{config_count} configurations**")
    
    if config_count > 500:
        st.warning(
            f"⚠️ Large sweep detected ({config_count} configs). "
            "Consider reducing axis combinations or running in batches."
        )
    
    # Preview YAML
    preview = {
        "stage": state.stage,
        "out_dir": state.out_dir,
        "axes": axes,
        "base_params": state.get_base_params_dict(),
    }
    st.code(yaml.safe_dump(preview, sort_keys=False), language="yaml")
    
    return state


def render_action_buttons(stage: int, sweep_name: str, out_dir: str) -> tuple[bool, bool, bool, bool]:
    """
    Render action buttons at top of sweep form.
    
    Args:
        stage: Stage number for display
        sweep_name: Sweep name for display
        out_dir: Output directory for display
    
    Returns:
        (save_clicked, generate_clicked, run_clicked, regen_run_clicked)
    """
    btn_cols = st.columns(4)
    with btn_cols[0]:
        save_spec_btn = st.button(
            "💾 Save spec YAML", 
            key="save_spec_top",
            help="Save the sweep specification to config/sweep_specs/<name>.yaml for later use"
        )
    with btn_cols[1]:
        gen_configs_btn = st.button(
            "⚙️ Generate configs", 
            key="gen_configs_top",
            help="Generate individual experiment YAML configs in the output folder"
        )
    with btn_cols[2]:
        run_sweep_btn = st.button(
            "▶️ Run existing sweep", 
            key="run_sweep_top",
            help=f"Run the existing configs in: {out_dir}"
        )
    with btn_cols[3]:
        regen_run_btn = st.button(
            "🔄 Regenerate & run",
            key="regen_run_top",
            help=f"Regenerate configs then run sweep: {out_dir}"
        )
    
    # Display target info below buttons
    st.caption(f"**Spec:** `config/sweep_specs/{sweep_name}.yaml` | **Sweep folder:** `{out_dir}`")
    
    return save_spec_btn, gen_configs_btn, run_sweep_btn, regen_run_btn


def sweep_actions(state: SweepState, feedback_placeholder: Any, 
                  save_clicked: bool, generate_clicked: bool, run_clicked: bool, regen_run_clicked: bool,
                  output_container: Any | None = None) -> None:
    """
    Handle sweep action button clicks.
    
    Args:
        state: Current sweep state with all parameters
        feedback_placeholder: Streamlit placeholder for feedback messages
        save_clicked: Whether save button was clicked
        generate_clicked: Whether generate button was clicked
        run_clicked: Whether run existing button was clicked
        regen_run_clicked: Whether regenerate & run button was clicked
    """
    
    # Prepare paths
    spec_dir = Path("config/sweep_specs")
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{state.sweep_name}.yaml"
    
    # Prepare preview dict for saving
    preview = {
        "stage": state.stage,
        "out_dir": state.out_dir,
        "axes": state.get_axes_dict(),
        "base_params": state.get_base_params_dict(),
    }
    
    # Helper: render an inline overwrite confirmation UI
    def prompt_overwrite(kind: str, target: Path, details: str | None = None) -> tuple[bool, bool]:
        box = feedback_placeholder.container()
        if kind == "spec":
            title = "Spec"
        elif kind == "sweep":
            title = "Sweep"
        elif kind in ("spec_sweep", "sweep_spec"):
            title = "Spec and sweep"
        else:
            title = kind.title()
        msg = f"⚠️ {title} target already exists: {target}. Are you sure you want to overwrite?"
        if details:
            msg += f"\n{details}"
        box.warning(msg)
        c1, c2 = box.columns(2)
        confirm = c1.button("✅ Yes, overwrite", key=f"confirm_overwrite_{kind}_{str(target)}")
        cancel = c2.button("✖️ Cancel", key=f"cancel_overwrite_{kind}_{str(target)}")
        return confirm, cancel

    # Session state for pending prompts
    if "pending_overwrite" not in st.session_state:
        st.session_state["pending_overwrite"] = None

    # If a prompt is pending, render it first and act on user choice
    pending = st.session_state.get("pending_overwrite")
    if pending:
        kind = pending.get("kind")  # 'spec' or 'sweep'
        target = Path(pending.get("target", ""))
        details = pending.get("details")
        confirm, cancel = prompt_overwrite(kind, target, details)
        if confirm:
            # Execute the stored action based on intent (save vs generate vs regen_run)
            intent = pending.get("op") or pending.get("payload", {}).get("op")
            if intent == "save":
                # Save spec only
                with open(target, "w", encoding="utf-8") as f:
                    yaml.safe_dump(pending["payload"], f, sort_keys=False)
                feedback_placeholder.success(f"✅ Saved spec to {target}")
            elif intent in ("generate", "regen_run"):
                # Generate flow: update spec, then clean and regenerate sweep configs
                spec_p = pending["payload"].get("spec_path")
                if spec_p:
                    try:
                        spec_doc = {
                            "stage": pending["payload"]["stage"],
                            "out_dir": pending["payload"]["out_dir"],
                            "axes": pending["payload"]["axes"],
                            "base_params": pending["payload"]["base_params"],
                        }
                        with open(spec_p, "w", encoding="utf-8") as f:
                            yaml.safe_dump(spec_doc, f, sort_keys=False)
                    except Exception as _e:
                        feedback_placeholder.warning(f"Could not update spec file: {_e}")
                # Clean existing YAMLs in the sweep directory before regenerating
                try:
                    sweep_dir = Path(pending["payload"]["out_dir"])
                    if sweep_dir.exists():
                        for p in sweep_dir.glob("*.yaml"):
                            p.unlink()
                except Exception as _e:
                    feedback_placeholder.warning(f"Could not remove some existing YAMLs in {sweep_dir}: {_e}")
                from birdnet_custom_classifier_suite.sweeps.sweep_generator import generate_sweep
                generate_sweep(
                    stage=pending["payload"]["stage"],
                    out_dir=pending["payload"]["out_dir"],
                    axes=pending["payload"]["axes"],
                    base_params=pending["payload"]["base_params"],
                )
                feedback_placeholder.success(f"✅ Generated configs at {pending['payload']['out_dir']}")
                
                # If regen_run, also run the sweep (stream live output)
                if intent == "regen_run":
                    out_dir_path = Path(pending["payload"]["out_dir"])
                    total = len([p for p in out_dir_path.glob("*.yaml") if p.name != "base.yaml"]) if out_dir_path.exists() else 0
                    if 'output_container' in pending and pending['output_container'] is not None:
                        with pending['output_container']:
                            status_box = st.empty()
                            progress_box = st.progress(0)
                            log_area = st.empty()
                    else:
                        status_box = st.empty()
                        progress_box = st.progress(0)
                        log_area = st.empty()

                    cmd = [
                        sys.executable,
                        "-m",
                        "birdnet_custom_classifier_suite.sweeps.run_sweep",
                        str(pending["payload"]["out_dir"]),
                        "--base-config",
                        str((Path(pending["payload"]["out_dir"]) / "base.yaml").as_posix()),
                        "--experiments-root",
                        "experiments",
                    ]
                    status_box.info("Running: " + " ".join(cmd))

                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                    completed = successes = failures = 0
                    current_exp = None
                    log_lines = []
                    last_ui = 0.0
                    exp_re = re.compile(r"^=== Running experiment: (.+) ===")
                    ok_re = re.compile(r"^Success:\s*(.+)")
                    fail_re = re.compile(r"^Failed:\s*(.+)")

                    for line in proc.stdout:
                        line = line.rstrip("\n")
                        log_lines.append(line)
                        m = exp_re.match(line)
                        if m:
                            current_exp = m.group(1)
                        if ok_re.match(line):
                            successes += 1
                            completed += 1
                        elif fail_re.match(line):
                            failures += 1
                            completed += 1

                        now = time.time()
                        if now - last_ui > 0.1:
                            progress = int((completed / max(total, 1)) * 100)
                            progress_box.progress(min(progress, 100))
                            log_area.text_area("Run log", value="\n".join(log_lines[-500:]), height=420, disabled=True)
                            status_box.info(f"Running… {completed}/{total} done | ✓ {successes} • ✗ {failures}" + (f" | current: {current_exp}" if current_exp else ""))
                            last_ui = now

                    ret = proc.wait()
                    progress_box.progress(100)
                    log_area.text_area("Run log", value="\n".join(log_lines[-800:]), height=420, disabled=True)
                    if ret != 0:
                        feedback_placeholder.error(f"❌ Sweep run failed with exit code {ret}. See logs above.")
                    else:
                        feedback_placeholder.success(f"✅ Sweep complete: {successes} succeeded, {failures} failed.")
            st.session_state["pending_overwrite"] = None
            return
        elif cancel:
            st.session_state["pending_overwrite"] = None
            feedback_placeholder.info("❎ Overwrite canceled.")
            return

    # Handle button actions (no pending prompt)
    if save_clicked:
        spec_dir.mkdir(parents=True, exist_ok=True)
        # If spec exists, open a confirmation prompt
        if spec_path.exists():
            st.session_state["pending_overwrite"] = {
                "kind": "spec",
                "target": str(spec_path),
                "payload": preview,
                "op": "save",
                "details": None,
            }
            # Render prompt now; user confirms or cancels on next interaction
            confirm, cancel = prompt_overwrite("spec", spec_path)
            return
        # Proceed with save (fresh)
        with open(spec_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(preview, f, sort_keys=False)
        feedback_placeholder.success(f"✅ Saved spec to {spec_path}")
    
    if generate_clicked:
        try:
            from birdnet_custom_classifier_suite.sweeps.sweep_generator import generate_sweep
            out_dir_path = Path(state.out_dir)
            spec_dir = Path("config/sweep_specs")
            spec_dir.mkdir(parents=True, exist_ok=True)
            spec_path = spec_dir / f"{state.sweep_name}.yaml"

            # Check for existing artifacts BEFORE creating or modifying anything
            existing_yaml = list(out_dir_path.glob("*.yaml")) if out_dir_path.exists() else []
            has_existing_sweep = bool(existing_yaml)
            has_existing_spec = spec_path.exists()

            if has_existing_sweep or has_existing_spec:
                details_list = []
                if has_existing_sweep:
                    details_list.append(f"{len(existing_yaml)} sweep YAML config(s)")
                if has_existing_spec:
                    details_list.append(f"Spec will be updated: {spec_path}")
                detail_str = ", ".join(details_list) if details_list else None
                kind = "spec_sweep" if has_existing_spec and has_existing_sweep else ("spec" if has_existing_spec else "sweep")
                target_for_prompt = spec_path if kind == "spec" else out_dir_path
                st.session_state["pending_overwrite"] = {
                    "kind": kind,
                    "target": str(target_for_prompt),
                    "payload": {
                        "stage": state.stage,
                        "out_dir": state.out_dir,
                        "axes": state.get_axes_dict(),
                        "base_params": state.get_base_params_dict(),
                        "spec_path": str(spec_path),
                        "op": "generate",
                    },
                    "details": detail_str,
                }
                # Render prompt inline; perform no side effects until confirmed
                confirm, cancel = prompt_overwrite(kind, target_for_prompt, detail_str)
                return

            # Fresh generation path: write spec first, then generate configs
            with open(spec_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({
                    "stage": state.stage,
                    "out_dir": state.out_dir,
                    "axes": state.get_axes_dict(),
                    "base_params": state.get_base_params_dict(),
                }, f, sort_keys=False)
            generate_sweep(
                stage=state.stage,
                out_dir=state.out_dir,
                axes=state.get_axes_dict(),
                base_params=state.get_base_params_dict()
            )
            feedback_placeholder.success(f"✅ Generated configs at {state.out_dir}")
        except Exception as e:
            feedback_placeholder.error(f"❌ Failed to generate sweep: {e}")
    
    if run_clicked:
        try:
            out_dir_path = Path(state.out_dir)

            # Check if configs exist
            experiment_yamls = sorted(p for p in out_dir_path.glob("*.yaml") if p.name != "base.yaml") if out_dir_path.exists() else []
            if not experiment_yamls:
                feedback_placeholder.error(f"❌ No experiment configs found in {state.out_dir}. Use 'Generate configs' or 'Regenerate & run' first.")
                return

            total = len(experiment_yamls)
            # Render logs/progress just below the buttons if a container is provided
            if output_container is not None:
                with output_container:
                    status_box = st.empty()
                    progress_box = st.progress(0)
                    log_area = st.empty()
            else:
                status_box = st.empty()
                progress_box = st.progress(0)
                log_area = st.empty()

            cmd = [
                sys.executable,
                "-m",
                "birdnet_custom_classifier_suite.sweeps.run_sweep",
                str(state.out_dir),
                "--base-config",
                str((Path(state.out_dir) / "base.yaml").as_posix()),
                "--experiments-root",
                "experiments",
            ]
            status_box.info("Running: " + " ".join(cmd))

            # Stream output live and track progress
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            completed = 0
            successes = 0
            failures = 0
            current_exp = None
            log_lines = []
            last_ui = 0.0
            exp_re = re.compile(r"^=== Running experiment: (.+) ===")
            ok_re = re.compile(r"^Success:\s*(.+)")
            fail_re = re.compile(r"^Failed:\s*(.+)")

            for line in proc.stdout:
                line = line.rstrip("\n")
                log_lines.append(line)
                m = exp_re.match(line)
                if m:
                    current_exp = m.group(1)
                if ok_re.match(line):
                    successes += 1
                    completed += 1
                elif fail_re.match(line):
                    failures += 1
                    completed += 1
                # Update UI (throttled) and use a fixed-height scrollable text area
                now = time.time()
                if now - last_ui > 0.1:
                    progress = int((completed / max(total, 1)) * 100)
                    progress_box.progress(min(progress, 100))
                    log_area.text_area("Run log", value="\n".join(log_lines[-500:]), height=420, disabled=True)
                    status_box.info(f"Running… {completed}/{total} done | ✓ {successes} • ✗ {failures}" + (f" | current: {current_exp}" if current_exp else ""))
                    last_ui = now

            ret = proc.wait()
            # Final update
            progress_box.progress(100)
            log_area.text_area("Run log", value="\n".join(log_lines[-800:]), height=420, disabled=True)
            if ret != 0:
                feedback_placeholder.error(f"❌ Sweep run failed with exit code {ret}. See logs above.")
            else:
                feedback_placeholder.success(f"✅ Sweep complete: {successes} succeeded, {failures} failed.")

            # Persist a summary in session state
            runs = st.session_state.get("sweep_runs", [])
            runs.append({
                "out_dir": str(out_dir_path),
                "total": total,
                "successes": successes,
                "failures": failures,
                "timestamp": time.time(),
            })
            st.session_state["sweep_runs"] = runs
        except Exception as e:
            feedback_placeholder.error(f"❌ Failed to run sweep: {e}")
    
    if regen_run_clicked:
        try:
            from birdnet_custom_classifier_suite.sweeps.sweep_generator import generate_sweep
            out_dir_path = Path(state.out_dir)
            spec_dir = Path("config/sweep_specs")
            spec_dir.mkdir(parents=True, exist_ok=True)
            spec_path = spec_dir / f"{state.sweep_name}.yaml"

            # Check for existing artifacts BEFORE creating or modifying anything
            existing_yaml = list(out_dir_path.glob("*.yaml")) if out_dir_path.exists() else []
            has_existing_sweep = bool(existing_yaml)
            has_existing_spec = spec_path.exists()

            if has_existing_sweep or has_existing_spec:
                details_list = []
                if has_existing_sweep:
                    details_list.append(f"{len(existing_yaml)} sweep YAML config(s)")
                if has_existing_spec:
                    details_list.append(f"Spec will be updated: {spec_path}")
                detail_str = ", ".join(details_list) if details_list else None
                kind = "spec_sweep" if has_existing_spec and has_existing_sweep else ("spec" if has_existing_spec else "sweep")
                target_for_prompt = spec_path if kind == "spec" else out_dir_path
                st.session_state["pending_overwrite"] = {
                    "kind": kind,
                    "target": str(target_for_prompt),
                    "payload": {
                        "stage": state.stage,
                        "out_dir": state.out_dir,
                        "axes": state.get_axes_dict(),
                        "base_params": state.get_base_params_dict(),
                        "spec_path": str(spec_path),
                        "op": "regen_run",
                    },
                    # Pass output container through so logs render under buttons after confirm
                    "output_container": output_container,
                    "details": detail_str,
                }
                # Render prompt inline; perform no side effects until confirmed
                confirm, cancel = prompt_overwrite(kind, target_for_prompt, detail_str)
                return

            # Fresh generation path: write spec first, then generate configs, then run
            with open(spec_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({
                    "stage": state.stage,
                    "out_dir": state.out_dir,
                    "axes": state.get_axes_dict(),
                    "base_params": state.get_base_params_dict(),
                }, f, sort_keys=False)
            generate_sweep(
                stage=state.stage,
                out_dir=state.out_dir,
                axes=state.get_axes_dict(),
                base_params=state.get_base_params_dict()
            )
            feedback_placeholder.success(f"✅ Generated configs at {state.out_dir}")

            # Now run the sweep with live streaming (reuse logic from run_clicked)
            total = len([p for p in out_dir_path.glob("*.yaml") if p.name != "base.yaml"]) if out_dir_path.exists() else 0
            if output_container is not None:
                with output_container:
                    status_box = st.empty()
                    progress_box = st.progress(0)
                    log_area = st.empty()
            else:
                status_box = st.empty()
                progress_box = st.progress(0)
                log_area = st.empty()

            cmd = [
                sys.executable,
                "-m",
                "birdnet_custom_classifier_suite.sweeps.run_sweep",
                str(state.out_dir),
                "--base-config",
                str((Path(state.out_dir) / "base.yaml").as_posix()),
                "--experiments-root",
                "experiments",
            ]
            status_box.info("Running: " + " ".join(cmd))
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            completed = successes = failures = 0
            current_exp = None
            log_lines = []
            last_ui = 0.0
            exp_re = re.compile(r"^=== Running experiment: (.+) ===")
            ok_re = re.compile(r"^Success:\s*(.+)")
            fail_re = re.compile(r"^Failed:\s*(.+)")
            for line in proc.stdout:
                line = line.rstrip("\n")
                log_lines.append(line)
                m = exp_re.match(line)
                if m:
                    current_exp = m.group(1)
                if ok_re.match(line):
                    successes += 1
                    completed += 1
                elif fail_re.match(line):
                    failures += 1
                    completed += 1
                now = time.time()
                if now - last_ui > 0.1:
                    progress = int((completed / max(total, 1)) * 100)
                    progress_box.progress(min(progress, 100))
                    log_area.text_area("Run log", value="\n".join(log_lines[-500:]), height=420, disabled=True)
                    status_box.info(f"Running… {completed}/{total} done | ✓ {successes} • ✗ {failures}" + (f" | current: {current_exp}" if current_exp else ""))
                    last_ui = now
            ret = proc.wait()
            progress_box.progress(100)
            log_area.text_area("Run log", value="\n".join(log_lines[-800:]), height=420, disabled=True)
            if ret != 0:
                feedback_placeholder.error(f"❌ Sweep run failed with exit code {ret}. See logs above.")
            else:
                feedback_placeholder.success(f"✅ Sweep complete: {successes} succeeded, {failures} failed.")
            runs = st.session_state.get("sweep_runs", [])
            runs.append({
                "out_dir": str(out_dir_path),
                "total": total,
                "successes": successes,
                "failures": failures,
                "timestamp": time.time(),
            })
            st.session_state["sweep_runs"] = runs
        except Exception as e:
            feedback_placeholder.error(f"❌ Failed to regenerate and run sweep: {e}")
