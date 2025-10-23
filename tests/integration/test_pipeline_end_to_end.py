import pytest
from pathlib import Path


@pytest.mark.integration
def test_pipeline_flow_with_mocks(tmp_path, monkeypatch):
    """Run a micro pipeline flow but mock training/inference/eval to be no-ops.

    This test ensures that `pipeline.main()` follows the happy-path control flow
    (build package -> train or skip -> inference -> evaluate -> collect) without
    launching heavy external tools. It's marked as integration and should be
    run explicitly.
    """
    from birdnet_custom_classifier_suite.pipeline import pipeline

    # Prepare a minimal config that the pipeline can load
    cfg = {
        "experiment": {"name": "int_test_exp", "seed": 42},
        "training": {"batch_size": 1},
        "training_args": {},
        "dataset": {"audio_root": str(tmp_path / "AudioData")},
    }
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text("{}")
    override_cfg = tmp_path / "override.yaml"
    override_cfg.write_text("experiment:\n  name: int_test_exp\n")

    # Monkeypatch functions that would run heavy work
    monkeypatch.setattr("birdnet_custom_classifier_suite.pipeline.make_training_package.run_from_config", lambda cfg, verbose=False: None)
    monkeypatch.setattr("birdnet_custom_classifier_suite.pipeline.evaluate_results.run_evaluation", lambda exp_dir: None)
    monkeypatch.setattr("birdnet_custom_classifier_suite.pipeline.collect_experiments.collect_experiments", lambda root, out: None)

    # Ensure the pipeline uses a tmp experiments directory and create a dummy model
    exp_dir = tmp_path / "experiments" / "int_test_exp"
    model_dir = exp_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    dummy_model = model_dir / "dummy.tflite"
    dummy_model.write_text("fake-tflite")

    # Monkeypatch pipeline.get_experiment_dir so pipeline uses our tmp experiments
    monkeypatch.setattr(pipeline, "get_experiment_dir", lambda cfg: exp_dir)

    # Monkeypatch subprocess.run used during inference to return success
    class DummyCompleted:
        def __init__(self):
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd, capture_output=False, text=False, check=False):
        return DummyCompleted()

    monkeypatch.setattr("subprocess.run", fake_run)

    # Run pipeline.main with args pointing to our temporary config files
    import sys
    sys_argv = [
        "pipeline",
        "--base-config",
        str(base_cfg),
        "--override-config",
        str(override_cfg),
        "--skip-training",
    ]
    monkeypatch.setattr("sys.argv", sys_argv)

    # Execute main (should not raise)
    pipeline.main()
