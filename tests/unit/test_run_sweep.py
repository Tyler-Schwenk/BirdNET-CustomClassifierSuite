import subprocess
import sys
from types import SimpleNamespace

import yaml


def test_get_experiment_name_and_invalid(tmp_path):
    from birdnet_custom_classifier_suite.sweeps.run_sweep import get_experiment_name

    cfg = tmp_path / "a.yaml"
    cfg.write_text(yaml.safe_dump({"experiment": {"name": "myexp"}}))
    assert get_experiment_name(cfg) == "myexp"

    # Missing experiment.name -> None
    cfg2 = tmp_path / "b.yaml"
    cfg2.write_text(yaml.safe_dump({"nope": 1}))
    assert get_experiment_name(cfg2) is None

    # Invalid YAML returns None (handled safely)
    cfg3 = tmp_path / "c.yaml"
    cfg3.write_text("::invalid:::")
    assert get_experiment_name(cfg3) is None


def test_run_pipeline_invokes_subprocess(monkeypatch, tmp_path, capsys):
    from birdnet_custom_classifier_suite.sweeps.run_sweep import run_pipeline

    cfg = tmp_path / "cfg.yaml"
    base = tmp_path / "base.yaml"
    cfg.write_text("x: 1")
    base.write_text("y: 2")

    calls = {}

    class DummyCompleted:
        def __init__(self):
            self.returncode = 0

    def fake_run(cmd, check=True):
        # basic checks about the constructed command
        assert isinstance(cmd, (list, tuple))
        assert cmd[0] == sys.executable
        # should call the pipeline module path string somewhere in the argv
        assert any("birdnet_custom_classifier_suite.pipeline.pipeline" in str(x) for x in cmd)
        # should pass both config flags
        assert "--base-config" in cmd
        assert "--override-config" in cmd
        calls["ok"] = True
        return DummyCompleted()

    # Patch the subprocess.run symbol where it's used in the module under test
    import birdnet_custom_classifier_suite.sweeps.run_sweep as rs
    S = SimpleNamespace(run=fake_run, CalledProcessError=subprocess.CalledProcessError)
    monkeypatch.setattr(rs, "subprocess", S)

    ok = run_pipeline(cfg, base, verbose=True)
    captured = capsys.readouterr()
    assert ok is True
    assert calls.get("ok")
    assert "Success" in captured.out

    # Simulate a CalledProcessError -> should return False
    def fake_run_fail(cmd, check=True):
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

    # Replace the run function on the same fake subprocess object so the
    # module's exception class reference matches the raised error type.
    def fake_run_fail(cmd, check=True):
        raise S.CalledProcessError(returncode=2, cmd=cmd)

    S.run = fake_run_fail
    ok2 = run_pipeline(cfg, base, verbose=False)
    captured2 = capsys.readouterr()
    assert ok2 is False
    assert "Failed" in captured2.out
