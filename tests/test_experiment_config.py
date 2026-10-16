"""Experiment configuration precedence and output containment regressions."""
import json
from pathlib import Path

import pytest

import run_jevlight as runner
from jevlight.config import setup_dataset_config


@pytest.fixture
def experiment_file(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "run_jevlight.py"))
    path = tmp_path / "config" / "jev" / "test.json"
    path.parent.mkdir(parents=True)
    return path


def test_config_and_cli_precedence(experiment_file):
    experiment_file.write_text(json.dumps({"agent": "jevlight", "count": 900, "seed": 7, "mock_jev": True, "speculative": True}))
    args = runner.parse_args(["--config", str(experiment_file), "--count", "60", "--no-mock_jev", "--speculative", "false"])
    assert (args.count, args.seed, args.mock_jev, args.speculative) == (60, 7, False, False)


@pytest.mark.parametrize("payload", [
    {"unknown": 2}, {"count": "60"}, {"count": True}, {"mock_jev": "false"},
    {"agent": "unknown-agent"}, {"jev_api_key": "secret"}, ["not", "object"],
])
def test_reject_invalid_or_secret_config(experiment_file, payload):
    experiment_file.write_text(json.dumps(payload))
    with pytest.raises(SystemExit):
        runner.parse_args(["--config", str(experiment_file)])


def test_reject_config_outside_model_directory(tmp_path):
    path = tmp_path / "params.json"
    path.write_text("{}")
    with pytest.raises(SystemExit):
        runner.parse_args(["--config", str(path)])


def test_results_contain_all_runtime_artifacts(experiment_file):
    experiment_file.write_text('{"memo": "smoke", "mock_jev": true}')
    args = runner.parse_args(["--config", str(experiment_file)])
    _, _, paths = runner.build_configs(args, setup_dataset_config("jinan"), "test-run")
    expected = Path("results/jev/smoke/test-run")
    assert Path(paths["PATH_TO_WORK_DIRECTORY"]) == expected
    assert Path(paths["PATH_TO_MODEL"]).is_relative_to(expected)
    assert Path(paths["PATH_TO_ERROR"]).is_relative_to(expected)
    args.evaluation_output_dir = "elsewhere"
    with pytest.raises(ValueError, match="inside results"):
        runner.build_configs(args, setup_dataset_config("jinan"), "test-run")
    args.evaluation_output_dir = None
    args.memo = "../escape"
    with pytest.raises(ValueError, match="single directory"):
        runner.build_configs(args, setup_dataset_config("jinan"), "test-run")


def test_legacy_cli_and_redacted_snapshot():
    args = runner.parse_args(["--mock_jev", "--count", "60", "--jev_api_key", "secret"])
    assert args.config is None and args.count == 60 and args.mock_jev
    assert "jev_api_key" not in runner.experiment_parameters(args)
    _, _, paths = runner.build_configs(args, setup_dataset_config("jinan"), "legacy")
    assert paths["PATH_TO_WORK_DIRECTORY"] == "results/jev/JevLight/legacy"


@pytest.mark.parametrize("agent", ["cojevlight", "collmlight"])
def test_cooperative_config_keeps_current_state_and_explicit_speculation(experiment_file, agent):
    experiment_file.write_text(json.dumps({"agent": agent, "cojev_history_steps": 0}))
    args = runner.parse_args(["--config", str(experiment_file), "--speculative", "false"])
    parameters = runner.experiment_parameters(args)
    assert parameters["cojev_history_steps"] == 0
    assert parameters["speculative"] is False
    for steps in ("-1", "2", "5"):
        with pytest.raises(SystemExit):
            runner.parse_args(["--cojev_history_steps", steps])
    experiment_file.write_text('{"cojev_history_steps": 5}')
    with pytest.raises(SystemExit):
        runner.parse_args(["--config", str(experiment_file)])
    assert runner.parse_args(["--config", str(experiment_file), "--cojev_history_steps", "0"]).cojev_history_steps == 0


def test_official_api_respects_frozen_connection_and_retry_settings(monkeypatch):
    import typesafe_sdk
    from jevlight.jev_client import JevClient

    captured = {}

    def client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", client)
    JevClient(transport="official", api_key="test-only", base_url="https://example.test",
              timeout=17.0, max_retries=0)
    assert captured["base_url"] == "https://example.test"
    assert captured["timeout"] == 17.0
    assert captured["retry"].max_retries == 0
    assert captured["retry"].timeout == 17.0


def test_official_probability_rounding_is_preserved():
    import copy
    from jevlight.backends.decision_model_service import validate_answers
    request = {"questions": {"phase": {"type": "choice", "criteria": dict.fromkeys("ABCD")}}}
    result = {"answers": {"phase": {"type": "choice", "choice": "A",
              "probabilities": {"A": 0.79, "B": 0.02, "C": 0.18, "D": 0.02}, "confidence": 0.7}}}
    original = copy.deepcopy(result)
    validate_answers(request, result, normalization_abs_tolerance=0.02000001)
    assert result == original
    with pytest.raises(ValueError, match="normalized"):
        validate_answers(request, result)
    result["answers"]["phase"]["probabilities"]["D"] = 0.05
    with pytest.raises(ValueError, match="normalized"):
        validate_answers(request, result, normalization_abs_tolerance=0.02000001)


@pytest.mark.parametrize("agent", ["cojevlight", "collmlight"])
def test_cooperative_network_packaging_is_rejected(experiment_file, agent):
    assert runner.parse_args(["--agent", agent]).packaging == "per_intersection"
    experiment_file.write_text(json.dumps({"agent": agent, "packaging": "network"}))
    with pytest.raises(SystemExit):
        runner.parse_args(["--config", str(experiment_file)])
    args = runner.parse_args(["--config", str(experiment_file), "--packaging", "per_intersection"])
    assert args.packaging == "per_intersection"
    with pytest.raises(SystemExit):
        runner.parse_args(["--agent", agent, "--packaging", "network"])


@pytest.mark.parametrize("agent", ["jevlight", "cojevlight"])
def test_single_congestion_switch_defaults_off_and_accepts_explicit_override(experiment_file, agent):
    assert runner.parse_args(["--agent", agent]).speculative is False
    assert runner.parse_args(["--agent", agent, "--speculative"]).speculative is True
    experiment_file.write_text(json.dumps({"agent": agent, "speculative": True}))
    assert runner.parse_args(["--config", str(experiment_file)]).speculative is True
    assert runner.parse_args(["--config", str(experiment_file), "--speculative", "false"]).speculative is False
    with pytest.raises(SystemExit):
        runner.parse_args(["--no_speculative"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--speculative", "invalid"])
    experiment_file.write_text('{"no_speculative": true}')
    with pytest.raises(SystemExit):
        runner.parse_args(["--config", str(experiment_file)])
    experiment_file.write_text('{"speculative": "false"}')
    with pytest.raises(SystemExit):
        runner.parse_args(["--config", str(experiment_file)])
