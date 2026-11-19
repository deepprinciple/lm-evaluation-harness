"""CLI wrapper tests; no model calls."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_sde_question.sh"
GROUPS = {
    "flash": "sde-chat-completions",
    "opus5.5": "sde-messages",
    "astra": "sde-responses",
    "flash@claude_code": "sde-claude-code",
    "opus5.5@claude_code": "sde-claude-code",
    "flash@codex": "sde-codex",
    "astra@codex": "sde-codex",
    "astra@codex_subscription": "sde-codex",
}


@pytest.fixture
def runner(tmp_path):
    script = tmp_path / "scripts/run_sde_question.sh"
    script.parent.mkdir()
    shutil.copyfile(SCRIPT, script)
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    # Dry runs resolve paths only; these files are never executed.
    env.update(NODE_BIN="/test/node", TOOL_PYTHON="/test/python")

    def run(*args, **overrides):
        return subprocess.run(
            ["bash", str(script), *args],
            env={**env, **overrides},
            text=True,
            capture_output=True,
            check=False,
        )

    return run


@pytest.mark.parametrize("group,adapter", GROUPS.items())
@pytest.mark.parametrize("mode", ["smoke", "full"])
@pytest.mark.parametrize("repeat", [1, 6])
def test_group_defaults(runner, group, adapter, mode, repeat):
    result = runner(
        "--group",
        group,
        "--output",
        "results/test",
        "--mode",
        mode,
        "--repeat",
        str(repeat),
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)

    def value(flag):
        return command[command.index(flag) + 1]

    args = dict(item.split("=", 1) for item in value("--model_args").split(","))
    concurrency = 1 if mode == "smoke" else (32 if "@" not in group else 16)
    assert value("--model") == adapter
    assert args["num_concurrent"] == str(concurrency)
    assert args["timeout"] == ("600" if mode == "smoke" else "3600")
    assert args["max_attempts"] == ("1" if mode == "smoke" else "3")
    assert args["max_tokens"] == "128000"
    assert value("--gen_kwargs") == "max_tokens=128000"
    assert value("--seed") == "1"
    assert value("--repeats") == str(repeat)
    assert ("--limit" in command) == (mode == "smoke")
    assert args["record_dir"] == value("--output_path") + "/raw"
    assert "--apply_chat_template" in command
    assert "--log_samples" in command
    assert args["resume"] == "false"


def test_explicit_overrides(runner):
    result = runner(
        "--group",
        "flash",
        "--mode",
        "full",
        "--output",
        "results/custom run",
        "--dry-run",
        SDE_CONCURRENCY="8",
        SDE_TIMEOUT="900",
        SDE_ATTEMPTS="2",
        SDE_TASKS="science_physics",
    )
    assert result.returncode == 0
    command = shlex.split(result.stdout)
    assert command[command.index("--output_path") + 1] == "results/custom run"
    args = command[command.index("--model_args") + 1]
    assert "num_concurrent=8,timeout=900,max_attempts=2" in args


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("--group",),
        ("--group", "unknown"),
        ("--group", "flash", "--mode", "repeat6"),
        ("--group", "flash", "--repeat", "0"),
        ("--group", "flash", "--repeat", "-1"),
        ("--group", "flash", "--repeat", "text"),
        ("--group", "--repeat", "1"),
        ("flash",),
    ],
)
def test_invalid_selection(runner, args):
    assert runner(*args, "--output", "results/test").returncode != 0


def test_missing_key_fails_before_evaluation(runner):
    result = runner("--group", "flash", "--output", "results/test")
    assert result.returncode != 0
    assert "Export DEEPSEEK_API_KEY" in result.stderr


def test_credentials_not_printed_in_dry_run(runner):
    secret = "synthetic-secret-never-output"
    result = runner(
        "--group",
        "astra",
        "--output",
        "results/test",
        "--mode",
        "smoke",
        "--dry-run",
        OPENAI_API_KEY=secret,
    )
    assert result.returncode == 0
    assert secret not in result.stdout + result.stderr


@pytest.mark.parametrize("group", GROUPS)
def test_omitted_mode_and_repeat_mean_one_full_run(runner, group):
    result = runner("--group", group, "--output", "results/test", "--dry-run")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[command.index("--repeats") + 1] == "1"
    assert (
        command[command.index("--tasks") + 1]
        == "science_biology,science_chemistry,science_materials,science_physics"
    )
    assert "--limit" not in command


def test_option_order_is_independent(runner):
    result = runner(
        "--repeat",
        "2",
        "--dry-run",
        "--output",
        "results/test",
        "--mode",
        "smoke",
        "--group",
        "flash",
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[command.index("--repeats") + 1] == "2"
    assert command[command.index("--limit") + 1] == "1"


@pytest.mark.parametrize("extra", [(), ("--output",), ("--output", "")])
def test_output_must_be_explicit(runner, extra):
    result = runner("--group", "flash", "--dry-run", *extra)
    assert result.returncode != 0
    assert "--output" in result.stderr


@pytest.mark.parametrize("group", GROUPS)
def test_resume_flag_reaches_each_adapter(runner, group):
    result = runner(
        "--group", group, "--output", "results/test", "--resume", "--dry-run"
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    args = command[command.index("--model_args") + 1]
    assert "resume=true" in args


@pytest.mark.parametrize("group", ["astra", "astra@codex"])
def test_astra_defaults_to_openai_key_for_relay(runner, group):
    result = runner("--group", group, "--output", "results/test", "--dry-run")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    args = command[command.index("--model_args") + 1]
    assert "api_key_env=OPENAI_API_KEY" in args
    assert "base_url=https://mira.grecharge.ai/v1/responses" in args


def test_missing_astra_key_names_openai_variable(runner):
    result = runner("--group", "astra", "--output", "results/test")
    assert result.returncode != 0
    assert "Export OPENAI_API_KEY" in result.stderr


def test_astra_key_variable_can_still_be_overridden(runner):
    result = runner(
        "--group",
        "astra",
        "--output",
        "results/test",
        "--dry-run",
        SDE_ASTRA_KEY_ENV="ALTERNATE_KEY",
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert "api_key_env=ALTERNATE_KEY" in command[command.index("--model_args") + 1]
