"""Self-contained regressions for live native tool smoke failures."""

import copy
import shutil
import subprocess

import pytest

from lm_eval.models.sde_claude_code import prepare_tool_shells
from lm_eval.models.sde_codex import final_api_call
from lm_eval.models.sde_codex_proxy import inspect_stream, prepare_payload
from tests.test_sde_responses import response, stream


def test_hosted_search_adds_native_tool_without_changing_cli_tools_or_input():
    body = {
        "model": "test-model",
        "reasoning": {"effort": "max"},
        "stream": True,
        "input": [{"role": "user", "content": "question"}],
        "tools": [{"type": "custom", "name": "exec", "format": {"type": "text"}}],
        "tool_choice": "auto",
    }
    original = copy.deepcopy(body)
    result = prepare_payload(body, "test-model", "max", 128000, hosted_web_search=True)
    assert body == original
    assert result["tools"] == original["tools"] + [{"type": "web_search"}]
    assert result["input"] == original["input"]
    assert result["tool_choice"] == "auto"
    assert result["max_output_tokens"] == 128000
    assert prepare_payload(body, "test-model", "max", 128000)["tools"] == body["tools"]
    assert (
        prepare_payload(result, "test-model", "max", 128000, hosted_web_search=True)
        == result
    )


def test_final_answer_matches_cli_without_discarding_commentary_evidence():
    final = response("<answer>A</answer>")
    message = final["output"][-1]
    message["phase"] = "final_answer"
    commentary = copy.deepcopy(message)
    commentary.update(id="commentary", phase="commentary")
    commentary["content"][0]["text"] = "I will search first."
    final["output"].insert(0, commentary)
    wire = stream([{"type": "response.completed", "response": final}])
    parsed = inspect_stream(wire, "gpt-6-astra", "max", 128000)
    assert parsed["status"] == "ok"
    assert parsed["answer"] == "<answer>A</answer>"
    assert parsed["response"]["output"][0] == commentary
    parsed.update(index=1, original_payload={"client_metadata": {"thread_id": "root"}})
    assert final_api_call([parsed], "root", "<answer>A</answer>")[0] is parsed
    assert final_api_call([parsed], "other-thread", "<answer>A</answer>")[0] is None
    assert final_api_call([parsed], "root", "different answer")[0] is None
    incomplete = wire.replace("response.completed", "response.in_progress")
    assert inspect_stream(incomplete, "gpt-6-astra", "max", 128000)["status"] != "ok"
    duplicated = copy.deepcopy(message)
    duplicated["id"] = "other-final"
    final["output"].append(duplicated)
    assert (
        inspect_stream(
            stream([{"type": "response.completed", "response": final}]),
            "gpt-6-astra",
            "max",
        )["status"]
        == "protocol_error"
    )


@pytest.mark.parametrize("shell_name", ["bash", "zsh"])
def test_login_shell_uses_selected_tool_python(tmp_path, shell_name):
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip("Shell is not installed")
    home = tmp_path / "home"
    home.mkdir()
    bindir = tmp_path / "scientific env" / "bin"
    bindir.mkdir(parents=True)
    python = bindir / "python3"
    python.write_text("#!/bin/sh\nprintf scientific-python-runtime\n")
    python.chmod(0o755)
    prepare_tool_shells(home, python, None)
    result = subprocess.run(
        [shell, "-l", "-c", "python3"],
        env={"HOME": str(home), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "scientific-python-runtime"
