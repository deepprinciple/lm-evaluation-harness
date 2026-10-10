"""Recorded Codex CLI agents with API-key or isolated subscription authentication."""

import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from lm_eval.api.instance import GenerationResult, Instance
from lm_eval.api.registry import register_model
from lm_eval.models.sde_cc_proxy import ALLOWED_TOOLS, redact
from lm_eval.models.sde_chat import SDEChatCompletion, digest
from lm_eval.models.sde_claude_code import (
    SYSTEM,
    capture_pipe,
    stop_process,
    validate_tool_runtime,
)
from lm_eval.models.sde_codex_proxy import ResponsesProxy
from lm_eval.models.sde_stream import stream_sha256

DISABLED_FEATURES = (
    "memories",
    "plugins",
    "apps",
    "multi_agent",
    "multi_agent_v2",
    "shell_tool",
    "unified_exec",
    "view_image",
    "browser_use",
    "computer_use",
    "image_generation",
    "sleep_tool",
    "hooks",
    "code_mode",
    "code_mode_host",
    "enable_request_compression",
    "unbounded_connection_retries",
    "tool_search",
    "tool_search_always_defer_mcp_tools",
)
SUBSCRIPTION_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"


def read_subscription(path: Path) -> tuple[bytes, str, list[str]]:
    raw = path.read_bytes()
    auth = json.loads(raw)
    tokens = auth.get("tokens")
    if (
        auth.get("auth_mode") != "chatgpt"
        or auth.get("OPENAI_API_KEY")
        or not isinstance(tokens, dict)
        or not all(
            isinstance(tokens.get(n), str) and tokens[n]
            for n in ("access_token", "refresh_token", "account_id")
        )
    ):
        raise ValueError("Expected an explicit ChatGPT subscription auth file")
    return (
        raw,
        digest({"mode": "chatgpt", "account_id": tokens["account_id"]}),
        [
            value
            for name, value in tokens.items()
            if name != "account_id" and isinstance(value, str)
        ],
    )


def summarize_trace(body: str, answer: str, returncode: int) -> dict[str, Any]:
    completed = False
    started = False
    thread = None
    tools = []
    messages = []
    errors: list[dict[str, Any]] = []
    usage = {}
    status = "transport_failed"
    try:
        for line in body.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Non-object trace event")
            kind = event.get("type")
            if completed:
                raise ValueError("Event after completed turn")
            if kind == "thread.started":
                if thread is not None or not isinstance(event.get("thread_id"), str):
                    raise ValueError("Invalid thread initialization")
                thread = event["thread_id"]
            elif kind == "turn.started":
                if started or thread is None:
                    raise ValueError("Unexpected turn initialization")
                started = True
            elif kind == "turn.completed":
                if not started or errors or not isinstance(event.get("usage"), dict):
                    raise ValueError("Invalid completed turn")
                completed = True
                usage = event["usage"]
            elif kind in {"error", "turn.failed"}:
                errors.append(event)
            elif kind in {"item.started", "item.updated", "item.completed"}:
                item = event.get("item")
                if not isinstance(item, dict):
                    raise ValueError("Invalid trace item")
                item_type = item.get("type")
                if item_type == "mcp_tool_call":
                    if (
                        item.get("server") != "sde_tools"
                        or "mcp__sde_tools__" + item.get("tool", "")
                        not in ALLOWED_TOOLS
                    ):
                        status = "tool_profile_mismatch"
                    if kind == "item.completed":
                        tools.append(item)
                elif item_type not in {"agent_message", "reasoning", "error"}:
                    status = "tool_profile_mismatch"
                elif item_type == "error":
                    errors.append(event)
                elif item_type == "agent_message" and kind == "item.completed":
                    if not isinstance(item.get("text"), str):
                        raise ValueError("Non-text answer")
                    messages.append(item["text"])
            else:
                raise ValueError("Unexpected trace event")
        if status != "tool_profile_mismatch" and completed and returncode == 0:
            if not answer.strip():
                status = "empty"
            elif not messages or messages[-1] != answer:
                status = "protocol_error"
            else:
                status = "ok"
        if any(
            s in json.dumps(errors).lower()
            for s in (
                "usage_limit_reached",
                "insufficient_quota",
                "usage limit exceeded",
                "hit your usage limit",
            )
        ):
            status = "quota_exhausted"
        return {
            "status": status,
            "answer": answer,
            "usage": usage,
            "thread_id": thread,
            "turn_completed": completed,
            "tool_calls": tools,
            "errors": errors,
            "returncode": returncode,
        }
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        return {
            "status": "protocol_error",
            "answer": answer,
            "error": str(exc),
            "tool_calls": tools,
            "returncode": returncode,
        }


def validate_runtime(
    codex: Path, version: str, python: Path, node: Path | None
) -> dict:
    env = {
        "PATH": f"{codex.parent}:{python.parent}:{node.parent if node else '/usr/bin'}:/usr/bin:/bin",
        "LANG": "C.UTF-8",
    }
    actual = subprocess.check_output(
        [str(codex), "--version"], text=True, timeout=15, env=env
    ).strip()
    if actual != "codex-cli " + version:
        raise ValueError("Codex version does not match cli_version")
    with tempfile.TemporaryDirectory(prefix="sde-codex-preflight-") as tmp:
        env.update(HOME=tmp, CODEX_HOME=tmp)
        features = subprocess.check_output(
            [str(codex), "features", "list"], text=True, timeout=15, env=env
        )
    known = {line.split()[0] for line in features.splitlines() if line.split()}
    if not set(DISABLED_FEATURES) | {"skip_host_skill_discovery"} <= known:
        raise ValueError("Pinned Codex does not expose required feature switches")
    return {
        "cli_version": actual,
        "feature_inventory": features,
        **validate_tool_runtime(python),
    }


@register_model("sde-codex")
class SDECodex(SDEChatCompletion):
    def __init__(
        self,
        model: str,
        max_tokens: int,
        record_dir: str,
        codex_bin: str,
        cli_version: str,
        tool_python: str,
        model_catalog: str,
        base_url: str,
        api_key_env: str = "DEEPSEEK_API_KEY",
        auth_mode: str = "api_key",
        auth_file: str | None = None,
        effort: str = "max",
        num_concurrent: int = 4,
        timeout: float = 3600,
        max_attempts: int = 3,
        batch_size: int = 1,
        max_batch_size: Any = None,
        device: Any = None,
        node_bin: str | None = None,
    ):
        if auth_mode not in {"api_key", "chatgpt"}:
            raise ValueError("auth_mode must be api_key or chatgpt")
        if auth_mode == "chatgpt" and base_url != SUBSCRIPTION_ENDPOINT:
            raise ValueError(
                "Subscription credentials may only use the Codex subscription endpoint"
            )
        if auth_mode == "api_key" and auth_file is not None:
            raise ValueError("API profile must not read subscription credentials")
        self.auth_mode = auth_mode
        self.auth_file = Path(auth_file).resolve() if auth_file else None
        if auth_mode == "chatgpt" and self.auth_file is None:
            raise ValueError("Subscription profile requires an explicit auth_file")
        self.account_identity: str | None = None
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            record_dir=record_dir,
            base_url=base_url,
            api_key_env=api_key_env,
            num_concurrent=num_concurrent,
            timeout=timeout,
            max_attempts=max_attempts,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            device=device,
        )
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported Codex effort")
        for name in ("SERPAPI_API_KEY", "JINA_API_KEY"):
            if not os.environ.get(name, "").strip():
                raise ValueError(f"Tool key not found in {name}")
        self.codex = Path(codex_bin).resolve()
        self.tool_python = Path(tool_python).absolute()
        self.node = Path(node_bin).absolute() if node_bin else None
        catalog_bytes = Path(model_catalog).read_bytes()
        catalog = json.loads(catalog_bytes)
        entries = [entry for entry in catalog["models"] if entry.get("slug") == model]
        if len(entries) != 1:
            raise ValueError(
                "Expected exactly one pinned catalog entry for the requested model"
            )
        if effort not in {
            entry.get("effort")
            for entry in entries[0].get("supported_reasoning_levels", [])
        }:
            raise ValueError("Pinned catalog does not support the requested effort")
        # Freeze only this model; no opportunistic lookup/fallback to another model.
        self.catalog = {"models": [dict(entries[0])]}
        pinned_tool_mode = entries[0].get("tool_mode")
        catalog_normalizations = []
        if pinned_tool_mode == "code_mode_only":
            self.catalog["models"][0]["tool_mode"] = None
            catalog_normalizations.append("disabled_native_code_mode")
        runtime = validate_runtime(self.codex, cli_version, self.tool_python, self.node)
        self.config.pop("thinking")
        self.config.pop("reasoning_effort")
        self.config["runtime_sha256"] = self.config.pop("adapter_sha256")
        self.config.update(
            api_format="codex",
            effort=effort,
            auth_mode=auth_mode,
            account_identity=self.account_identity,
            codex_bin=str(self.codex),
            tool_python=str(self.tool_python),
            model_catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
            selected_catalog=self.catalog,
            pinned_tool_mode=pinned_tool_mode,
            catalog_normalizations=catalog_normalizations,
            **runtime,
            search_backend="serpapi",
            read_backend="jina",
            search_attempt_limit=3,
            native_tools=False,
            agent_turn_limit=None,
            system_append=SYSTEM,
            output_budget="proxy_injected_per_response",
            native_model_retries=0 if auth_mode == "api_key" else "builtin_proxy_gated",
            node_bin=str(self.node) if self.node else None,
            stream_sha256=stream_sha256(),
        )
        for name in (
            "sde_codex",
            "sde_codex_proxy",
            "sde_tools",
            "sde_sandbox",
            "sde_claude_code",
            "sde_cc_proxy",
            "sde_responses",
        ):
            self.config[name + "_sha256"] = hashlib.sha256(
                Path(__file__).with_name(name + ".py").read_bytes()
            ).hexdigest()
        self.config["cli_sha256"] = hashlib.sha256(self.codex.read_bytes()).hexdigest()
        if self.node:
            self.config["node_version"] = subprocess.check_output(
                [str(self.node), "--version"], text=True, timeout=15
            ).strip()
            self.config["node_sha256"] = hashlib.sha256(
                self.node.read_bytes()
            ).hexdigest()

    def validate_auth(self, api_key_env: str) -> None:
        if self.auth_mode == "api_key":
            super().validate_auth(api_key_env)
        else:
            assert self.auth_file is not None
            _, identity, _ = read_subscription(self.auth_file)
            if self.account_identity is not None and identity != self.account_identity:
                raise ValueError("Subscription account changed during the experiment")
            self.account_identity = identity

    def get_api_key(self) -> str:
        return super().get_api_key() if self.auth_mode == "api_key" else ""

    def create_payload(self, request: Instance) -> dict[str, Any]:
        messages, budget = self.chat_input(request)
        system = []
        conversation: list[dict[str, str]] = []
        for message in messages:
            if message["role"] == "system":
                if conversation:
                    raise ValueError("System messages must precede the conversation")
                system.append(message["content"])
            else:
                conversation.append(message)
        if not conversation or conversation[-1]["role"] != "user":
            raise ValueError("Codex input must end with a user question")
        prompt = (
            conversation[0]["content"]
            if len(conversation) == 1
            else (
                "Conversation supplied by the evaluation harness (role/content JSON):\n"
                + json.dumps(conversation, ensure_ascii=False)
            )
        )
        return {
            "model": self.model,
            "prompt": prompt,
            "input_messages": messages,
            "system_append": "\n\n".join([SYSTEM, *system]),
            "max_tokens": budget,
            "effort": self.config["effort"],
        }

    def command(
        self,
        directory: Path,
        quota: Path,
        system: str,
        proxy: ResponsesProxy,
        home: Path,
    ) -> list[str]:
        catalog = directory / "catalog.json"
        catalog.write_text(json.dumps(self.catalog, ensure_ascii=False))
        provider = "openai" if self.auth_mode == "chatgpt" else "sde_recorded"
        cfg = {
            "model_provider": provider,
            "model_catalog_json": str(catalog),
            "model_reasoning_effort": self.config["effort"],
            "web_search": "disabled",
            "approval_policy": "never",
            "agents.enabled": False,
            "project_root_markers": [],
            "project_doc_max_bytes": 0,
            "suppress_unstable_features_warning": True,
            "project_doc_fallback_filenames": [],
            "shell_environment_policy.inherit": "none",
            "developer_instructions": system,
            "cli_auth_credentials_store": "file",
            f"model_providers.{provider}.name": provider,
            f"model_providers.{provider}.base_url": proxy.url + "/" + proxy.token,
            f"model_providers.{provider}.wire_api": "responses",
            f"model_providers.{provider}.requires_openai_auth": self.auth_mode
            == "chatgpt",
            f"model_providers.{provider}.supports_websockets": False,
            f"model_providers.{provider}.request_max_retries": 0,
            f"model_providers.{provider}.stream_max_retries": 0,
            f"model_providers.{provider}.stream_idle_timeout_ms": int(
                self.config["timeout"] * 1000
            ),
            "mcp_servers.sde_tools.command": str(self.tool_python),
            "mcp_servers.sde_tools.args": [
                str(Path(__file__).with_name("sde_tools.py")),
                str(directory / "work/compute"),
                str(quota),
                str(directory / "tools.jsonl"),
            ],
            "mcp_servers.sde_tools.env_vars": [
                "SERPAPI_API_KEY",
                "JINA_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
            ],
            "mcp_servers.sde_tools.enabled_tools": [
                "web_search",
                "read_url",
                "run_shell",
            ],
            "mcp_servers.sde_tools.required": True,
            "mcp_servers.sde_tools.startup_timeout_sec": 30,
            "mcp_servers.sde_tools.tool_timeout_sec": 90,
            "mcp_servers.sde_tools.default_tools_approval_mode": "approve",
        }
        if self.auth_mode == "chatgpt":
            cfg = {
                key: value
                for key, value in cfg.items()
                if not key.startswith("model_providers.openai.")
            }
            cfg["openai_base_url"] = proxy.url + "/" + proxy.token
            cfg["forced_login_method"] = "chatgpt"
        else:
            cfg[f"model_providers.{provider}.env_key"] = "SDE_PROXY_TOKEN"
        # Isolated CODEX_HOME holds auth only, not personal settings/hooks/skills.
        cmd = [
            str(self.codex),
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "--json",
            "--model",
            self.model,
            "--sandbox",
            "read-only",
            "-C",
            str(directory / "work"),
            "-o",
            str(directory / "answer.txt"),
        ]
        for key, value in cfg.items():
            cmd.extend(["-c", key + "=" + json.dumps(value, ensure_ascii=False)])
        for feature in DISABLED_FEATURES:
            cmd.extend(["--disable", feature])
        cmd.extend(["--enable", "skip_host_skill_discovery"])
        return [*cmd, "-"]

    async def request_once(
        self, session: Any, payload: dict[str, Any], key: str, attempt: dict[str, Any]
    ) -> None:
        record = Path(attempt["record_path"])
        directory = record.with_suffix(f".attempt{attempt['attempt']:02}.codex")
        directory.mkdir(exist_ok=False)
        work = directory / "work"
        for name in ("tmp", "compute"):
            (work / name).mkdir(parents=True)
        quota = record.with_suffix(".search.json")
        secrets = [key, os.environ["SERPAPI_API_KEY"], os.environ["JINA_API_KEY"]]
        attempt.update(
            trace_path=str(directory / "trace.jsonl"),
            stderr_path=str(directory / "stderr.log"),
            tools_path=str(directory / "tools.jsonl"),
            search_quota_path=str(quota),
        )
        proc = None
        captures: list[asyncio.Task] = []
        cancelled = False
        proxy = ResponsesProxy(
            session,
            self.config["base_url"],
            self.model,
            self.config["effort"],
            payload["max_tokens"],
            key,
            directory,
            secrets,
            auth_mode=self.auth_mode,
        )
        try:
            # OAuth files must never be written into retained raw-result trees.
            with tempfile.TemporaryDirectory(prefix="sde-codex-auth-") as tmp:
                home = Path(tmp)
                codex_home = home / ".codex"
                codex_home.mkdir(mode=0o700)
                if self.auth_mode == "chatgpt":
                    self.validate_auth(self.config["api_key_env"])
                    assert self.auth_file is not None
                    raw, identity, tokens = read_subscription(self.auth_file)
                    if identity != self.account_identity:
                        raise ValueError("Subscription account changed")
                    secrets.extend(tokens)
                    proxy.secrets.extend(tokens)
                    auth = codex_home / "auth.json"
                    auth.touch(mode=0o600)
                    auth.write_bytes(raw)
                async with proxy:
                    cmd = self.command(
                        directory, quota, payload["system_append"], proxy, home
                    )
                    attempt["command"] = [redact(arg, [proxy.token]) for arg in cmd]
                    env = {
                        "HOME": str(home),
                        "CODEX_HOME": str(codex_home),
                        "TMPDIR": str(work / "tmp"),
                        "LANG": "C.UTF-8",
                        "PATH": f"{self.codex.parent}:{self.tool_python.parent}:{self.node.parent if self.node else '/usr/bin'}:/usr/bin:/bin",
                        "SDE_PROXY_TOKEN": proxy.token,
                        "ANTHROPIC_AUTH_TOKEN": proxy.token,
                        "SERPAPI_API_KEY": os.environ["SERPAPI_API_KEY"],
                        "JINA_API_KEY": os.environ["JINA_API_KEY"],
                    }
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=work,
                        env=env,
                        start_new_session=True,
                        limit=32 * 1024**2,
                    )
                    attempt["pid"] = proc.pid
                    captures = [
                        asyncio.create_task(
                            capture_pipe(reader, path, proxy.secrets + [proxy.token])
                        )
                        for reader, path in (
                            (proc.stdout, directory / "trace.jsonl"),
                            (proc.stderr, directory / "stderr.log"),
                        )
                    ]
                    assert proc.stdin is not None
                    proc.stdin.write(payload["prompt"].encode())
                    await proc.stdin.drain()
                    proc.stdin.close()
                    waiters = [
                        asyncio.create_task(proc.wait()),
                        asyncio.create_task(proxy.failure.wait()),
                    ]
                    try:
                        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                        await stop_process(proc)
                        await asyncio.gather(*captures)
                    finally:
                        for waiter in waiters:
                            waiter.cancel()
                        await asyncio.gather(*waiters, return_exceptions=True)
                    answer_path = directory / "answer.txt"
                    answer = redact(
                        answer_path.read_text() if answer_path.exists() else "",
                        proxy.secrets + [proxy.token],
                    )
                    if answer_path.exists():
                        answer_path.write_text(answer)
                    assert proc.returncode is not None
                    attempt.update(
                        summarize_trace(
                            (directory / "trace.jsonl").read_text(),
                            answer.rstrip("\n"),
                            proc.returncode,
                        )
                    )
        except OSError as exc:
            attempt.update(status="transport_failed", error=type(exc).__name__)
        except ValueError as exc:
            attempt.update(status="auth_profile_mismatch", error=str(exc))
        except asyncio.CancelledError:
            cancelled = True
            attempt.update(status="transport_failed", error="Cancelled")
            raise
        finally:
            if proc is not None:
                await stop_process(proc)
            if captures:
                await asyncio.gather(*captures, return_exceptions=True)
            for name in ("trace.jsonl", "stderr.log", "answer.txt"):
                path = directory / name
                if path.exists():
                    path.write_text(
                        redact(path.read_text(), proxy.secrets + [proxy.token])
                    )
            attempt["api_calls"] = json.loads(
                redact(
                    json.dumps(proxy.calls, ensure_ascii=False),
                    proxy.secrets + [proxy.token],
                )
            )
            attempt["cli_controls"] = json.loads(
                redact(json.dumps(proxy.controls), proxy.secrets + [proxy.token])
            )
            failed = [call for call in proxy.calls if call["status"] != "ok"]
            if failed and not cancelled:
                attempt["status"] = failed[0]["status"]
                if failed[0].get("http_status") in (401, 403):
                    attempt["http_status"] = failed[0]["http_status"]
            elif attempt.get("status") == "ok" and not proxy.calls:
                attempt["status"] = "protocol_error"
            # If an API terminal answer exists, the CLI final must agree exactly.
            if attempt.get("status") == "ok" and proxy.calls:
                last = proxy.calls[-1]
                if last.get("answer", "").rstrip("\n") != attempt.get("answer", ""):
                    attempt["status"] = "protocol_error"
            attempt["response"] = {
                "model": (proxy.calls[-1].get("response") or {}).get("model")
                if proxy.calls
                else None,
                "usage": attempt.get("usage") or {},
            }
            attempt["elapsed_seconds"] = time.time() - attempt["started_at"]

    async def generate_one(self, *args, **kwargs) -> GenerationResult:
        result = await super().generate_one(*args, **kwargs)
        if result.metadata["status"] == "quota_exhausted":
            raise RuntimeError(
                "Subscription quota exhausted; record saved, batch stopped"
            )
        return result

    def result(self, record: dict[str, Any], path: Path) -> GenerationResult:
        output = super().result(record, path)
        last = record["attempts"][-1] if record["attempts"] else {}
        output.metadata.update(
            cli_version=self.config["cli_version"],
            auth_mode=self.auth_mode,
            tool_calls=last.get("tool_calls", []),
            agent_metadata=[
                {
                    key: attempt.get(key)
                    for key in (
                        "trace_path",
                        "stderr_path",
                        "tools_path",
                        "search_quota_path",
                        "returncode",
                        "api_calls",
                        "cli_controls",
                    )
                }
                for attempt in record["attempts"]
            ],
        )
        return output
