"""Exclude recognizable benchmark sources from readable tool returns."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import re
from typing import Any
from urllib.parse import unquote


POLICY = "benchmark-source-exclusion-v1"
NOTICE = (
    "Tool content withheld: benchmark or answer-key source. "
    "Use independent scientific sources or computation instead."
)
SOURCE_RULES = {
    "sde_dataset": re.compile(
        r"deep[-_]?principle[/\\]+science_(?:biology|chemistry|materials|physics)\b"
        r"|[/\\]science_(?:biology|chemistry|materials|physics)(?:[/\\\s\"'?#]|$)",
        re.IGNORECASE,
    ),
    "sde_mirror": re.compile(r"SCILLM[-_]benchmarks", re.IGNORECASE),
    "gpqa_answers": re.compile(
        r"(?:huggingface\.co|hf\.co)/[^\s\"<>]*gpqa"
        r"|GPQA[-_ ]diamond|gpqa198(?:[/\\.\s]|$)",
        re.IGNORECASE,
    ),
    "benchmark_repository": re.compile(
        r"(?:github\.com|raw\.githubusercontent\.com)/"
        r"(?:deep[-_]?principle|HowieHwong)/(?:lm-evaluation-harness|sde-harness)\b"
        r"|lm[-_]evaluation[-_]harness[/\\]+(?:results(?:_final)?|lm_eval[/\\]+tasks[/\\]+science)\b",
        re.IGNORECASE,
    ),
    "answer_key_claim": re.compile(
        r"(?:benchmark|dataset)(?:['’]s)?[^.\n]{0,40}"
        r"(?:answer(?:\s+key)?|ground[- ]truth)\s*(?:\([^)]*\))?\s*(?:is\b|:|=)"
        r"|recorded answer(?:\s+key)?\s*(?:is\b|:|=)",
        re.IGNORECASE,
    ),
}


def readable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: readable(child)
            for key, child in value.items()
            if key not in {"encrypted_output", "encrypted_content"}
        }
    if isinstance(value, list):
        return [readable(child) for child in value]
    return value


def source_rules(value: Any) -> list[str]:
    value = readable(value)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\\u002[fF]", "/", text).replace("\\/", "/")
    for _ in range(2):
        text = html.unescape(unquote(text))
    return [name for name, pattern in SOURCE_RULES.items() if pattern.search(text)]


def filter_tool_returns(body: dict[str, Any]) -> tuple[dict[str, Any], list[dict]]:
    """Only tool outputs, never initial prompts, instructions or assistant text."""
    edits: list[tuple[str, int, int | None, Any]] = []
    for index, message in enumerate((body.get("messages") or [])):
        if (
            not isinstance(message, dict)
            or message.get("role") != "user"
            or not isinstance(message.get("content"), list)
        ):
            continue
        for content_index, block in enumerate(message["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                edits.append(
                    ("messages", index, content_index, block.get("content", ""))
                )
    for index, item in enumerate((body.get("input") or [])):
        if isinstance(item, dict) and item.get("type") in {
            "function_call_output",
            "custom_tool_call_output",
        }:
            edits.append(("input", index, None, item.get("output", "")))
    filtered = []
    result = body
    for field, index, block_index, value in edits:
        rules = source_rules(value)
        if not rules:
            continue
        if result is body:
            result = copy.deepcopy(body)
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        filtered.append(
            {
                "path": f"{field}[{index}]"
                + (f".content[{block_index}]" if block_index is not None else ""),
                "rules": rules,
                "original_content": value,
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
        target = result[field][index]
        if block_index is not None:
            target = target["content"][block_index]
            target["content"], target["is_error"] = NOTICE, True
        else:
            target["output"] = NOTICE
    return result, filtered
