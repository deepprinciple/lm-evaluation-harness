"""SSE framing and durable capture shared by the SDE streaming adapters."""

import asyncio
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def stream_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# Failures before any upstream generation is returned: retrying them does not
# consume the attempt budget and remains bounded by the question deadline.
PRE_RESPONSE_RETRY_STATUSES = frozenset({429, 503, 529})
QUOTA_MARKERS = (
    "usage_limit_reached",
    "insufficient_quota",
    "usage limit exceeded",
    "hit your usage limit",
)


def pre_response_retryable(status: int, body: str) -> bool:
    lowered = body.lower()
    return status in PRE_RESPONSE_RETRY_STATUSES and not any(
        marker in lowered for marker in QUOTA_MARKERS
    )


def connect_failed(exc: BaseException) -> bool:
    import aiohttp

    return isinstance(
        exc, (aiohttp.ClientConnectorError, aiohttp.ConnectionTimeoutError)
    )


def retry_delay(retry_after: Any, retry: int) -> float:
    """Honor a numeric Retry-After, otherwise use capped exponential backoff."""
    try:
        seconds = float(retry_after)
    except (TypeError, ValueError):
        seconds = -1.0
    return min(seconds, 60.0) if seconds >= 0 else float(min(2**retry, 60))


def sse_events(body: str, allow_done: bool = False) -> Iterator[dict[str, Any]]:
    frames = body.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")
    for position, frame in enumerate(frames):
        data = []
        event_name = None
        for line in frame.split("\n"):
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif line.startswith("event:"):
                event_name = line[6:].strip()
        if not data:
            continue
        if position == len(frames) - 1:
            raise ValueError("unterminated SSE frame")
        encoded = "\n".join(data)
        if allow_done and encoded == "[DONE]":
            yield {"type": "[DONE]"}
            continue
        event = json.loads(encoded)
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("SSE data must be a typed object")
        if event_name is not None and event_name != event["type"]:
            raise ValueError("SSE event/data type mismatch")
        yield event


async def capture_stream(
    session: Any,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    key: str,
    attempt: dict[str, Any],
) -> None:
    import aiohttp

    body_path = Path(attempt["record_path"]).with_suffix(
        f".attempt{attempt['attempt']:02d}.sse"
    )
    attempt["body_path"] = str(body_path)
    chunks = []
    pending = b""
    with body_path.open("w", encoding="utf-8") as raw:
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                attempt["http_status"] = response.status
                async for chunk in response.content.iter_chunked(65536):
                    chunks.append(chunk)
                    pending += chunk
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        raw.write(
                            line.decode("utf-8", errors="replace").replace(
                                key, "<redacted>"
                            )
                            + "\n"
                        )
                    raw.flush()
                if response.status != 200:
                    attempt["retry_after"] = response.headers.get("Retry-After")
            code = attempt["http_status"]
            if code != 200:
                attempt["status"] = (
                    "transport_failed"
                    if code in (408, 429) or code >= 500
                    else "http_error"
                )
                if pre_response_retryable(
                    code, b"".join(chunks).decode("utf-8", errors="replace")
                ):
                    attempt["budget_exempt"] = True
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            attempt.update(status="transport_failed", error=type(exc).__name__)
            if connect_failed(exc):
                attempt["budget_exempt"] = True
        except asyncio.CancelledError:
            attempt.update(status="transport_failed", error="CancelledError")
            raise
        finally:
            raw.write(
                pending.decode("utf-8", errors="replace").replace(key, "<redacted>")
            )
            raw.flush()
            os.fsync(raw.fileno())
            captured = b"".join(chunks)
            try:
                body = captured.decode("utf-8")
            except UnicodeDecodeError:
                body = captured.decode("utf-8", errors="replace")
                if attempt["status"] == "running":
                    attempt.update(status="protocol_error", error="UnicodeDecodeError")
            attempt["body"] = body.replace(key, "<redacted>")
