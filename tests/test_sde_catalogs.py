"""Validate bundled experiment assets without personal caches or provider calls."""

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / "config/sde-catalogs"


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    "name,slug", [("flash", "deepseek-flash"), ("astra", "gpt-6-astra")]
)
def test_catalog_identity_and_pinned_checksum(name, slug):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    data = (ROOT / f"{name}.json").read_bytes()
    catalog = json.loads(data)
    assert len(catalog["models"]) == 1
    entry = catalog["models"][0]
    assert entry["slug"] == slug
    assert any(
        level["effort"] == "max" for level in entry["supported_reasoning_levels"]
    )
    assert hashlib.sha256(data).hexdigest() == manifest["catalogs"][name]["sha256"]
    assert not {"api_key", "access_token", "refresh_token"} & entry.keys()


def test_flash_and_astra_share_exact_native_model_messages():
    flash = json.loads((ROOT / "flash.json").read_text())["models"][0]
    astra = json.loads((ROOT / "astra.json").read_text())["models"][0]
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert "base_instructions" not in flash
    assert "base_instructions" not in astra
    assert flash["model_messages"] == astra["model_messages"]
    assert canonical_sha(flash["model_messages"]) == manifest["model_messages_sha256"]
    assert flash["model_messages"]["instructions_template"].startswith(
        "You are Codex, an agent based on GPT-6."
    )


def test_flash_retains_its_model_capability_profile():
    flash = json.loads((ROOT / "flash.json").read_text())["models"][0]
    manifest = json.loads((ROOT / "manifest.json").read_text())
    profile = {
        key: value
        for key, value in flash.items()
        if key not in {"base_instructions", "model_messages"}
    }
    assert canonical_sha(profile) == manifest["flash_source_profile_sha256"]
    assert flash["context_window"] == 1048576
    assert flash["tool_mode"] is None


def test_astra_snapshot_is_unmodified():
    data = (ROOT / "astra.json").read_bytes()
    assert (
        hashlib.sha256(data).hexdigest()
        == "efdc8fdf6b392a398150ef26915a9c6306cfd9d59a37d97c3c7a71141ba3804e"
    )
