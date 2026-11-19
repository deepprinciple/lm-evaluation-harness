import json

from lm_eval.sde_audit import audit


def write_record(root, record_id, doc_id, status="ok", extra=None):
    record = {
        "identity": {"task": "toy", "doc_id": doc_id, "config": extra or {}},
        "repeat_index": 0,
        "status": status,
        "attempts": [],
    }
    (root / f"{record_id}.json").write_text(json.dumps(record))


def test_audit_flags_benchmark_sources_and_record_access(tmp_path):
    leaked, clean, snooped = "a" * 64, "b" * 64, "c" * 64
    write_record(tmp_path, leaked, 0)
    # Main records hold harness config and are not model-facing evidence.
    write_record(tmp_path, clean, 1, extra={"cache": "/home/u/.cache/huggingface"})
    write_record(tmp_path, snooped, 2)

    attempt = tmp_path / f"{leaked}.attempt01.cc"
    attempt.mkdir()
    url = "https://huggingface.co/datasets/deep-principle/science-chemistry"
    (attempt / "trace.jsonl").write_text(
        json.dumps({"type": "assistant", "tool": "WebFetch", "url": url}) + "\n"
    )
    # Record paths written by the proxy must not count as record-dir access.
    (attempt / "api_001.json").write_text(
        json.dumps({"raw_path": str(attempt / "api_001.sse"), "payload": {}})
    )
    (attempt / "stderr.log").write_text(str(tmp_path))

    (tmp_path / f"{clean}.attempt01.sse").write_text('data: {"type":"ping"}\n\n')

    other = tmp_path / f"{snooped}.attempt01.codex"
    other.mkdir()
    (other / "trace.jsonl").write_text(
        json.dumps({"item": {"type": "command_execution", "command": f"ls {tmp_path}"}})
    )

    report = audit(tmp_path)
    assert report["records"] == 3
    assert report["scanned_files"] == 4
    rows = {row["record"]: row for row in report["flagged_records"]}
    assert set(rows) == {leaked, snooped}
    leak_hits = {hit["pattern"]: hit for hit in rows[leaked]["hits"]}
    assert set(leak_hits) == {"huggingface", "sde_datasets_org"}
    assert leak_hits["huggingface"]["file"] == f"{leaked}.attempt01.cc/trace.jsonl"
    assert "science-chemistry" in leak_hits["huggingface"]["context"]
    assert rows[leaked]["doc_id"] == 0 and rows[leaked]["task"] == "toy"
    assert [hit["pattern"] for hit in rows[snooped]["hits"]] == ["record_dir"]
