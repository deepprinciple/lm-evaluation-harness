"""Post-hoc audit of recorded agent evidence for benchmark-source access.

Native agent tools are not domain- or filesystem-restricted; only the system
prompt asks the agent not to look up answers. This scans what the model saw or
sent (provider streams, API calls, CLI traces, search and tool records) for
benchmark sources, local dataset caches and the record directory itself, and
lists each hit with context for manual review. A hit is not proof of leakage,
and no hit does not prove its absence: encrypted search results and pages read
without a recorded URL cannot be checked.

Usage: python -m lm_eval.sde_audit RECORD_DIR [RECORD_DIR ...] > audit.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


PATTERNS = {
    "huggingface": "huggingface",  # Hub URLs and local dataset caches
    "hf_short_url": "hf.co/",
    "sde_datasets_org": "deep-principle",
    "tmc_dataset": "peteroldriver",
    "tmc_ground_truth": "zenodo.org/records/14328055",
    "sde_repository": "howiehwong/sde-harness",
    "deepprinciple_github": "github.com/deepprinciple",
    "sde_paper": "2512.15567",
}
# Harness bookkeeping, not model-facing: CLI stderr, the pinned model catalog
# and the record paths written into api/search records.
SKIPPED_FILES = {"stderr.log", "catalog.json"}
HARNESS_KEYS = {"raw_path", "response_path", "record_path", "body_path"}
SUFFIXES = {".sse", ".json", ".jsonl", ".txt"}
RECORD = re.compile(r"[0-9a-f]{64}")
CONTEXT = 80


def evidence_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        try:
            value = json.loads(text)
        except ValueError:
            return text
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in HARNESS_KEYS}
        return json.dumps(value, ensure_ascii=False)
    return text


def scan(text: str, patterns: dict[str, str]) -> list[dict[str, Any]]:
    lowered = text.lower()
    hits = []
    for name, needle in patterns.items():
        start = lowered.find(needle)
        if start < 0:
            continue
        context = text[max(0, start - CONTEXT) : start + len(needle) + CONTEXT]
        hits.append(
            {
                "pattern": name,
                "count": lowered.count(needle),
                "context": " ".join(context.split()),
            }
        )
    return hits


def audit(record_dir: Path) -> dict[str, Any]:
    record_dir = record_dir.resolve()
    patterns = {**PATTERNS, "record_dir": str(record_dir).lower()}
    records = sorted(
        path for path in record_dir.glob("*.json") if RECORD.fullmatch(path.stem)
    )
    flagged: dict[str, list[dict[str, Any]]] = {}
    scanned = 0
    for path in sorted(record_dir.rglob("*")):
        top = path.relative_to(record_dir).parts[0]
        if (
            not path.is_file()
            or ".attempt" not in top
            or path.name in SKIPPED_FILES
            or path.suffix not in SUFFIXES
        ):
            continue
        scanned += 1
        for hit in scan(evidence_text(path), patterns):
            hit["file"] = str(path.relative_to(record_dir))
            flagged.setdefault(top.split(".", 1)[0], []).append(hit)
    rows = []
    for record_id, hits in sorted(flagged.items()):
        main = record_dir / f"{record_id}.json"
        record = json.loads(main.read_text()) if main.exists() else {}
        identity = record.get("identity") or {}
        rows.append(
            {
                "record": record_id,
                "task": identity.get("task"),
                "doc_id": identity.get("doc_id"),
                "repeat_index": record.get("repeat_index"),
                "status": record.get("status"),
                "hits": hits,
            }
        )
    return {
        "record_dir": str(record_dir),
        "patterns": patterns,
        "records": len(records),
        "scanned_files": scanned,
        "flagged_records": rows,
    }


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit(__doc__)
    reports = [audit(Path(arg)) for arg in argv]
    for report in reports:
        print(
            f"{report['record_dir']}: {len(report['flagged_records'])} of "
            f"{report['records']} records flagged",
            file=sys.stderr,
        )
    json.dump(reports, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
