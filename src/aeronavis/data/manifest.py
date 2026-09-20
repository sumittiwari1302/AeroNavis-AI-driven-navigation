"""Manifest CLI: idempotent corpus ingest → cache → splits → `manifest.json`.

Usage:
    python -m aeronavis.data.manifest --rebuild [--no-download] [--only SOURCE]
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aeronavis.config import get_config
from aeronavis.data.downloader import DownloadError, HANDLERS
from aeronavis.data.preprocess import NavSequence, preprocess_source
from aeronavis.data.splits import SPLIT_NAMES, make_splits, write_split_manifests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("aeronavis.data.manifest")

VEHICLE_SOURCES = ("io_vnbd", "self_collected")


def _cache_stats(seqs: list[NavSequence]) -> dict[str, Any]:
    hours = sum(s.hours for s in seqs)
    return {
        "sequences": len(seqs),
        "hours": round(hours, 3),
        "km": round(sum(s.km for s in seqs), 2),
        "outage_minutes": round(sum(s.outage_minutes for s in seqs), 2),
        "vehicles": sorted({s.vehicle for s in seqs} - {""}),
        "regions": sorted({s.region for s in seqs} - {""}),
    }


def build(
    rebuild: bool = False, no_download: bool = False, only: str | None = None
) -> dict[str, Any]:
    config = get_config()
    raw_root = Path(config.paths.raw)
    processed_root = Path(config.paths.processed)
    split_dir = Path(config.paths.splits)

    names = [only] if only else list(HANDLERS)
    per_source: dict[str, dict[str, Any]] = {}
    bucket_lists: dict[str, dict[str, list[NavSequence]]] = {}

    for name in names:
        handler = HANDLERS[name](config, root=raw_root)
        if not handler.available():
            if no_download:
                logger.warning(f"{name}: not available locally (--no-download), skipped")
                per_source[name] = {
                    "sequences": 0,
                    "hours": 0.0,
                    "km": 0.0,
                    "outage_minutes": 0.0,
                    "vehicles": [],
                    "regions": [],
                    "status": "not_available",
                }
                continue
            try:
                handler.download()
            except DownloadError as exc:
                logger.error(str(exc))
                per_source[name] = {
                    "sequences": 0,
                    "hours": 0.0,
                    "km": 0.0,
                    "outage_minutes": 0.0,
                    "vehicles": [],
                    "regions": [],
                    "status": "download_failed",
                }
                continue
        if not handler.available():
            logger.error(f"{name}: reported available but no payload found")
            per_source[name] = {
                "sequences": 0,
                "hours": 0.0,
                "km": 0.0,
                "outage_minutes": 0.0,
                "vehicles": [],
                "regions": [],
                "status": "empty",
            }
            continue

        cache_root = processed_root / name
        cache_root.mkdir(parents=True, exist_ok=True)
        raw_sequences = preprocess_source(name, config)
        logger.info(f"{name}: preprocessing produced {len(raw_sequences)} candidate sequences")

        for seq in raw_sequences:
            seq_dir = cache_root / seq.seq_id
            if rebuild or not seq_dir.exists():
                seq.save(cache_root)

        cached = NavSequence.load_all(cache_root)
        stats = _cache_stats(cached)
        stats["status"] = "ok"
        per_source[name] = stats
        bucket_lists[name] = make_splits(cached, config.splits)
        counts = write_split_manifests(bucket_lists[name], name, split_dir, processed_root)
        stats["splits"] = counts

        (raw_root / name / "_stats.json").write_text(
            json.dumps({k: v for k, v in stats.items() if k != "splits"}, indent=2),
            encoding="utf-8",
        )

    total_hours = round(sum(per_source[n]["hours"] for n in VEHICLE_SOURCES if n in per_source), 3)
    total_outage = round(
        sum(per_source[n]["outage_minutes"] for n in VEHICLE_SOURCES if n in per_source), 2
    )
    split_totals = {"train_seqs": 0, "val_seqs": 0, "test_seqs": 0}
    for name in names:
        for split in SPLIT_NAMES:
            split_totals[f"{split}_seqs"] += per_source[name].get("splits", {}).get(split, 0)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": per_source,
        "total_vehicle_hours": total_hours,
        "outage_minutes_total": total_outage,
        "splits": split_totals,
    }
    (processed_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(f"manifest written: {processed_root / 'manifest.json'}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m aeronavis.data.manifest")
    parser.add_argument(
        "--rebuild", action="store_true", help="reprocess/re-split and rewrite manifest"
    )
    parser.add_argument("--no-download", action="store_true", help="never touch the network")
    parser.add_argument("--only", choices=list(HANDLERS), help="process a single source")
    args = parser.parse_args(argv)
    manifest = build(rebuild=args.rebuild, no_download=args.no_download, only=args.only)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
