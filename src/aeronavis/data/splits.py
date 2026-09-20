"""Deterministic, leak-free train/val/test splits.

The grouping atom is the drive/subject: `drive_id`. No atom may appear in more
than one split, and (for GPS corpora) a drive is a contiguous route, which also
keeps neighboring geo-tiles out of train+test simultaneously.
"""

import hashlib
import logging
from pathlib import Path

import pandas as pd

from aeronavis.config import SplitsConfig
from aeronavis.data.preprocess import NavSequence

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def stable_atom(source: str, drive_id: str) -> int:
    digest = hashlib.sha256(f"{source}:{drive_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _target_counts(total: int, cfg: SplitsConfig) -> dict[str, int]:
    return {
        "train": int(round(total * cfg.train_ratio)),
        "val": int(round(total * cfg.val_ratio)),
        "test": total - int(round(total * cfg.train_ratio)) - int(round(total * cfg.val_ratio)),
    }


def _can_fit(counts: dict[str, int], targets: dict[str, int], split: str, add: int) -> bool:
    if targets[split] == 0:
        return False
    return counts[split] + add <= targets[split]


def make_splits(seqs: list[NavSequence], cfg: SplitsConfig) -> dict[str, list[NavSequence]]:
    """Greedy placement of whole drive-groups onto target proportions."""
    atoms: dict[tuple[str, str], list[NavSequence]] = {}
    for seq in seqs:
        atoms.setdefault((seq.source, seq.drive_id), []).append(seq)

    order = sorted(atoms.items(), key=lambda kv: (stable_atom(*kv[0]), kv[0]))
    total = len(seqs)
    targets = _target_counts(total, cfg)
    counts = {"train": 0, "val": 0, "test": 0}
    buckets: dict[str, list[NavSequence]] = {"train": [], "val": [], "test": []}

    for (source, drive), group in order:
        placed = False
        for split in ("train", "val"):
            if _can_fit(counts, targets, split, len(group)):
                buckets[split].extend(group)
                counts[split] += len(group)
                placed = True
                break
        if not placed:
            buckets["test"].extend(group)
            counts["test"] += len(group)

    for split in SPLIT_NAMES:
        logging_share = 100.0 * len(buckets[split]) / max(total, 1)
        logger.info(f"  split {split}: {len(buckets[split])} sequences ({logging_share:.1f}%)")
    return buckets


def leakage_sources(buckets: dict[str, list[NavSequence]]) -> set[tuple[str, str]]:
    """Atoms present in more than one split — must be empty."""
    split_atoms: dict[str, set[tuple[str, str]]] = {}
    for split, seqs in buckets.items():
        split_atoms[split] = {(s.source, s.drive_id) for s in seqs}
    atoms_train = split_atoms.get("train", set())
    atoms_val = split_atoms.get("val", set())
    atoms_test = split_atoms.get("test", set())
    cross = (atoms_train & atoms_val) | (atoms_train & atoms_test) | (atoms_val & atoms_test)
    return cross


def write_split_manifests(
    buckets: dict[str, list[NavSequence]],
    source: str,
    split_dir: Path,
    processed_root: Path,
) -> dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        rows = []
        for seq in buckets[split]:
            if seq.source != source:
                continue
            cache = processed_root / source / seq.seq_id
            rows.append(
                {
                    "seq_id": seq.seq_id,
                    "file_path": str(cache.relative_to(split_dir.parent)),
                    "start_ts": round(seq.start_ts, 4),
                    "end_ts": round(seq.end_ts, 4),
                    "hours": round(seq.hours, 4),
                    "km": round(seq.km, 3),
                    "outage_minutes": round(seq.outage_minutes, 3),
                }
            )
        frame = pd.DataFrame(
            rows,
            columns=[
                "seq_id",
                "file_path",
                "start_ts",
                "end_ts",
                "hours",
                "km",
                "outage_minutes",
            ],
        )
        frame.to_csv(split_dir / f"{source}_{split}.csv", index=False)
        counts[split] = len(frame)
        logger.info(f"wrote {source}_{split}.csv with {len(frame)} rows")
    return counts
