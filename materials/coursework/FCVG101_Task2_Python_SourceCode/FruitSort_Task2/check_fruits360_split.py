#!/usr/bin/env python3
"""Audit the Fruits-360 import manifest and group-wise dataset split.

Run from the Task 2 project root:
    python check_fruits360_split.py

This checks bookkeeping and split leakage by the current group_id. It does NOT
prove that frame_index // 4 corresponds to physical fruit identity; that
assumption must be supported by the exact dataset release/version.
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

IMPORT_MANIFEST = Path("artifacts/fruits360_import_manifest.csv")
SPLIT_MANIFEST = Path("artifacts/data_manifest.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run the importer and prepare_dataset.py first.")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> int:
    imported = read_csv(IMPORT_MANIFEST)
    split_rows = read_csv(SPLIT_MANIFEST)

    # Summarise each imported group and verify each index belongs to its declared block.
    group_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    class_images = Counter()
    class_groups: dict[str, set[str]] = defaultdict(set)
    for row in imported:
        label = row["target_class"]
        gid = row["group_id"]
        idx = int(row["frame_index"])
        class_images[label] += 1
        class_groups[label].add(gid)
        group_rows[gid].append(row)

        source_slug = re.sub(r"[^a-z0-9]+", "_", row["source_class"].casefold()).strip("_")
        expected_gid = f"{source_slug}_{idx // 4:07d}"
        if gid != expected_gid:
            print(f"ERROR: group id mismatch: {gid} vs expected {expected_gid}")
            return 2

    cross_subset_groups = 0
    group_size_by_class: dict[str, list[int]] = defaultdict(list)
    for gid, rows in group_rows.items():
        subsets = {r["source_subset"].casefold() for r in rows}
        indices = {int(r["frame_index"]) for r in rows}
        labels = {r["target_class"] for r in rows}
        if len(labels) != 1:
            print(f"ERROR: group spans target labels: {gid}: {labels}")
            return 2
        if len(subsets) > 1:
            cross_subset_groups += 1
        label = rows[0]["target_class"]
        group_size_by_class[label].append(len(rows))
        if any(i // 4 != int(gid.rsplit("_", 1)[1]) for i in indices):
            print(f"ERROR: frame index outside group block: {gid}: {sorted(indices)}")
            return 2

    # Verify the final split manifest never places one group_id in multiple splits.
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[tuple[str, str], int] = Counter()
    split_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in split_rows:
        gid, split, label = row["group_id"], row["split"], row["label"]
        splits_by_group[gid].add(split)
        split_counts[(split, label)] += 1
        split_groups[(split, label)].add(gid)

    leaking = {gid: s for gid, s in splits_by_group.items() if len(s) > 1}
    print("Fruits-360 split audit")
    print(f"Imported images: {len(imported)}")
    print(f"Imported groups: {len(group_rows)}")
    print(f"Groups represented in both original Training and Test folders: {cross_subset_groups}")
    print("\nPer-class import summary:")
    for label in sorted(class_images):
        sizes = sorted(group_size_by_class[label])
        median = sizes[len(sizes) // 2] if sizes else 0
        print(
            f"  {label}: {class_images[label]} images; {len(class_groups[label])} groups; "
            f"group size min/median/max={sizes[0]}/{median}/{sizes[-1]}"
        )

    print("\nFinal split counts (images / groups):")
    for split in ("train", "val", "test"):
        for label in ("apple", "banana", "orange"):
            print(
                f"  {split:5s} {label:6s}: {split_counts[(split, label)]:5d} / "
                f"{len(split_groups[(split, label)]):4d}"
            )

    if leaking:
        print(f"\nFAIL: {len(leaking)} group IDs occur in multiple final splits.")
        for gid, splits in list(leaking.items())[:10]:
            print(f"  {gid}: {sorted(splits)}")
        return 2

    print("\nPASS: no group_id appears in more than one final split.")
    print("CAUTION: this validates the script's group bookkeeping, not the physical-object grouping rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
