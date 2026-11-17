#!/usr/bin/env python3
"""Import Apple, Banana and Orange images from Fruits-360 100x100.

Expected source layout:
    <Fruits-360 root>/Training/<fruit class>/*.jpg
    <Fruits-360 root>/Test/<fruit class>/*.jpg

Copies selected classes into Task 2's data/raw/{apple,banana,orange} folders.
Filenames are prefixed with a group ID so prepare_dataset.py keeps related
Fruits-360 frames together when it creates train/validation/test splits.

The group rule follows the Fruits-360 original-size README description that
object indices occur in blocks of four (k, k+1, k+2, k+3). Confirm this against
the downloaded release before treating the split as final experimental evidence.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_INDEX = re.compile(r"^(?:r\d*_)?(?P<index>\d+)_100$", re.IGNORECASE)


def target_class(source_folder_name: str) -> str | None:
    name = source_folder_name.strip().casefold()
    if name.startswith("apple"):
        return "apple"
    if name.startswith("banana"):
        return "banana"
    if name == "orange" or name.startswith("orange "):
        return "orange"
    return None


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "source_class"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import the Fruits-360 Apple, Banana and Orange subsets into data/raw."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the extracted Fruits-360 100x100 folder containing Training and Test.",
    )
    parser.add_argument("--raw", default="data/raw", help="Task 2 raw-data folder (default: data/raw).")
    parser.add_argument("--manifest", default="artifacts/fruits360_import_manifest.csv")
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    raw_root = Path(args.raw).expanduser().resolve()
    subsets = [source_root / "Training", source_root / "Test"]
    missing = [str(p) for p in subsets if not p.is_dir()]
    if missing:
        raise SystemExit(
            "Could not find the required source folders:\n  " + "\n  ".join(missing)
            + "\nPass --source as the extracted Fruits-360 folder containing Training and Test."
        )

    # Avoid silently mixing an old or synthetic dataset with the imported subset.
    for class_name in ("apple", "banana", "orange"):
        target_dir = raw_root / class_name
        if target_dir.exists() and any(p.is_file() for p in target_dir.rglob("*")):
            raise SystemExit(
                f"{target_dir} already contains files. Move/backup them first; "
                "the importer will not mix datasets or overwrite your data."
            )

    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int]] = []
    image_counts: Counter[str] = Counter()
    group_sets: dict[str, set[str]] = defaultdict(set)
    skipped_unmatched_names: list[str] = []

    for subset_dir in subsets:
        subset_name = subset_dir.name.casefold()
        for source_class_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
            label = target_class(source_class_dir.name)
            if label is None:
                continue

            source_slug = safe_slug(source_class_dir.name)
            for image_path in sorted(source_class_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_EXTENSIONS:
                    continue

                match = IMAGE_INDEX.fullmatch(image_path.stem)
                if match is None:
                    skipped_unmatched_names.append(str(image_path))
                    continue

                frame_index = int(match.group("index"))
                object_block = frame_index // 4
                group_id = f"{source_slug}_{object_block:07d}"
                destination_dir = raw_root / label
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination_name = f"{group_id}__{subset_name}__{image_path.name}"
                destination = destination_dir / destination_name

                if destination.exists():
                    raise RuntimeError(f"Destination collision: {destination}")
                shutil.copy2(image_path, destination)

                image_counts[label] += 1
                group_sets[label].add(group_id)
                manifest_rows.append(
                    {
                        "source_file": str(image_path),
                        "source_subset": subset_dir.name,
                        "source_class": source_class_dir.name,
                        "target_class": label,
                        "frame_index": frame_index,
                        "group_id": group_id,
                        "copied_file": str(destination),
                    }
                )

    missing_classes = [name for name in ("apple", "banana", "orange") if image_counts[name] == 0]
    if missing_classes:
        raise SystemExit(
            "No matching images were imported for: " + ", ".join(missing_classes)
            + ". Check that the downloaded release has the expected class folders and *_100.jpg filenames."
        )

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_file", "source_subset", "source_class", "target_class",
        "frame_index", "group_id", "copied_file",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    source_license = source_root / "LICENSE"
    if source_license.is_file():
        license_copy = manifest_path.parent / "Fruits-360-LICENSE.txt"
        shutil.copy2(source_license, license_copy)
        print(f"License copy: {license_copy}")

    print("Fruits-360 import complete.")
    for label in ("apple", "banana", "orange"):
        print(f"{label:>7}: {image_counts[label]:>6} images | {len(group_sets[label]):>5} groups")
    print(f"Total imported: {len(manifest_rows)} images")
    print(f"Import manifest: {manifest_path}")
    print("Next: inspect the counts, then run: python prepare_dataset.py")
    print(
        "Grouping note: files are grouped using frame_index // 4 and their original "
        "source class; verify the release's object-index convention before final evaluation."
    )
    if skipped_unmatched_names:
        print(f"Skipped {len(skipped_unmatched_names)} image files with unexpected filenames.")
        print("Examples:")
        for name in skipped_unmatched_names[:5]:
            print("  ", name)


if __name__ == "__main__":
    main()
