"""Inspect the instructor's tweet ZIP without extracting or modifying it.

Usage: python scripts/audit_dataset.py --archive tweet.zip
Optional: --output docs/DATA_AUDIT.json
Reports document counts, empty files and exact train/test duplicate content.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def audit(archive):
    counts = Counter()
    hashes = defaultdict(list)
    empty_files = []
    decode_errors = []
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            parts = PurePosixPath(entry.filename).parts
            if (entry.is_dir() or len(parts) != 4 or parts[0] != "tweet"
                    or parts[1] not in {"train", "test"}
                    or parts[2] not in {"positive", "negative"}
                    or not parts[3].endswith(".txt")):
                continue
            raw = bundle.read(entry)
            counts[f"{parts[1]}/{parts[2]}"] += 1
            if not raw.strip():
                empty_files.append(entry.filename)
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                decode_errors.append(entry.filename)
            hashes[hashlib.sha256(raw).hexdigest()].append(entry.filename)
    overlaps = [
        names for names in hashes.values()
        if any(name.startswith("tweet/train/") for name in names)
        and any(name.startswith("tweet/test/") for name in names)
    ]
    return {
        "archive": Path(archive).name,
        "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        "documents": sum(counts.values()),
        "counts": dict(sorted(counts.items())),
        "empty_files": empty_files,
        "non_utf8_files": decode_errors,
        "exact_train_test_overlap_groups": overlaps,
        "note": "Exact byte matches only; no normalization or split changes applied.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("tweet.zip"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.archive)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
