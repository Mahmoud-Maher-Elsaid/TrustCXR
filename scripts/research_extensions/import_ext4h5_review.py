"""Import completed H5 ratings; never performs inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustcxr.grounded_llm.ext4h5_review import load_bundle, score_review_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("ratings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_bundle(args.bundle)
    ratings = json.loads(args.ratings.read_text(encoding="utf-8"))
    if isinstance(ratings, dict) and "completed_reviews" in ratings:
        integrity = json.loads((args.bundle.parent / "integrity_manifest.json").read_text(encoding="utf-8"))
        blind_map = json.loads((args.bundle.parent / "internal_blind_map.json").read_text(encoding="utf-8"))
        from trustcxr.grounded_llm.ext4h5_review import import_completed_review
        result = import_completed_review(bundle, ratings, integrity=integrity, blind_map=blind_map)
    else:
        result = score_review_rows(bundle, ratings["rows"] if isinstance(ratings, dict) else ratings)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
