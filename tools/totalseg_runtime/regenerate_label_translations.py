"""Regenerate structures.en.json from cached TotalSegmentator sidecars.

Run after downloading new task weights so the English translation table
covers every structure the shipped models can output.
"""

import argparse
import json
from pathlib import Path


def display(name: str) -> str:
    return " ".join(w.capitalize() for w in name.split("_"))


def collect_names(sidecars_dir: Path) -> set[str]:
    names: set[str] = set()
    for p in sorted(sidecars_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skipping {p.name}: {e}")
            continue
        labels = data.get("labels")
        if not isinstance(labels, dict):
            continue
        for name in labels:
            if name and name != "background":
                names.add(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sidecars-dir",
        type=Path,
        default=Path.home() / ".config" / "invesalius" / "deep_learning" / "weights",
        help="Directory holding TotalSegmentator sidecar JSONs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "invesalius/segmentation/deep_learning/totalseg/translations/structures.en.json"
        ),
        help="Where to write the regenerated English translations.",
    )
    args = parser.parse_args()

    if not args.sidecars_dir.is_dir():
        raise SystemExit(f"Sidecars dir does not exist: {args.sidecars_dir}")

    names = collect_names(args.sidecars_dir)
    if not names:
        raise SystemExit("No structure names found in any sidecar; nothing to write.")

    en = {name: display(name) for name in sorted(names)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(en, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(en)} structure names to {args.out}")


if __name__ == "__main__":
    main()
