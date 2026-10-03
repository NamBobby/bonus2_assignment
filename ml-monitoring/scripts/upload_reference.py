"""Upload the training reference to Evidently."""
import csv
import json
import os
import urllib.request
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "data/feature_schema.json").read_text(encoding="utf-8")
    )
    names = schema["feature_names"]
    with (root / "data/reference.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = [
            {name: float(row[name]) for name in names}
            for row in csv.DictReader(stream)
        ]

    base = os.getenv(
        "EVIDENTLY_URL", "http://localhost:28001"
    ).rstrip("/")
    request = urllib.request.Request(
        base + "/reference",
        data=json.dumps({
            "data": rows,
            "feature_names": names,
            "description": (
                "Wine training split: seed 42, stratified, "
                f"{len(rows)} rows"
            ),
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if result["samples"] != len(rows) or result["features"] != names:
        raise RuntimeError("Uploaded reference does not match training")
    print(json.dumps(result, indent=2))
    print("REFERENCE UPLOAD PASSED")


if __name__ == "__main__":
    main()
