"""Compare metric cardinality after two batches of unique request IDs."""
import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from app.main import app
from app.metrics import TRAP

ROOT = Path(__file__).resolve().parents[1]


def snapshot(client):
    response = client.get("/metrics")
    response.raise_for_status()
    samples = [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name.startswith("wdbc_")
    ]
    return len(samples), len(response.content)


def main():
    card = json.loads((ROOT / "models/model_card.json").read_text())
    data = pd.read_csv(ROOT / "data/raw/wdbc.csv")
    rows = [
        data.loc[data["diagnosis"] == label].iloc[0]
        for label in ("B", "M")
    ]
    features = [
        {name: float(row[name]) for name in card["features"]}
        for row in rows
    ]

    with TestClient(app) as client:
        # Ensure the error counter is present in both modes.
        response = client.post(
            "/predict", json={"sample_id": "missing", "features": {}}
        )
        assert response.status_code == 422

        counts = []
        print(f"T04_TRAP={int(TRAP)}")
        for batch in range(2):
            for index in range(200):
                response = client.post("/predict", json={
                    "sample_id": f"batch-{batch}-request-{index}",
                    "features": features[index % 2],
                })
                assert response.status_code == 200, response.text

            count, size = snapshot(client)
            counts.append(count)
            print(
                f"After {(batch + 1) * 200} predictions: "
                f"{count} WDBC series; /metrics={size} bytes"
            )

        if TRAP:
            assert counts[1] > counts[0], counts
        else:
            assert counts[1] == counts[0], counts
        print("CARDINALITY CHECK PASSED")


if __name__ == "__main__":
    main()
