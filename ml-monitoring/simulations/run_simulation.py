"""Run bounded prediction and drift experiments; fail on HTTP errors."""
import argparse
import json
import time
import urllib.request
from pathlib import Path
from data_generator import WineDataGenerator


def call(url, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="normal", choices=[
        "normal", "slight_drift", "moderate_drift",
        "severe_drift", "sudden_shift",
    ])
    parser.add_argument("--requests", "-n", type=int, default=400)
    parser.add_argument("--rps", "-r", type=float, default=20)
    parser.add_argument("--duration", "-d", type=int)
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--config", default=str(
        Path(__file__).with_name("config.yaml")
    ))
    parser.add_argument("--api-url")
    parser.add_argument("--evidently-url")
    args = parser.parse_args()

    count = (
        int(args.duration * args.rps)
        if args.duration is not None else args.requests
    )
    if count < 1 or args.rps < 0:
        parser.error("Require positive request count and nonnegative rps")
    if not 0 < args.threshold <= 1:
        parser.error("threshold must be in (0, 1]")
    if args.analyze and (
        args.no_capture or min(count, args.window) < 100
        or not 100 <= args.window <= 10000
    ):
        parser.error("Analysis requires capture and a window of at least 100")

    generator = WineDataGenerator(args.config)
    settings = generator.config["api"]
    api = (args.api_url or settings["base_url"]).rstrip("/")
    evidently = (
        args.evidently_url or settings["evidently_base_url"]
    ).rstrip("/")

    health = call(api + "/health")
    if not health.get("model_loaded"):
        raise RuntimeError("API model is not ready")
    info = call(api + "/model/info")
    names = generator.get_feature_names()
    if info["feature_names"] != names:
        raise RuntimeError("API and generator schemas differ")

    if not args.no_capture:
        reference = call(evidently + "/reference")
        if not reference.get("loaded") or reference["features"] != names:
            raise RuntimeError("Evidently reference schema differs")
        if args.reset:
            call(evidently + "/production-data", method="DELETE")

    samples = generator.generate_batch(count, args.scenario)
    predictions = {}
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        tick = time.perf_counter()
        result = call(api + "/predict", "POST", {
            "features": [sample[name] for name in names],
            "feature_names": names,
        })
        prediction = str(int(result["prediction"]))
        predictions[prediction] = predictions.get(prediction, 0) + 1

        if not args.no_capture:
            call(evidently + "/capture", "POST", {
                "features": sample,
                "prediction": result["prediction"],
                "timestamp": result["timestamp"],
                "model_version": result["model_version"],
            })

        if not args.quiet and index % 100 == 0:
            print(f"Predicted and captured: {index}/{count}", flush=True)
        if args.rps > 0:
            time.sleep(max(0, 1 / args.rps - (time.perf_counter() - tick)))

    summary = {
        "scenario": args.scenario,
        "seed": generator.config.get("seed", 501),
        "successful_requests": count,
        "captured_requests": 0 if args.no_capture else count,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prediction_counts": predictions,
        "model_version": info["model_version"],
    }
    if args.analyze:
        result = call(evidently + "/analyze", "POST", {
            "window_size": args.window,
            "threshold": args.threshold,
        })
        if result["total_features"] != 13:
            raise RuntimeError("Analysis did not monitor all 13 features")
        summary["analysis"] = result

    output = Path(__file__).resolve().parents[1] / "data"
    output.mkdir(exist_ok=True)
    filename = output / f"simulation_{args.scenario}.json"
    filename.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved:", filename)
    print("SIMULATION PASSED")


if __name__ == "__main__":
    main()
