import argparse
import hashlib
import json
from pathlib import Path

from . import data
from .benchmark import run


def main():
    parser = argparse.ArgumentParser(description="Evidence-gated fraud triage benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Stream canonical rows from pinned public Sparkov CSV")
    fetch.add_argument("--output", type=Path, default=Path("data/public.csv"))
    fetch.add_argument("--rows", type=int, default=100_000)
    demo = sub.add_parser("demo", help="Generate local synthetic smoke-test data")
    demo.add_argument("--rows", type=int, default=5000)
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--max-cases", type=int, default=100)
    bench = sub.add_parser("benchmark", help="Benchmark temporal holdout on a canonical CSV")
    bench.add_argument("--data", type=Path, required=True)
    bench.add_argument("--output", type=Path, default=Path("artifacts/benchmark"))
    bench.add_argument("--backend", choices=["replay", "hf"], default="replay")
    bench.add_argument("--model-id")
    bench.add_argument("--max-cases", type=int, default=100)
    bench.add_argument("--flag-quantile", type=float, default=0.9)
    bench.add_argument("--human-review-usd", type=float, default=3)
    bench.add_argument("--input-per-million", type=float)
    bench.add_argument("--output-per-million", type=float)
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            print(json.dumps(data.fetch(args.output, args.rows), indent=2))
            return
        if args.command == "demo":
            report = run(
                data.demo(args.rows),
                args.output,
                max_cases=args.max_cases,
                provenance={"kind": "local_synthetic_fixture", "seed": 42},
            )
        else:
            manifest_path = args.data.with_suffix(".manifest.json")
            provenance = (
                json.loads(manifest_path.read_text())
                if manifest_path.exists()
                else {"kind": "user_csv"}
            )
            if (
                "sha256" in provenance
                and hashlib.sha256(args.data.read_bytes()).hexdigest() != provenance["sha256"]
            ):
                raise ValueError("Dataset checksum does not match its manifest")
            report = run(
                data.read(args.data),
                args.output,
                backend=args.backend,
                max_cases=args.max_cases,
                flag_quantile=args.flag_quantile,
                model_id=args.model_id,
                human_review_usd=args.human_review_usd,
                input_per_million=args.input_per_million,
                output_per_million=args.output_per_million,
                provenance=provenance,
            )
        print(
            json.dumps(
                {
                    "backend": report["backend"],
                    "report": str(args.output / "benchmark.md"),
                    "agent": report["agent_same_cohort"],
                },
                indent=2,
            )
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
