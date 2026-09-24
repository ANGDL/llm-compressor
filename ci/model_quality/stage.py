"""Execute model quality CI stages and persist auditable results."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .config import evaluation_fingerprint as configured_evaluation_fingerprint
from .config import load_model_config
from .executor import (
    StageError,
    find_model,
    run_aggregate,
    run_evaluate,
    run_preflight,
    run_publish,
    run_quantize,
    run_report,
    run_runtime_smoke,
    run_validate,
)
from .state import atomic_write_json, model_run_dir, write_stage_result

KNOWN_STAGES = {
    "preflight",
    "quantize",
    "validate",
    "runtime-smoke",
    "evaluate",
    "report",
    "publish",
    "aggregate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", default="default")
    parser.add_argument("--stage", choices=sorted(KNOWN_STAGES), required=True)
    parser.add_argument("--models-json", default="[]")
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--fingerprint")
    parser.add_argument("--evaluation-fingerprint")
    parser.add_argument(
        "--run-mode",
        choices=("quantize", "quantize_and_eval", "eval_only", "upload_only"),
        default="quantize",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_model_config(args.config)
    if args.stage != "aggregate" and not args.model:
        raise SystemExit("--model is required for model stages")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "model": args.model,
                    "stage": args.stage,
                    "configured_models": len(config["models"]),
                    "status": "DRY_RUN",
                },
                sort_keys=True,
            )
        )
        return 0

    if args.stage == "aggregate":
        expected_models = json.loads(args.models_json)
        result = run_aggregate(config, args.run_id, expected_models=expected_models)
        print(json.dumps(result, sort_keys=True))
        return 1 if result["status"] == "FAIL" else 0

    model = find_model(config, args.model)
    effective_artifact_fingerprint = args.fingerprint
    effective_evaluation_fingerprint = args.evaluation_fingerprint
    if args.stage != "preflight":
        manifest_path = model_run_dir(args.run_id, args.model) / "input-manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            effective_artifact_fingerprint = manifest.get(
                "finalized_artifact_fingerprint", args.fingerprint
            )
            if effective_artifact_fingerprint is not None:
                effective_evaluation_fingerprint = configured_evaluation_fingerprint(
                    model, effective_artifact_fingerprint
                )
    executors = {
        "preflight": lambda selected_model, run_id: run_preflight(
            selected_model,
            run_id,
            run_mode=args.run_mode,
            git_sha=args.git_sha,
            fingerprint=args.fingerprint,
            evaluation_fingerprint=args.evaluation_fingerprint,
            attempt_id=args.attempt_id,
        ),
        "quantize": lambda selected_model, run_id: run_quantize(
            selected_model,
            run_id,
            attempt_id=args.attempt_id,
            artifact_fingerprint=effective_artifact_fingerprint,
        ),
        "validate": run_validate,
        "runtime-smoke": run_runtime_smoke,
        "evaluate": lambda selected_model, run_id: run_evaluate(
            selected_model,
            run_id,
            artifact_fingerprint=effective_artifact_fingerprint,
            evaluation_fingerprint=effective_evaluation_fingerprint,
            attempt_id=args.attempt_id,
        ),
        "report": lambda selected_model, run_id: run_report(
            selected_model,
            run_id,
            run_mode=args.run_mode,
            artifact_fingerprint=effective_artifact_fingerprint,
            evaluation_fingerprint=effective_evaluation_fingerprint,
            attempt_id=args.attempt_id,
        ),
        "publish": lambda selected_model, run_id: run_publish(
            selected_model,
            run_id,
            explicitly_requested=args.run_mode == "upload_only",
            artifact_fingerprint=effective_artifact_fingerprint,
            evaluation_fingerprint=effective_evaluation_fingerprint,
        ),
    }
    try:
        result = executors[args.stage](model, args.run_id)
    except StageError as error:
        result = {
            "status": "FAIL",
            "reason_code": error.reason_code,
            "message": str(error),
        }
        write_stage_result(
            args.run_id,
            args.model,
            args.stage,
            result,
            attempt_id=args.attempt_id,
            artifact_fingerprint=effective_artifact_fingerprint,
            evaluation_fingerprint=effective_evaluation_fingerprint,
        )
        print(json.dumps(result, sort_keys=True))
        return 1
    except Exception as error:  # pragma: no cover - last-resort audit record
        result = {
            "status": "FAIL",
            "reason_code": "UNEXPECTED_ERROR",
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_stage_result(
            args.run_id,
            args.model,
            args.stage,
            result,
            attempt_id=args.attempt_id,
            artifact_fingerprint=effective_artifact_fingerprint,
            evaluation_fingerprint=effective_evaluation_fingerprint,
        )
        print(json.dumps(result, sort_keys=True))
        return 1

    if args.stage == "preflight":
        effective_artifact_fingerprint = result.get(
            "finalized_artifact_fingerprint", effective_artifact_fingerprint
        )
        effective_evaluation_fingerprint = result.get(
            "finalized_evaluation_fingerprint",
            effective_evaluation_fingerprint,
        )
    if args.stage == "validate" and result.get("status") in {"PASS", "WARN"}:
        run_dir = model_run_dir(args.run_id, args.model)
        input_manifest_path = run_dir / "input-manifest.json"
        source_provenance = None
        if input_manifest_path.is_file():
            source_provenance = json.loads(
                input_manifest_path.read_text(encoding="utf-8")
            ).get("source_provenance")
        atomic_write_json(
            run_dir / "artifact-manifest.json",
            {
                "schema_version": 1,
                "run_id": args.run_id,
                "model_id": args.model,
                "attempt_id": args.attempt_id,
                "artifact_fingerprint": effective_artifact_fingerprint,
                "source_provenance": source_provenance,
                "artifact_content_fingerprint": result.get(
                    "artifact_content_fingerprint"
                ),
                "files": result.get("artifact_files", []),
            },
        )
    write_stage_result(
        args.run_id,
        args.model,
        args.stage,
        result,
        attempt_id=args.attempt_id,
        artifact_fingerprint=effective_artifact_fingerprint,
        evaluation_fingerprint=effective_evaluation_fingerprint,
    )
    print(json.dumps(result, sort_keys=True))
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
