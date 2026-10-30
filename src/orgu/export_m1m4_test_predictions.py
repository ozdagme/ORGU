from __future__ import annotations

import argparse
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file

from . import neural, train_encoder


TEST_INPUT_SHA256 = "bc8cc058fbd0ca75926063bfe7b9d0984746d494c53d406c797464cf246c233f"
TEST_CANDIDATE_COUNT = 701
DEFAULT_SEEDS = (13, 42, 87)
RUN_SPECS = (
    ("BCE_seed13", 13, "L0"),
    ("BCE_seed42", 42, "L0"),
    ("BCE_seed87", 87, "L0"),
    ("NegDW-BCE_seed13", 13, "L1"),
    ("NegDW-BCE_seed42", 42, "L1"),
    ("NegDW-BCE_seed87", 87, "L1"),
)
EXPECTED_RUN_MANIFEST_SHA256 = {
    "BCE_seed13": "9f1ebdeab15b77d56b012a22d606d79043af9524409749e9129ab7a27f45ff50",
    "BCE_seed42": "64c10cfe7c5ded16d48fd741747f303b9a4fa9295eca0ae74738a968defc2093",
    "BCE_seed87": "032f6e9308a894e21b867540e7b1f53a1002d8538b46ae397ca19ff5490a276d",
    "NegDW-BCE_seed13": "389f69d5fe8e3e265e16467e84cce88d280e5561fc4c92c2db362faf8831aa2a",
    "NegDW-BCE_seed42": "5478c7961354c2eb44417cc211ead578a3779ab9cda9d341798765e57cd6ebbe",
    "NegDW-BCE_seed87": "13399789624afc6165835c04fcbed3aba14b478ee057c13bc23525e597d2612c",
}
EXPECTED_CHECKPOINT_SHA256 = {
    "BCE_seed13": "1cf1eea6658494f05a7f6851e9b7c29ed44aa8d43a7d8d730ae8641590ab119f",
    "BCE_seed42": "19dcd89af9b6baf456c7cdfd26b6723bbab99612bf291cda206303de121b5a03",
    "BCE_seed87": "42bfa490c6b339fa706216c4e60aa02add910cf3356070bb608ff22b7e89c6fb",
    "NegDW-BCE_seed13": "5471ca0d727fe1e18f2a146f7389c0e85e2d8eeeebc5e49a9da40b132d0dc104",
    "NegDW-BCE_seed42": "24fe0ff95bd0fb45ea7f1ed6cbb40d063f3805145cecaa1192182b78780a1f44",
    "NegDW-BCE_seed87": "10ac359cf7278db12eb5749962e311b4d5a95708cf3c0b3b4eaf8160a6d5405f",
}


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    train_encoder.require(
        device.type != "cuda" or torch.cuda.is_available(),
        "CUDA was selected but is unavailable",
    )
    return device


def _preflight_runs(
    run_root: Path, artifacts: dict, seeds: tuple[int, ...] = DEFAULT_SEEDS
) -> list[dict]:
    current_hashes = {
        "training_code_sha256": train_encoder.sha256_file(Path(train_encoder.__file__)),
        "neural_code_sha256": train_encoder.sha256_file(Path(neural.__file__)),
        "metrics_code_sha256": train_encoder.sha256_file(
            Path(train_encoder.__file__).with_name("metrics.py")
        ),
        "feature_manifest_sha256": train_encoder.sha256_file(artifacts["manifest_path"]),
        "tokenizer_sha256": train_encoder.sha256_file(artifacts["tokenizer_path"]),
        "label_schema_sha256": train_encoder.sha256_file(artifacts["schema_path"]),
    }
    validated_runs = []
    requested_specs = tuple(
        (f"{loss_name}_seed{seed}", seed, loss_arm)
        for loss_name, loss_arm in (("BCE", "L0"), ("NegDW-BCE", "L1"))
        for seed in seeds
    )
    for run_name, expected_seed, expected_loss in requested_specs:
        run_dir = train_encoder.project_path(run_root / run_name.lower().replace("-", "_"), "Run directory")
        manifest_path = run_dir / "run_manifest.json"
        train_encoder.require(manifest_path.is_file(), f"Missing run manifest: {manifest_path}")
        manifest_hash = train_encoder.sha256_file(manifest_path)
        manifest = train_encoder.read_json(manifest_path)
        train_encoder.require(
            manifest.get("status") == "completed", f"Run is incomplete: {run_name}"
        )
        train_encoder.require(manifest.get("arm") == "C2", f"Unexpected feature arm: {run_name}")
        train_encoder.require(
            manifest.get("seed") == expected_seed, f"Unexpected seed: {run_name}"
        )
        train_encoder.require(
            manifest.get("loss_arm") == expected_loss,
            f"Unexpected loss arm: {run_name}",
        )
        train_encoder.require(
            manifest.get("test_file_read") is False,
            f"Run manifest already records test access: {run_name}",
        )
        train_encoder.require(manifest.get("labels") == artifacts["labels"],
                              f"Label order differs: {run_name}")
        train_encoder.require(
            manifest.get("train_sha256") == train_encoder.sha256_file(artifacts["train_path"]),
            f"Training input hash differs: {run_name}",
        )
        train_encoder.require(
            manifest.get("validation_sha256")
            == train_encoder.sha256_file(artifacts["validation_path"]),
            f"Validation input hash differs: {run_name}",
        )
        for key, current_hash in current_hashes.items():
            train_encoder.require(
                manifest.get(key) == current_hash,
                f"{key} differs for {run_name}; refusing test inference",
            )
        model_id = manifest.get("pretrained_model")
        model_revision = manifest.get("pretrained_model_revision")
        train_encoder.require(
            model_id == artifacts["manifest"].get("model_id"),
            f"Pretrained model differs from feature manifest: {run_name}",
        )
        train_encoder.require(
            model_revision == artifacts["manifest"].get("model_revision"),
            f"Pretrained model revision differs from feature manifest: {run_name}",
        )
        config = manifest.get("config", {})
        train_encoder.require(
            config.get("threshold") == 0.5
            and config.get("selection_metric") == "validation_micro_f1_at_fixed_threshold_0.5",
            f"Unexpected checkpoint-selection protocol: {run_name}",
        )
        checkpoint = manifest.get("best_checkpoint", {})
        checkpoint_path = train_encoder.project_path(
            checkpoint.get("path", ""), f"Selected checkpoint for {run_name}"
        )
        train_encoder.require(
            checkpoint_path == run_dir / "best.safetensors",
            f"Selected checkpoint path differs from the expected run directory: {run_name}",
        )
        train_encoder.require(checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}")
        checkpoint_hash = train_encoder.sha256_file(checkpoint_path)
        train_encoder.require(
            checkpoint_hash == checkpoint.get("sha256"),
            f"Checkpoint hash differs from its run manifest: {run_name}",
        )
        eval_batch_size = int(config.get("eval_batch_size", 0))
        train_encoder.require(eval_batch_size > 0, f"Invalid evaluation batch size: {run_name}")
        validated_runs.append(
            {
                "name": run_name,
                "seed": expected_seed,
                "loss_arm": expected_loss,
                "directory": run_dir,
                "manifest": manifest,
                "manifest_path": manifest_path,
                "manifest_sha256": manifest_hash,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_hash,
                "eval_batch_size": eval_batch_size,
            }
        )
    return validated_runs


def _validate_test_rows(rows: list[dict], labels: list[str]) -> None:
    train_encoder.require(len(rows) == TEST_CANDIDATE_COUNT,
                          f"Unexpected test candidate count: {len(rows)}")
    candidate_ids = []
    label_set = set(labels)
    for index, row in enumerate(rows):
        candidate_id = row.get("id")
        train_encoder.require(candidate_id is not None, f"Missing candidate ID at row {index}")
        candidate_ids.append(str(candidate_id))
        train_encoder.require(row.get("record_id") is not None,
                              f"Missing record ID at row {index}")
        train_encoder.require(row.get("split") in (None, "test"),
                              f"Unexpected split value at row {index}")
        row_labels = row.get("labels")
        train_encoder.require(isinstance(row_labels, list) and row_labels,
                              f"Missing recorded labels at row {index}")
        unknown = set(row_labels) - label_set
        train_encoder.require(not unknown,
                              f"Unknown labels at row {index}: {sorted(unknown)}")
        train_encoder.require(
            isinstance(row.get("model_inputs", row).get("input_ids"), list),
            f"Missing encoded model inputs at row {index}",
        )
    train_encoder.require(len(candidate_ids) == len(set(candidate_ids)),
                          "Test file contains duplicate candidate IDs")


@torch.inference_mode()
def _score_rows(model, rows: list[dict], artifacts: dict, device: torch.device,
                batch_size: int, split_hash: str) -> list[dict]:
    model.eval()
    predictions = []
    labels = artifacts["labels"]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs, positions = neural.make_batch(
            batch, artifacts["tokenizer"], artifacts["marker_ids"], device
        )
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs, positions)
        batch_scores = torch.sigmoid(logits.float()).cpu().tolist()
        for row, scores in zip(batch, batch_scores):
            train_encoder.require(len(scores) == len(labels),
                                  f"Score width differs for candidate {row.get('id')}")
            valid_scores = all(
                math.isfinite(float(score)) and 0.0 <= float(score) <= 1.0
                for score in scores
            )
            train_encoder.require(
                valid_scores, f"Invalid probability for candidate {row.get('id')}"
            )
            gold = set(row["labels"])
            predictions.append(
                {
                    "candidate_index": len(predictions),
                    "candidate_id": row["id"],
                    "record_id": row["record_id"],
                    "split": "test",
                    "split_sha256": split_hash,
                    "head": row.get("head"),
                    "tail": row.get("tail"),
                    "gold_labels": sorted(gold),
                    "gold_targets": [int(label in gold) for label in labels],
                    "scores": [float(score) for score in scores],
                }
            )
    train_encoder.require(len(predictions) == len(rows), "Test prediction count mismatch")
    train_encoder.require(
        [prediction["candidate_id"] for prediction in predictions]
        == [row["id"] for row in rows],
        "Test prediction order differs from the source file",
    )
    return predictions


def export_test_predictions(
    feature_dir_value: str,
    test_file_value: str,
    run_root_value: str,
    device_name: str,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    manifest_suffix: str | None = None,
) -> Path:
    feature_dir = train_encoder.project_path(feature_dir_value, "Feature directory")
    test_file = train_encoder.project_path(test_file_value, "Test file")
    run_root = train_encoder.project_path(run_root_value, "Run root")
    artifacts = train_encoder.load_feature_data(feature_dir)
    train_encoder.require(
        artifacts["manifest"].get("test_inference_run") is False,
        "Feature manifest does not preserve the pre-inference test lock",
    )
    train_encoder.require(bool(seeds), "At least one seed is required")
    train_encoder.require(len(seeds) == len(set(seeds)), "Seed IDs must be unique")
    train_encoder.require(all(0 <= seed < 2**63 for seed in seeds), "Seed IDs must be in [0, 2^63)")
    validated_runs = _preflight_runs(run_root, artifacts, seeds)
    suffix = manifest_suffix if manifest_suffix is not None else (
        "" if seeds == DEFAULT_SEEDS else f"_seeds_{'_'.join(map(str, seeds))}"
    )
    output_manifest_path = run_root / f"test_inference_manifest{suffix}.json"
    output_paths = [run["directory"] / "test_predictions.jsonl" for run in validated_runs]
    temporary_paths = [path.with_name(path.name + ".tmp") for path in output_paths]
    temporary_manifest_path = output_manifest_path.with_name(output_manifest_path.name + ".tmp")
    for path in [output_manifest_path, *output_paths, *temporary_paths, temporary_manifest_path]:
        train_encoder.require(
            not path.exists(), f"Output already exists; refusing overwrite: {path}"
        )

    test_hash = train_encoder.sha256_file(test_file)
    train_encoder.require(
        test_hash == TEST_INPUT_SHA256,
        f"Test input hash differs from the frozen file: {test_hash}",
    )
    test_rows = train_encoder.read_jsonl(test_file)
    _validate_test_rows(test_rows, artifacts["labels"])
    test_order_hash = train_encoder.ordered_ids_sha256(test_rows)
    device = _resolve_device(device_name)
    predictions_by_run = {}
    for run in validated_runs:
        manifest = run["manifest"]
        model = neural.EntityMarkerEncoder(
            manifest["pretrained_model"],
            artifacts["tokenizer"].get_vocab_size(with_added_tokens=True),
            len(artifacts["labels"]),
            revision=manifest["pretrained_model_revision"],
        ).to(device)
        state = load_file(str(run["checkpoint_path"]), device="cpu")
        model.load_state_dict(state, strict=True)
        predictions_by_run[run["name"]] = _score_rows(
            model,
            test_rows,
            artifacts,
            device,
            run["eval_batch_size"],
            test_hash,
        )
        del state
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[test-export] scored {run['name']} ({len(test_rows)} candidates)", flush=True)

    result_records = {}
    for run, output_path, temporary_path in zip(validated_runs, output_paths, temporary_paths):
        predictions = predictions_by_run[run["name"]]
        train_encoder.write_jsonl(temporary_path, predictions)
        result_records[run["name"]] = {
            "seed": run["seed"],
            "loss_arm": run["loss_arm"],
            "source_run_manifest_sha256": run["manifest_sha256"],
            "checkpoint_path": str(run["checkpoint_path"]),
            "checkpoint_sha256": run["checkpoint_sha256"],
            "test_predictions_path": str(output_path),
            "test_predictions_sha256": train_encoder.sha256_file(temporary_path),
            "test_prediction_rows": len(predictions),
            "test_candidate_order_sha256": test_order_hash,
        }
    export_manifest = {
        "status": "completed",
        "operation": "post-training raw-probability inference on the entity-disjoint test",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_previously_inspected": True,
        "interpretation": "exploratory; not an independent test result",
        "test_file_read": True,
        "test_input_path": str(test_file),
        "test_input_sha256": test_hash,
        "test_candidate_count": len(test_rows),
        "test_candidate_order_sha256": test_order_hash,
        "seeds": list(seeds),
        "labels": artifacts["labels"],
        "scores": "raw sigmoid probabilities in labels order; no threshold applied",
        "test_metrics_computed": False,
        "test_used_for_checkpoint_or_threshold_selection": False,
        "feature_manifest_sha256": train_encoder.sha256_file(artifacts["manifest_path"]),
        "tokenizer_sha256": train_encoder.sha256_file(artifacts["tokenizer_path"]),
        "label_schema_sha256": train_encoder.sha256_file(artifacts["schema_path"]),
        "exporter_code_sha256": train_encoder.sha256_file(Path(__file__)),
        "device": str(device),
        "python_version": platform.python_version(),
        "runs": result_records,
    }
    train_encoder.write_json(temporary_manifest_path, export_manifest)
    for temporary_path, output_path in zip(temporary_paths, output_paths):
        temporary_path.replace(output_path)
    temporary_manifest_path.replace(output_manifest_path)
    print(f"[test-export] manifest: {output_manifest_path}", flush=True)
    print("[test-export] raw scores exported; no threshold or test metric was applied", flush=True)
    return output_manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export raw test probabilities for hash-pinned or newly trained paired M1/M4 seeds."
    )
    parser.add_argument("--feature-dir", default="data/features/finred_entity_disjoint_v1")
    parser.add_argument("--test-file", default="data/features/finred_entity_disjoint_v1/C2/test.jsonl")
    parser.add_argument("--run-root", default="runs/m1_m4_loss_family_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"), default="auto")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Seed IDs to export; defaults to the original hash-pinned seeds 13 42 87.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
        suffix = "" if args.seeds is None else f"_seeds_{'_'.join(map(str, seeds))}"
        export_test_predictions(
            args.feature_dir,
            args.test_file,
            args.run_root,
            args.device,
            seeds=seeds,
            manifest_suffix=suffix,
        )
    except Exception as exc:
        print(f"[test-export] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())