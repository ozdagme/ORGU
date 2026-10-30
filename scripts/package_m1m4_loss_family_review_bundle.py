#!/usr/bin/env python3
"""Validate and bundle the completed M1/M4 loss-family review artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path


ROOT = Path.cwd().resolve()
RUN_ROOT = Path("runs/m1_m4_loss_family_v1")
DEFAULT_OUTPUT = Path("results/m1_m4_loss_family_review_bundle.zip")
ARMS = (
    ("BCE", "L0"),
    ("NegDW-BCE", "L1"),
    ("Focal", "FOCAL"),
    ("ASL", "ASL"),
    ("pos_weight-BCE", "POS_WEIGHT"),
)
SEEDS = (13, 42, 87, 101, 202, 303, 404, 505, 606, 707)
POLICIES = (
    "shared_threshold_0.5",
    "validation_macro_f1",
    "validation_recall_matched_to_NegDW_BCE_at_0.5",
)
SOURCE_FILES = (
    Path("scripts/run_m1m4_loss_family_comparison.py"),
    Path("src/orgu/train_encoder.py"),
    Path("src/orgu/neural.py"),
    Path("src/orgu/metrics.py"),
    Path("data/splits/finred_entity_disjoint_v1/split_manifest.json"),
    Path("data/features/finred_entity_disjoint_v1/context_manifest.json"),
    Path("data/features/finred_entity_disjoint_v1/label_schema.json"),
)
OPTIONAL_SOURCE_FILES = (
    Path("m1_m4_ek_kayiplar_calistirma_tr.md"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def present_optional_files(root: Path, paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(path for path in paths if (root / path).is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_file(root: Path, raw_path: str | Path, description: str) -> Path:
    path = Path(raw_path)
    require(not path.is_absolute(), f"{description} must be repository-relative: {path}")
    require(".." not in path.parts, f"{description} cannot contain '..': {path}")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), f"{description} escapes the project: {path}")
    require(resolved.is_file(), f"Missing {description}: {path}")
    return resolved


def repository_file(root: Path, raw_path: str | Path, description: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        resolved_path = path.resolve()
        try:
            path = resolved_path.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"{description} must be inside the project: {path}") from exc
    return project_file(root, path, description)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_dir(arm: str, seed: int) -> Path:
    return RUN_ROOT / f"{arm.lower().replace('-', '_')}_seed{seed}"


def validate_prediction_rows(
    path: Path,
    expected_count: int,
    label_count: int,
    expected_split: str,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    rows = read_jsonl(path)
    require(len(rows) == expected_count, f"{path}: expected {expected_count} rows, found {len(rows)}")
    alignment = []
    seen_ids = set()
    for index, row in enumerate(rows):
        candidate_id = row.get("candidate_id")
        require(isinstance(candidate_id, str) and candidate_id, f"{path}: row {index} lacks candidate_id")
        require(candidate_id not in seen_ids, f"{path}: duplicate candidate_id {candidate_id}")
        seen_ids.add(candidate_id)
        require(row.get("candidate_index") == index, f"{path}: candidate order/index mismatch at row {index}")
        require(row.get("split") == expected_split, f"{path}: unexpected split in row {index}")
        gold = row.get("gold_targets")
        scores = row.get("scores")
        require(isinstance(gold, list) and len(gold) == label_count, f"{path}: invalid gold vector at row {index}")
        require(all(value in (0, 1) for value in gold), f"{path}: non-binary gold vector at row {index}")
        require(isinstance(scores, list) and len(scores) == label_count, f"{path}: invalid score vector at row {index}")
        require(
            all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in scores),
            f"{path}: invalid probability at row {index}",
        )
        alignment.append((candidate_id, tuple(int(value) for value in gold)))
    return tuple(alignment)


def build_bundle(root: Path, output: Path) -> tuple[Path, str, int]:
    run_root = root / RUN_ROOT
    plan_path = project_file(root, RUN_ROOT / "experiment_plan.json", "experiment plan")
    policy_path = project_file(root, RUN_ROOT / "threshold_policy.json", "frozen threshold policy")
    report_path = project_file(root, RUN_ROOT / "test_metrics.json", "post-hoc test metrics")
    evaluation_path = project_file(root, RUN_ROOT / "test_evaluation_manifest.json", "test evaluation manifest")
    plan = read_json(plan_path)
    policy = read_json(policy_path)
    report = read_json(report_path)
    evaluation = read_json(evaluation_path)

    require(plan.get("status") == "training_completed", "All planned training runs must be complete")
    require(plan.get("seeds") == list(SEEDS), "Experiment plan does not contain the expected ten seeds")
    require([entry.get("name") for entry in plan.get("loss_arms", [])] == [name for name, _ in ARMS],
            "Experiment plan loss arms do not match this review bundle")
    require(policy.get("status") == "frozen_before_test_evaluation", "Thresholds are not frozen")
    require(policy.get("experiment_plan_sha256") == sha256_file(plan_path), "Threshold policy plan hash mismatch")
    require(policy.get("test_file_read") is False, "Threshold policy was not frozen before test access")
    require(set(policy.get("thresholds", {})) == set(POLICIES), "Unexpected threshold policies")
    require(report.get("status") == "posthoc_exploratory", "Test report is not marked post-hoc/exploratory")
    require(report.get("test_previously_inspected") is True, "Test exposure history is missing")
    require(report.get("seeds") == list(SEEDS), "Test report seed set mismatch")
    require(report.get("loss_arms") == [name for name, _ in ARMS], "Test report loss-arm mismatch")
    require(evaluation.get("status") == "completed_posthoc_test_evaluation", "Test evaluation is incomplete")

    plan_hash = sha256_file(plan_path)
    policy_hash = sha256_file(policy_path)
    report_hash = sha256_file(report_path)
    require(evaluation.get("experiment_plan_sha256") == plan_hash, "Evaluation manifest plan hash mismatch")
    require(evaluation.get("threshold_policy_sha256") == policy_hash, "Evaluation manifest threshold hash mismatch")
    require(evaluation.get("summary_sha256") == report_hash, "Evaluation manifest report hash mismatch")
    require(evaluation.get("summary_path") == str(report_path.relative_to(root)), "Evaluation summary path mismatch")
    require(report.get("threshold_policy_sha256") == policy_hash, "Test report threshold hash mismatch")
    require(report.get("test_sha256") == evaluation.get("test_sha256"), "Test input hash mismatch")

    split_manifest_path = project_file(
        root, "data/splits/finred_entity_disjoint_v1/split_manifest.json", "split manifest"
    )
    split_manifest = read_json(split_manifest_path)
    expected_test = split_manifest.get("files", {}).get("test", {})
    require(report.get("test_sha256") == expected_test.get("sha256"), "Reported test hash differs from frozen split")
    require(report.get("test_candidates") == split_manifest.get("counts", {}).get("test"),
            "Reported test candidate count differs from frozen split")
    require(report.get("test_file") == expected_test.get("path"), "Test report does not use the canonical split path")

    context_path = project_file(
        root, "data/features/finred_entity_disjoint_v1/context_manifest.json", "feature context manifest"
    )
    context = read_json(context_path)
    require(sha256_file(context_path) == plan.get("feature_manifest_sha256"), "Feature manifest hash mismatch")
    schema_path = project_file(
        root, "data/features/finred_entity_disjoint_v1/label_schema.json", "relation label schema"
    )
    require(sha256_file(schema_path) == context.get("label_schema_sha256"), "Label schema hash mismatch")
    schema = read_json(schema_path)
    schema_labels = schema.get("labels", [])
    if schema_labels and isinstance(schema_labels[0], dict):
        schema_labels = [entry.get("name", entry.get("label")) for entry in schema_labels]
    labels = [str(label) for label in schema_labels]
    require(labels == plan.get("relation_labels"), "Relation label order differs from experiment plan")
    label_count = len(labels)

    planned_train = context.get("arms", {}).get("C2", {}).get("files", {})
    for split_name in ("train", "validation"):
        split_info = planned_train.get(split_name, {})
        require(plan.get(f"{split_name}_sha256") == split_info.get("sha256"),
                f"Experiment plan {split_name} hash differs from feature manifest")

    source_hash_names = {
        "train_encoder.py": Path("src/orgu/train_encoder.py"),
        "neural.py": Path("src/orgu/neural.py"),
        "metrics.py": Path("src/orgu/metrics.py"),
    }
    source_hashes = plan.get("training_source_hashes", {})
    for name, path in source_hash_names.items():
        require(sha256_file(project_file(root, path, name)) == source_hashes.get(name),
                f"Training source changed since the experiment: {path}")

    require(set(report.get("per_seed_metrics", {})) == set(POLICIES), "Per-seed metrics omit a threshold policy")
    require(set(report.get("thresholds", {})) == set(POLICIES), "Test report thresholds omit a policy")
    require(report.get("thresholds") == policy.get("thresholds"), "Test thresholds differ from frozen policy")
    for policy_name in POLICIES:
        thresholds = policy["thresholds"][policy_name]
        require(set(thresholds) == {name for name, _ in ARMS}, f"{policy_name}: incomplete threshold map")
        per_arm = report["per_seed_metrics"][policy_name]
        require(set(per_arm) == {name for name, _ in ARMS}, f"{policy_name}: incomplete metric map")
        for arm, _ in ARMS:
            require(set(per_arm[arm]) == {str(seed) for seed in SEEDS},
                    f"{policy_name}/{arm}: incomplete seed metrics")
            require(math.isfinite(float(thresholds[arm])) and 0.0 <= float(thresholds[arm]) <= 1.0,
                    f"{policy_name}/{arm}: invalid threshold")

    contents: dict[str, str] = {}

    def add(path: Path, category: str) -> Path:
        resolved = repository_file(root, path, category)
        relative = resolved.relative_to(root).as_posix()
        contents[relative] = category
        return resolved

    optional_source_files = present_optional_files(root, OPTIONAL_SOURCE_FILES)
    for path in (plan_path, policy_path, report_path, evaluation_path, *SOURCE_FILES, *optional_source_files):
        add(path, "experiment metadata, code, schema, or instructions")

    validation_alignment = None
    expected_validation_count = int(plan.get("validation_candidates", 0))
    require(expected_validation_count > 0, "Experiment plan lacks validation candidate count")
    for arm, loss_code in ARMS:
        for seed in SEEDS:
            key = f"{arm}_seed{seed}"
            directory = run_dir(arm, seed)
            manifest_path = add(directory / "run_manifest.json", "run manifest")
            run_manifest = read_json(manifest_path)
            require(run_manifest.get("status") == "completed", f"Incomplete run: {key}")
            require(run_manifest.get("arm") == "C2" and run_manifest.get("loss_arm") == loss_code,
                    f"Run identity mismatch: {key}")
            require(run_manifest.get("loss_method") == arm and run_manifest.get("seed") == seed,
                    f"Run method/seed mismatch: {key}")
            require(run_manifest.get("labels") == labels, f"Run relation order mismatch: {key}")
            require(run_manifest.get("train_sha256") == plan.get("train_sha256"), f"Run train hash mismatch: {key}")
            require(run_manifest.get("validation_sha256") == plan.get("validation_sha256"),
                    f"Run validation hash mismatch: {key}")
            require(run_manifest.get("training_code_sha256") == source_hashes.get("train_encoder.py"),
                    f"Run training code hash mismatch: {key}")
            require(run_manifest.get("neural_code_sha256") == source_hashes.get("neural.py"),
                    f"Run model code hash mismatch: {key}")
            require(run_manifest.get("metrics_code_sha256") == source_hashes.get("metrics.py"),
                    f"Run metrics code hash mismatch: {key}")

            validation_path = project_file(root, run_manifest.get("validation_predictions_path", ""),
                                           f"{key} validation predictions")
            validation_metrics_path = project_file(root, run_manifest.get("validation_metrics_path", ""),
                                                   f"{key} validation metrics")
            require(sha256_file(validation_path) == run_manifest.get("validation_predictions_sha256"),
                    f"Validation prediction hash mismatch: {key}")
            require(sha256_file(validation_metrics_path) == run_manifest.get("validation_metrics_sha256"),
                    f"Validation metrics hash mismatch: {key}")
            require(run_manifest.get("validation_prediction_rows") == expected_validation_count,
                    f"Validation row count mismatch: {key}")
            current_alignment = validate_prediction_rows(
                validation_path, expected_validation_count, label_count, "validation"
            )
            if validation_alignment is None:
                validation_alignment = current_alignment
            else:
                require(current_alignment == validation_alignment, f"Validation candidate/gold alignment mismatch: {key}")
            add(validation_path, "raw validation probabilities and gold labels")
            add(validation_metrics_path, "selected-checkpoint validation metrics")

            validation_evidence = policy.get("validation_run_evidence", {}).get(key, {})
            require(validation_evidence.get("run_manifest_sha256") == sha256_file(manifest_path),
                    f"Frozen threshold evidence differs from run manifest: {key}")
            require(validation_evidence.get("validation_predictions_sha256") == sha256_file(validation_path),
                    f"Frozen threshold evidence differs from predictions: {key}")
            checkpoint_hash = run_manifest.get("best_checkpoint", {}).get("sha256")
            require(validation_evidence.get("checkpoint_sha256") == checkpoint_hash,
                    f"Frozen threshold checkpoint evidence mismatch: {key}")
            require(report.get("checkpoint_hashes", {}).get(key) == checkpoint_hash,
                    f"Test report checkpoint evidence mismatch: {key}")

    require(len(policy.get("validation_run_evidence", {})) == len(ARMS) * len(SEEDS),
            "Frozen validation evidence does not cover all runs")

    report_predictions = report.get("prediction_artifacts", {})
    require(report_predictions == evaluation.get("prediction_artifacts"),
            "Evaluation manifest prediction artifacts differ from test report")
    expected_keys = {f"{arm}_seed{seed}" for arm, _ in ARMS for seed in SEEDS}
    require(set(report_predictions) == expected_keys, "Test prediction artifact set is incomplete")
    test_alignment = None
    expected_test_count = int(report["test_candidates"])
    for arm, _ in ARMS:
        for seed in SEEDS:
            key = f"{arm}_seed{seed}"
            artifact = report_predictions[key]
            prediction_path = project_file(root, artifact.get("path", ""), f"{key} test predictions")
            digest = sha256_file(prediction_path)
            require(digest == artifact.get("sha256"), f"Test prediction hash mismatch: {key}")
            require(artifact.get("checkpoint_sha256") == report["checkpoint_hashes"].get(key),
                    f"Test prediction checkpoint mismatch: {key}")
            current_alignment = validate_prediction_rows(prediction_path, expected_test_count, label_count, "test")
            if test_alignment is None:
                test_alignment = current_alignment
            else:
                require(current_alignment == test_alignment, f"Test candidate/gold alignment mismatch: {key}")
            add(prediction_path, "raw post-hoc test probabilities and gold labels")

    contents[str(Path(__file__).resolve().relative_to(root))] = "review bundle generator"
    inventory = []
    for relative, category in sorted(contents.items()):
        path = project_file(root, relative, category)
        inventory.append({
            "path": relative,
            "category": category,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    readme = (
        "M1/M4 LOSS-FAMILY REVIEW BUNDLE\n"
        "\n"
        "This archive contains the completed five-arm, ten-seed entity-disjoint comparison;\n"
        "validation and post-hoc test prediction probabilities; recorded targets; per-run\n"
        "manifests; frozen threshold policies; aggregate/per-seed metrics; and source-code\n"
        "and split provenance. The entity-disjoint test was previously inspected, so its results are\n"
        "exploratory and are not independent confirmation or verified relation truth.\n"
        "\n"
        "Raw train/validation/test JSONL inputs, sentence text, and model checkpoint weights\n"
        "are intentionally excluded. Test prediction rows include candidate IDs, entity\n"
        "surfaces, recorded gold labels, and model scores.\n"
    )
    inventory.append({
        "path": "BUNDLE_README.txt",
        "category": "bundle contents and interpretation notes",
        "bytes": len(readme.encode("utf-8")),
        "sha256": hashlib.sha256(readme.encode("utf-8")).hexdigest(),
    })
    bundle_manifest = {
        "status": "ready_for_external_review",
        "experiment_plan_sha256": plan_hash,
        "threshold_policy_sha256": policy_hash,
        "test_report_sha256": report_hash,
        "test_sha256": report["test_sha256"],
        "seeds": list(SEEDS),
        "loss_arms": [name for name, _ in ARMS],
        "threshold_policies": list(POLICIES),
        "validation_candidates": expected_validation_count,
        "test_candidates": expected_test_count,
        "raw_split_files_included": False,
        "model_checkpoints_included": False,
        "omitted_optional_files": [path.as_posix() for path in OPTIONAL_SOURCE_FILES if path not in optional_source_files],
        "files": inventory,
    }

    output_path = output.resolve()
    require(output_path.is_relative_to(root), "Output ZIP must remain inside the project directory")
    require(not output_path.exists(), f"Refusing to overwrite existing review bundle: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="m1_m4_bundle_", suffix=".tmp", dir=output_path.parent,
                                         delete=False) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative in sorted(contents):
                archive.write(root / relative, arcname=relative)
            archive.writestr("BUNDLE_README.txt", readme)
            archive.writestr(
                "bundle_manifest.json",
                json.dumps(bundle_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
        os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    archive_hash = sha256_file(output_path)
    return output_path, archive_hash, len(inventory)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and bundle completed M1/M4 results for external review"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Repository-relative output ZIP (default: {DEFAULT_OUTPUT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = Path(args.output)
        require(not output.is_absolute() and ".." not in output.parts,
                "Output must be a repository-relative path without '..'")
        resolved = (ROOT / output).resolve()
        require(resolved.is_relative_to(ROOT), "Output path escapes the project")
        path, digest, file_count = build_bundle(ROOT, resolved)
        print(f"Review bundle: {path.relative_to(ROOT)}")
        print(f"Files verified and included: {file_count}")
        print(f"ZIP SHA-256: {digest}")
        print("Upload this ZIP for result review; no raw split files or model weights are included.")
    except Exception as exc:
        print(f"[bundle] ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())