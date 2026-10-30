#!/usr/bin/env python3
"""Prepare and execute reviewer experiments without opening any test split.

Preparation uses the frozen FinRED entity-disjoint training file only. The
synthetic censoring artifacts are pipeline fixtures, not evidence that missing
labels were recovered. Model runs use the existing ORGU trainer and validation
split; they do not read or score a test set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path


FEATURE_DIR = Path("data/features/finred_entity_disjoint_v1")
CONTEXT_PATH = FEATURE_DIR / "context_manifest.json"
OUTPUT_ROOT = Path("runs/reviewer_experiments_v1")
RHO_VALUES = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
RHO_SEEDS = (13, 42, 87)
CENSOR_RATES = (0.0, 0.1, 0.3, 0.5)
CENSOR_MASK_SEEDS = (13, 42, 87)
CENSOR_TRAIN_SEEDS = (13, 42, 87)
REQUIRED_TRAINING_PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "numpy")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def preserve(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, f"Refusing to overwrite changed artifact: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def load_frozen_train(root: Path) -> tuple[list[dict], list[str], Path, str, str, str]:
    context_path = root / CONTEXT_PATH
    schema_path = root / FEATURE_DIR / "label_schema.json"
    require(context_path.is_file() and schema_path.is_file(), "Missing frozen feature context or label schema")
    context = read_json(context_path)
    require(context.get("test_inference_run") is False, "Feature context does not preserve the test lock")
    schema = read_json(schema_path)
    labels = schema.get("labels")
    if labels and isinstance(labels[0], dict):
        labels = [item.get("name", item.get("label")) for item in labels]
    require(isinstance(labels, list) and len(labels) == 29, "Expected the frozen 29-relation inventory")
    require(len(set(labels)) == len(labels) and all(isinstance(label, str) for label in labels),
            "Invalid or duplicate relation labels")

    train_info = context.get("arms", {}).get("C2", {}).get("files", {}).get("train", {})
    expected_hash = train_info.get("sha256")
    expected_rows = train_info.get("candidates")
    expected_assignments = train_info.get("unique_label_assignments")
    require(isinstance(expected_hash, str) and len(expected_hash) == 64, "Invalid frozen train hash")
    require(isinstance(expected_rows, int), "Missing frozen train row count")
    require(isinstance(expected_assignments, int), "Missing frozen positive-assignment count")
    canonical_train_path = train_info.get("path")
    require(isinstance(canonical_train_path, str) and not Path(canonical_train_path).is_absolute()
            and ".." not in Path(canonical_train_path).parts,
            "Frozen train path must be repository-relative")
    candidates = [root / canonical_train_path]
    source_path = next((path for path in candidates if path.is_file() and sha256_file(path) == expected_hash), None)
    require(source_path is not None, "No local train file matches the frozen C2 SHA-256")

    rows = read_jsonl(source_path)
    require(len(rows) == expected_rows, f"Expected {expected_rows} train rows, found {len(rows)}")
    allowed = set(labels)
    seen_ids = set()
    positive_count = 0
    for row in rows:
        require(row.get("split") == "train", f"Non-training row in frozen train file: {row.get('id')}")
        require(row.get("eligible_for_training", True), f"Ineligible row in frozen training input: {row.get('id')}")
        candidate_id = row.get("id")
        require(isinstance(candidate_id, str) and candidate_id and candidate_id not in seen_ids,
                "Missing or duplicate candidate ID in frozen train file")
        seen_ids.add(candidate_id)
        row_labels = row.get("labels")
        require(isinstance(row_labels, list) and set(row_labels) <= allowed,
                f"Invalid labels for training candidate {candidate_id}")
        positive_count += len(row_labels)
    require(positive_count == expected_assignments,
            f"Expected {expected_assignments} positive assignments, found {positive_count}")
    return rows, labels, source_path, expected_hash, sha256_file(schema_path), canonical_train_path


def mask_payloads(rows: list[dict], labels: list[str], seed: int) -> dict[float, tuple[bytes, dict]]:
    assignments = [(row["id"], label) for row in rows for label in row["labels"]]
    permutation = list(assignments)
    random.Random(seed).shuffle(permutation)
    label_order = {label: index for index, label in enumerate(labels)}
    outputs = {}
    for rate in CENSOR_RATES:
        count = math.floor(len(assignments) * rate + 0.5)
        removed_by_candidate: dict[str, set[str]] = {}
        for candidate_id, label in permutation[:count]:
            removed_by_candidate.setdefault(candidate_id, set()).add(label)
        mask_rows = [
            {"candidate_id": row["id"],
             "removed_labels": sorted(removed_by_candidate[row["id"]], key=label_order.__getitem__)}
            for row in rows if row["id"] in removed_by_candidate
        ]
        payload = jsonl_bytes(mask_rows)
        outputs[rate] = (payload, {
            "rate": rate,
            "mask_seed": seed,
            "original_positive_assignments": len(assignments),
            "removed_positive_assignments": count,
            "remaining_positive_assignments": len(assignments) - count,
            "affected_candidates": len(mask_rows),
            "mask_sha256": sha256_bytes(payload),
        })
    return outputs


def verify_masks(rows: list[dict], mask_details: list[dict]) -> None:
    source_labels = {row["id"]: set(row["labels"]) for row in rows}
    masks_by_seed: dict[int, dict[float, set[tuple[str, str]]]] = {}
    for details in mask_details:
        path = Path(details["path"])
        require(path.is_file() and sha256_file(path) == details["sha256"],
                f"Mask hash mismatch: {path}")
        assignments = set()
        for row in read_jsonl(path):
            candidate_id = row.get("candidate_id")
            removed_labels = row.get("removed_labels")
            require(candidate_id in source_labels, f"Mask includes unknown candidate: {candidate_id}")
            require(isinstance(removed_labels, list) and removed_labels,
                    f"Mask has no removed labels: {candidate_id}")
            require(set(removed_labels) <= source_labels[candidate_id],
                    f"Mask removes a label not present in the source: {candidate_id}")
            assignments.update((candidate_id, label) for label in removed_labels)
        expected_count = math.floor(details["original_positive_assignments"] * details["rate"] + 0.5)
        require(len(assignments) == details["removed_positive_assignments"] == expected_count,
                f"Wrong number of removed assignments in {path}")
        if details["rate"] > 0:
            masks_by_seed.setdefault(details["mask_seed"], {})[details["rate"]] = assignments
    for masks in masks_by_seed.values():
        require(masks[0.1] <= masks[0.3] <= masks[0.5],
                "Censor masks are not nested within a removal-repeat seed")


def rho_jobs() -> list[dict]:
    jobs = []
    for seed in RHO_SEEDS:
        jobs.append({"run_id": f"rho_bce_seed{seed}", "family": "rho_sweep",
                     "loss": "L0", "rho": None, "seed": seed})
        for rho in RHO_VALUES:
            rho_tag = str(rho).replace(".", "p")
            jobs.append({"run_id": f"rho_{rho_tag}_seed{seed}", "family": "rho_sweep",
                         "loss": "L1", "rho": rho, "seed": seed})
    return jobs


def censor_jobs(mask_refs: dict[tuple[float, int], dict]) -> list[dict]:
    jobs = []
    for rate in CENSOR_RATES:
        mask_seeds = (0,) if rate == 0.0 else CENSOR_MASK_SEEDS
        for mask_seed in mask_seeds:
            mask_info = mask_refs[(rate, mask_seed)]
            for seed in CENSOR_TRAIN_SEEDS:
                for loss, rho in (("L0", None), ("L1", 0.1)):
                    rate_tag = str(rate).replace(".", "p")
                    loss_tag = "BCE" if loss == "L0" else "NegDW"
                    run_id = f"censor_{rate_tag}_mask{mask_seed}_seed{seed}_{loss_tag}"
                    jobs.append({"run_id": run_id, "family": "synthetic_censoring_pilot",
                                 "loss": loss, "rho": rho, "seed": seed,
                                 "censor_rate": rate, "mask_seed": mask_seed,
                                 "mask_path": mask_info["path"],
                                 "mask_sha256": mask_info["sha256"],
                                 "removed_positive_assignments": mask_info["removed_positive_assignments"]})
    return jobs


def prepare(root: Path, output_root: Path) -> dict:
    rows, labels, source_path, train_hash, schema_hash, canonical_train_path = load_frozen_train(root)
    mask_refs = {}
    mask_details = []
    seeds_by_rate = {0.0: (0,), **{rate: CENSOR_MASK_SEEDS for rate in CENSOR_RATES if rate > 0}}
    for seed in sorted(set(seed for values in seeds_by_rate.values() for seed in values)):
        payloads = mask_payloads(rows, labels, seed)
        for rate, mask_seed_values in seeds_by_rate.items():
            if seed not in mask_seed_values:
                continue
            payload, details = payloads[rate]
            rate_tag = str(rate).replace(".", "p")
            relative = Path("masks") / f"censor_{rate_tag}_seed{seed}.jsonl"
            preserve(output_root / relative, payload)
            details.update({"path": str((output_root / relative).as_posix()),
                            "sha256": sha256_bytes(payload)})
            mask_refs[(rate, seed)] = details
            mask_details.append(details)

    verify_masks(rows, mask_details)
    jobs = rho_jobs() + censor_jobs(mask_refs)
    plan = {
        "format_version": 1,
        "status": "prepared_not_trained",
        "dataset": "FinRED entity-disjoint training candidates",
        "training_input": canonical_train_path,
        "preparation_source_sha256": sha256_file(source_path),
        "training_input_sha256": train_hash,
        "training_candidates": len(rows),
        "relation_inventory": labels,
        "label_schema_sha256": schema_hash,
        "positive_assignments": sum(len(row["labels"]) for row in rows),
        "rho_sweep": {"rho_values": list(RHO_VALUES), "seeds": list(RHO_SEEDS),
                      "validation_only": True, "test_read": False},
        "synthetic_censoring": {
            "rates": list(CENSOR_RATES),
            "mask_seeds": list(CENSOR_MASK_SEEDS),
            "training_seeds": list(CENSOR_TRAIN_SEEDS),
            "changed_split": "training labels only",
            "validation_labels_modified": False,
            "test_read": False,
            "gold_status": "unknown; FinRED completeness is not independently verified",
            "external_gold_dataset_available": False,
            "reviewer_M5_eligible": False,
            "interpretation": "pipeline pilot only; does not establish recovery of naturally missing labels",
        },
        "masks": mask_details,
        "mask_integrity_verified": True,
        "jobs": jobs,
        "job_count": len(jobs),
        "runner_sha256": sha256_file(Path(__file__)),
        "test_file_opened": False,
    }
    preserve(output_root / "experiment_plan.json", json_bytes(plan))
    return plan


def check_training_dependencies() -> None:
    missing = [package for package in REQUIRED_TRAINING_PACKAGES
               if importlib.util.find_spec(package) is None]
    require(not missing,
            "Training dependencies unavailable in the current interpreter: " + ", ".join(missing)
            + ". Use an already provisioned compatible environment; this pipeline will not install packages or create a virtual environment.")


def execute(root: Path, output_root: Path, run_id: str, device: str) -> None:
    plan_path = output_root / "experiment_plan.json"
    require(plan_path.is_file(), "Experiment plan is missing; run --prepare-only first")
    plan = read_json(plan_path)
    matches = [job for job in plan["jobs"] if job["run_id"] == run_id]
    require(len(matches) == 1, f"Unknown or duplicate run ID: {run_id}")
    job = matches[0]
    require(plan.get("mask_integrity_verified") is True, "Experiment plan lacks verified censor masks")
    require(plan.get("runner_sha256") == sha256_file(Path(__file__)),
            "Experiment runner changed since plan creation; regenerate the plan first")
    require(plan.get("training_input_sha256") == sha256_file(root / plan["training_input"]),
            "Frozen training input changed since plan creation")
    check_training_dependencies()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(root / "src"))
    import orgu.train_encoder as trainer

    feature_context = read_json(root / CONTEXT_PATH)
    model_id = feature_context.get("model_id")
    model_revision = feature_context.get("model_revision")
    require(isinstance(model_id, str) and isinstance(model_revision, str),
            "Feature context lacks a pinned encoder identity")
    output_rel = output_root / "runs" / run_id
    output_path = root / output_rel
    experiment_record = {
        "run_id": run_id,
        "family": job["family"],
        "rho": job["rho"],
        "censor_rate": job.get("censor_rate", 0.0),
        "mask_seed": job.get("mask_seed"),
        "training_seed": job["seed"],
        "mask_sha256": job.get("mask_sha256"),
        "removed_positive_assignments": job.get("removed_positive_assignments", 0),
        "synthetic_censoring_pilot_only": job["family"] == "synthetic_censoring_pilot",
        "validation_gold_status": "unknown_distant_supervision",
        "test_read": False,
    }
    manifest_path = output_path / "run_manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        require(existing.get("reviewer_experiment") == experiment_record,
                f"Existing run is not provenance-matched; preserving it: {manifest_path}")
    elif output_path.exists():
        raise RuntimeError(f"Partial run directory exists; preserve and inspect it: {output_path}")

    mask_by_id = {}
    if job.get("mask_path"):
        mask_path = root / job["mask_path"]
        require(mask_path.is_file() and sha256_file(mask_path) == job["mask_sha256"],
                "Censoring mask is missing or has changed")
        for row in read_jsonl(mask_path):
            candidate_id = row.get("candidate_id")
            removed_labels = row.get("removed_labels")
            require(isinstance(candidate_id, str) and candidate_id not in mask_by_id,
                    "Invalid or duplicate candidate in censoring mask")
            require(isinstance(removed_labels, list) and removed_labels,
                    "Censoring mask contains an empty removal entry")
            mask_by_id[candidate_id] = set(removed_labels)

    original_loader = trainer.load_feature_data
    original_weight = trainer.UNKNOWN_LABEL_WEIGHT
    effective_train_label_rows = []

    def load_with_censoring(feature_dir):
        nonlocal effective_train_label_rows
        artifacts = original_loader(feature_dir)
        applied = set()
        for row in artifacts["train_rows"]:
            removed = mask_by_id.get(row["id"], set())
            require(removed <= set(row["labels"]), f"Mask removes unrecorded labels: {row['id']}")
            if removed:
                row["labels"] = [label for label in row["labels"] if label not in removed]
                applied.add(row["id"])
        require(applied == set(mask_by_id), "Mask candidate IDs do not align with frozen training data")
        effective_train_label_rows = [
            (row["id"], sorted(row["labels"])) for row in artifacts["train_rows"]
        ]
        return artifacts

    trainer.load_feature_data = load_with_censoring
    trainer.UNKNOWN_LABEL_WEIGHT = job["rho"] if job["loss"] == "L1" else original_weight
    try:
        args = trainer.build_parser().parse_args([
            "--feature-dir", str(FEATURE_DIR), "--arm", "C2", "--loss", job["loss"],
            "--seed", str(job["seed"]), "--output-dir", str(output_rel),
            "--pretrained-model", model_id, "--pretrained-model-revision", model_revision,
            "--device", device,
        ])
        trainer.train(args)
    finally:
        trainer.load_feature_data = original_loader
        trainer.UNKNOWN_LABEL_WEIGHT = original_weight

    trained_manifest = read_json(manifest_path)
    trained_manifest["reviewer_experiment"] = experiment_record
    trained_manifest["effective_training_label_assignments_sha256"] = sha256_bytes(
        json.dumps(effective_train_label_rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    trained_manifest["reviewer_runner_sha256"] = sha256_file(Path(__file__))
    manifest_path.write_bytes(json_bytes(trained_manifest))
    print(f"Completed validation-only run: {output_rel}")
    print("No test split was opened. Synthetic masking remains a non-gold pilot.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Create train-only censor masks and the run matrix; default when no run ID is supplied")
    parser.add_argument("--run-id", help="Execute exactly one prepared run, never a test evaluation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"), default="auto")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    root = Path.cwd()
    output_root = Path(args.output_root)
    require(not output_root.is_absolute() and ".." not in output_root.parts,
            "Output root must be a repository-relative path")
    resolved_root = root.resolve()
    resolved_output = (root / output_root).resolve()
    require(resolved_output != resolved_root and resolved_output.is_relative_to(resolved_root),
            "Output root must remain inside the workspace")
    output_root = resolved_output.relative_to(resolved_root)
    if args.run_id:
        execute(root, output_root, args.run_id, args.device)
        return 0
    plan = prepare(root, output_root)
    print(f"Prepared {len(plan['masks'])} train-only masks and {plan['job_count']} run specifications.")
    print(f"Plan: {output_root / 'experiment_plan.json'}")
    print("Training has not run; test files were not opened.")
    print("M5 remains ineligible until suitable external gold data are supplied.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[reviewer-pipeline] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)