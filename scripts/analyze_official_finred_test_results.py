#!/usr/bin/env python3
"""Audit and summarize the post-hoc official FinRED result bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_m1m4_test_archive import (
    ARMS,
    RUN_NAMES,
    SEEDS,
    calculate_macro_average_precision,
    calculate_metrics,
    check_close,
    two_way_posthoc_sensitivity_bootstrap,
)
from scripts.run_official_finred_test import (
    EXPECTED_SOURCE_COUNTS,
    EXPECTED_SOURCE_HASHES,
    THRESHOLD_POLICIES,
)

EXPECTED_ARCHIVE_MEMBERS = {"README.txt", "scores.npz", "summary.json"}
DEFAULT_RESULTS_ARCHIVE = "results/official_finred_test_results.zip"
ORIGINAL_RESULTS_ARCHIVE = "results/official_finred_test_results_original.zip"
ORIGINAL_RESULTS_ARCHIVE_SHA256 = "99bad88417cbde527673364452238a6f8abbc147588361ec581cdb56dcc976e0"
EXPECTED_ARRAYS = {
    "scores",
    "gold",
    "labels",
    "run_names",
    "seeds",
    "arms",
    "candidate_ids",
    "record_indices",
}
OPERATING_METRICS = (
    "macro_f1_fixed_inventory",
    "micro_f1",
    "micro_precision",
    "micro_recall",
    "exact_set_match",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(value: str) -> Path:
    path = Path(value)
    require(not path.is_absolute(), f"Path must be repository-relative: {path}")
    require(".." not in path.parts, f"Path cannot contain '..': {path}")
    resolved = (ROOT / path).resolve()
    require(resolved.is_relative_to(ROOT), f"Path escapes the repository: {path}")
    return resolved


def paired_summary(values_by_run: dict[str, dict[str, float]], metric: str) -> dict[str, Any]:
    values_by_arm = {
        arm: [values_by_run[f"{arm}_seed{seed}"][metric] for seed in SEEDS]
        for arm in ARMS
    }
    differences = {
        seed: values_by_run[f"NegDW-BCE_seed{seed}"][metric]
        - values_by_run[f"BCE_seed{seed}"][metric]
        for seed in SEEDS
    }
    return {
        "arms": {
            arm: {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
            }
            for arm, values in values_by_arm.items()
        },
        "paired_difference_negdw_minus_bce": {
            "mean": statistics.mean(differences.values()),
            "sample_sd": statistics.stdev(differences.values()),
            "by_seed": differences,
        },
    }


def analyze_archive(archive_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, "ZIP CRC check failed")
        require(set(archive.namelist()) == EXPECTED_ARCHIVE_MEMBERS,
                f"Unexpected result archive members: {archive.namelist()}")
        summary = json.loads(archive.read("summary.json"))
        npz_data = archive.read("scores.npz")

    require(summary.get("format") == "orgu-official-finred-test-evaluation-v1",
            "Unexpected result bundle format")
    archive_prior_inspection = summary.get("test_previously_inspected")
    require(archive_prior_inspection is False,
            "The result archive must record no prior official-test use")
    history_correction = summary.get("history_correction")
    original_summary = None
    if history_correction is None:
        require(summary.get("prior_team_use_author_confirmation_date") == "2026-10-30",
                "The result archive is missing the author-confirmed first-use record")
    else:
        require(isinstance(history_correction, dict),
                "The archive history-correction record is malformed")
        require(history_correction.get("correction_date") == "2026-10-30",
                "Unexpected official-test history-correction date")
        require(history_correction.get("previous_archive_sha256") == ORIGINAL_RESULTS_ARCHIVE_SHA256,
                "History correction does not identify the preserved original archive")
        require(history_correction.get("previous_archive_test_previously_inspected") is True,
                "History correction does not preserve the original archive's superseded flag")
        require(history_correction.get("scores_npz_modified") is False,
                "History correction must not modify the score arrays")
    require(summary.get("test_used_for_training_checkpoint_or_threshold_selection") is False,
            "The archive indicates test use for training/checkpoint/threshold selection")
    require(summary.get("threshold_policies") == THRESHOLD_POLICIES,
            "Threshold policies differ from the frozen evaluator settings")
    require(summary["dataset"]["source_hashes"] == EXPECTED_SOURCE_HASHES,
            "Official test source hashes differ from the audited source")
    require(summary["dataset"]["source_bundle_sha256"] == hashlib.sha256(
        json.dumps(summary["dataset"]["source_hashes"], sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest(), "Source bundle digest does not match its recorded file hashes")
    require({key: summary["dataset"][key] for key in EXPECTED_SOURCE_COUNTS}
            == EXPECTED_SOURCE_COUNTS, "Official test source counts differ")

    if history_correction is not None:
        original_archive_path = repository_path(ORIGINAL_RESULTS_ARCHIVE)
        require(file_sha256(original_archive_path) == ORIGINAL_RESULTS_ARCHIVE_SHA256,
                "Preserved original result archive hash differs")
        with zipfile.ZipFile(original_archive_path) as original_archive:
            require(original_archive.testzip() is None, "Original result ZIP CRC check failed")
            original_summary = json.loads(original_archive.read("summary.json"))
            original_scores = original_archive.read("scores.npz")
        require(original_summary.get("test_previously_inspected") is True,
                "The preserved original archive does not contain the superseded history flag")
        require(original_scores == npz_data,
                "Corrected archive score arrays differ from the preserved original")

    with np.load(io.BytesIO(npz_data), allow_pickle=False) as arrays:
        require(set(arrays.files) == EXPECTED_ARRAYS,
                f"Unexpected score archive arrays: {arrays.files}")
        scores = arrays["scores"].copy()
        gold_raw = arrays["gold"].copy()
        labels = arrays["labels"].tolist()
        run_names = arrays["run_names"].tolist()
        seeds = arrays["seeds"].tolist()
        arms = arrays["arms"].tolist()
        candidate_ids = arrays["candidate_ids"].tolist()
        record_indices = arrays["record_indices"].astype(np.int64)

    gold = gold_raw.astype(bool)
    require(scores.shape == (20, EXPECTED_SOURCE_COUNTS["candidates"], 29),
            f"Unexpected score matrix shape: {scores.shape}")
    require(gold.shape == scores.shape[1:], f"Unexpected gold matrix shape: {gold.shape}")
    require(set(np.unique(gold_raw).tolist()) <= {0, 1}, "Gold matrix is not binary")
    require(np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all(),
            "Scores must be finite probabilities")
    require(labels == summary["feature_assets"]["labels_in_score_order"],
            "Score label order differs from the result summary")
    require(len(labels) == len(set(labels)) == 29, "Expected 29 unique relations")
    require(run_names == list(RUN_NAMES), "Run order or paired run inventory differs")
    require(len(set(candidate_ids)) == len(candidate_ids), "Candidate IDs are duplicated")
    require(len(set(record_indices.tolist())) == EXPECTED_SOURCE_COUNTS["records"],
            "Source-record cluster count differs from the official record count")
    require((gold.sum(axis=1) > 0).all(), "Found candidate with no recorded positive labels")
    require(hashlib.sha256("\n".join(candidate_ids).encode("utf-8")).hexdigest()
            == summary["dataset"]["candidate_order_sha256"],
            "Candidate order digest differs from the summary")

    support = {label: int(gold[:, index].sum()) for index, label in enumerate(labels)}
    require(support == summary["dataset"]["positive_candidate_support_by_label"],
            "Per-relation support differs from the summary")
    require(min(support.values()) >= 5,
            "The official-test support-filtered macro-F1 should include all 29 labels")
    require(len(summary["checkpoint_runs"]) == len(summary["per_run_results"]) == 20,
            "Expected twenty checkpoint result records")

    run_metrics: dict[str, dict[str, Any]] = {}
    macro_ap_by_run: dict[str, float] = {}
    for run_index, run_name in enumerate(run_names):
        arm = arms[run_index]
        seed = int(seeds[run_index])
        require(run_name == f"{arm}_seed{seed}", f"Run identity mismatch: {run_name}")
        result = summary["per_run_results"][run_name]
        checkpoint = summary["checkpoint_runs"][run_name]
        require(result["arm"] == checkpoint["arm"] == arm,
                f"Arm metadata mismatch for {run_name}")
        require(result["seed"] == checkpoint["seed"] == seed,
                f"Seed metadata mismatch for {run_name}")
        require(result["checkpoint_sha256"] == checkpoint["checkpoint_sha256"],
                f"Checkpoint digest mismatch for {run_name}")
        require(result["run_manifest_sha256"] == checkpoint["run_manifest_sha256"],
                f"Run manifest digest mismatch for {run_name}")
        run_metrics[run_name] = {"operating_points": {}, "per_label": {}}
        for policy_name, thresholds in THRESHOLD_POLICIES.items():
            metrics = calculate_metrics(gold, scores[run_index] >= thresholds[arm], labels)
            archived = result["operating_points"][policy_name]
            for metric in OPERATING_METRICS:
                check_close(archived[metric], metrics[metric], f"{run_name}/{policy_name}/{metric}")
            for label in labels:
                archived_label = archived["per_label"][label]
                calculated_label = metrics["per_class"][label]
                for metric in ("support", "predicted", "precision", "recall", "f1"):
                    check_close(archived_label[metric], calculated_label[metric],
                                f"{run_name}/{policy_name}/{label}/{metric}")
                if policy_name == "shared_0_5":
                    run_metrics[run_name]["per_label"][label] = calculated_label
            run_metrics[run_name]["operating_points"][policy_name] = metrics

        average_precision = calculate_macro_average_precision(gold, scores[run_index], labels)
        archived_ap = result["threshold_independent"]["macro_average_precision_fixed_inventory"]
        check_close(archived_ap, average_precision["macro_average_precision_fixed_inventory"],
                    f"{run_name}/macro_average_precision")
        macro_ap_by_run[run_name] = average_precision["macro_average_precision_fixed_inventory"]

    operating_summary = {}
    for policy_name in THRESHOLD_POLICIES:
        operating_summary[policy_name] = {}
        for metric in OPERATING_METRICS:
            values_by_run = {
                run_name: {
                    metric: run_data["operating_points"][policy_name][metric]
                }
                for run_name, run_data in run_metrics.items()
            }
            operating_summary[policy_name][metric] = paired_summary(values_by_run, metric)

    ap_values_by_run = {
        run_name: {"macro_ap": value}
        for run_name, value in macro_ap_by_run.items()
    }
    ap_summary = paired_summary(ap_values_by_run, "macro_ap")
    run_arrays = {
        run_name: (gold, scores[index]) for index, run_name in enumerate(run_names)
    }
    record_ids = [str(index) for index in record_indices]
    bootstrap = two_way_posthoc_sensitivity_bootstrap(
        run_arrays, record_ids, labels, THRESHOLD_POLICIES
    )
    bootstrap["sampling"] = (
        f"For each replicate, sample {len(SEEDS)} paired seed indices and "
        f"{len(set(record_ids))} source-record clusters with replacement; use the same draws "
        "for both arms and every metric. Thresholds remain frozen at their validation-selected values."
    )
    bootstrap["cluster_count"] = len(set(record_ids))
    posthoc_note = (
        "The team had not used the official test before this study (author confirmation, 2026-10-30); "
        "its first evaluation followed the ten-seed expansion after initial results on the previously "
        "inspected entity-disjoint test. Results remain exploratory; this split is not entity-disjoint "
        "or an external dataset."
    )
    bootstrap["interpretation"] = posthoc_note
    for metric_result in bootstrap["metrics"].values():
        metric_result["exact_seed_sign_flip_test"]["interpretation"] = posthoc_note

    per_relation = []
    for label in labels:
        values_by_run = {
            run_name: {"f1": run_data["per_label"][label]["f1"]}
            for run_name, run_data in run_metrics.items()
        }
        per_relation.append({
            "relation": label,
            "support": support[label],
            **paired_summary(values_by_run, "f1"),
        })

    return {
        "input_archive": archive_path.relative_to(ROOT).as_posix(),
        "archive_sha256": file_sha256(archive_path),
        "history_correction": {
            "author_confirmation_date": (
                history_correction["correction_date"] if history_correction is not None
                else summary["prior_team_use_author_confirmation_date"]
            ),
            "team_prior_official_test_use": False,
            "original_uploaded_archive": (
                ORIGINAL_RESULTS_ARCHIVE if history_correction is not None else None
            ),
            "original_uploaded_archive_sha256": (
                ORIGINAL_RESULTS_ARCHIVE_SHA256 if history_correction is not None else None
            ),
            "original_uploaded_archive_superseded_history_flag": (
                original_summary["test_previously_inspected"]
                if original_summary is not None else None
            ),
            "score_arrays_byte_identical": (
                True if history_correction is not None else None
            ),
        },
        "analysis_code_sha256": file_sha256(Path(__file__)),
        "bundle_created_utc": summary["created_utc"],
        "evaluator_code_sha256": summary["evaluator_code_sha256"],
        "inference_runtime": {
            "device": summary["device"],
            "python_version": summary["python_version"],
            "numpy_version": summary["numpy_version"],
            "torch_version": summary["torch_version"],
        },
        "source_bundle_sha256": summary["dataset"]["source_bundle_sha256"],
        "source_hashes": summary["dataset"]["source_hashes"],
        "dataset": {
            "name": summary["dataset"]["name"],
            "records": summary["dataset"]["records"],
            "triples": summary["dataset"]["triples"],
            "candidates": summary["dataset"]["candidates"],
            "source_record_clusters": len(set(record_ids)),
            "labels": len(labels),
            "positive_candidate_support_by_label": support,
            "candidate_order_sha256": summary["dataset"]["candidate_order_sha256"],
        },
        "protocol": {
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "threshold_policies": THRESHOLD_POLICIES,
            "checkpoint_selection": summary["selection_protocol"],
            "test_previously_inspected": False,
            "original_uploaded_archive_superseded_history_flag": (
                original_summary["test_previously_inspected"]
                if original_summary is not None else None
            ),
            "test_history_note": (
                "The author confirmed on 2026-10-30 that the team had not used the official test before this study. The original uploaded archive's positive history flag reflected the mistaken earlier declaration and is superseded; the corrected archive preserves identical score arrays."
                if history_correction is not None else
                "The author-confirmed run record states that the team had not used the official test before this study."
            ),
            "test_used_for_training_checkpoint_or_threshold_selection": summary[
                "test_used_for_training_checkpoint_or_threshold_selection"],
            "score_interpretation": summary["score_interpretation"],
        },
        "operating_point_summary": operating_summary,
        "threshold_independent_macro_ap": ap_summary,
        "bootstrap": bootstrap,
        "per_relation_shared_0_5": per_relation,
    }


def render_markdown(report: dict[str, Any]) -> str:
    history = report["history_correction"]
    history_lines = [
        f"- Prior team use: no, per author confirmation on {history['author_confirmation_date']}; this is the first official-test evaluation in the study."
    ]
    if history["original_uploaded_archive"] is not None:
        history_lines.extend([
            f"- Corrected archive: `{report['input_archive']}`; original upload preserved as `{history['original_uploaded_archive']}` (SHA-256 `{history['original_uploaded_archive_sha256']}`).",
            "- The original uploaded archive's positive history flag reflected the mistaken earlier declaration; only history metadata was corrected, and the score arrays are byte-identical.",
        ])
    lines = [
        "# First-Time Official FinRED Test Evaluation: 10-Seed Analysis",
        "",
        f"- Result ZIP SHA-256: `{report['archive_sha256']}`.",
        f"- Input archive: `{report['input_archive']}`.",
        f"- Dataset: {report['dataset']['records']:,} source records, "
        f"{report['dataset']['triples']:,} triples, {report['dataset']['candidates']:,} directed pair candidates, "
        f"{report['dataset']['labels']} relations.",
        f"- Paired seeds: {', '.join(map(str, report['protocol']['seeds']))}; "
        f"source-record clusters: {report['dataset']['source_record_clusters']:,}.",
        *history_lines,
        "- This standard FinRED split is not entity-disjoint or an external dataset; scores indicate agreement with recorded annotations.",
        "",
        "## Operating-Point Results",
        "",
        "All entries are mean (sample SD) across ten seeds. Paired differences are NegDW-BCE minus BCE.",
        "",
        "| Policy | BCE threshold | NegDW threshold | BCE macro-F1 | NegDW macro-F1 | Paired difference (SD) | Two-way 95% CI | Exact sign-flip p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy_name, thresholds in report["protocol"]["threshold_policies"].items():
        bootstrap_result = report["bootstrap"]["metrics"][policy_name]
        paired = report["operating_point_summary"][policy_name]["macro_f1_fixed_inventory"][
            "paired_difference_negdw_minus_bce"]
        ci = bootstrap_result["two_way_bootstrap_95_percentile_ci"]
        lines.append(
            f"| {policy_name} | {thresholds['BCE']:.6f} | {thresholds['NegDW-BCE']:.6f} | "
            f"{report['operating_point_summary'][policy_name]['macro_f1_fixed_inventory']['arms']['BCE']['mean']:.4f} "
            f"({report['operating_point_summary'][policy_name]['macro_f1_fixed_inventory']['arms']['BCE']['sample_sd']:.4f}) | "
            f"{report['operating_point_summary'][policy_name]['macro_f1_fixed_inventory']['arms']['NegDW-BCE']['mean']:.4f} "
            f"({report['operating_point_summary'][policy_name]['macro_f1_fixed_inventory']['arms']['NegDW-BCE']['sample_sd']:.4f}) | "
            f"{paired['mean']:+.4f} ({paired['sample_sd']:.4f}) | "
            f"[{ci['lower']:+.4f}, {ci['upper']:+.4f}] | "
            f"{bootstrap_result['exact_seed_sign_flip_test']['p_value_two_sided']:.6f} |"
        )
    lines.extend([
        "",
        "## Shared-0.5 Secondary Metrics",
        "",
        "| Metric | BCE mean (SD) | NegDW mean (SD) | Paired difference (SD) |",
        "|---|---:|---:|---:|",
    ])
    for metric in OPERATING_METRICS:
        result = report["operating_point_summary"]["shared_0_5"][metric]
        lines.append(
            f"| {metric} | {result['arms']['BCE']['mean']:.4f} ({result['arms']['BCE']['sample_sd']:.4f}) | "
            f"{result['arms']['NegDW-BCE']['mean']:.4f} ({result['arms']['NegDW-BCE']['sample_sd']:.4f}) | "
            f"{result['paired_difference_negdw_minus_bce']['mean']:+.4f} "
            f"({result['paired_difference_negdw_minus_bce']['sample_sd']:.4f}) |"
        )
    ap = report["threshold_independent_macro_ap"]
    ap_bootstrap = report["bootstrap"]["metrics"]["macro_average_precision_fixed_inventory"]
    ap_ci = ap_bootstrap["two_way_bootstrap_95_percentile_ci"]
    lines.extend([
        "",
        "## Threshold-Independent Macro-AP",
        "",
        f"BCE {ap['arms']['BCE']['mean']:.4f} ({ap['arms']['BCE']['sample_sd']:.4f}); "
        f"NegDW-BCE {ap['arms']['NegDW-BCE']['mean']:.4f} ({ap['arms']['NegDW-BCE']['sample_sd']:.4f}); "
        f"paired difference {ap['paired_difference_negdw_minus_bce']['mean']:+.4f} "
        f"({ap['paired_difference_negdw_minus_bce']['sample_sd']:.4f}); "
        f"two-way 95% CI [{ap_ci['lower']:+.4f}, {ap_ci['upper']:+.4f}], "
        f"exact sign-flip p={ap_bootstrap['exact_seed_sign_flip_test']['p_value_two_sided']:.6f}.",
        "",
        "## Uncertainty and Scope",
        "",
        f"Two-way bootstrap: {report['bootstrap']['replicates']:,} paired resamples of training seeds and source-record clusters "
        f"(NumPy PCG64; RNG seed {report['bootstrap']['rng_seed']}); thresholds are frozen within replicates. "
        "Exact sign-flip tests use ten seed-level differences and are unadjusted exploratory summaries. "
        "The official test had not been used by the team before this study, but its first evaluation followed "
        "the ten-seed expansion after initial results on the previously inspected entity-disjoint test. "
        "The analysis is exploratory and does not establish recovery of missing labels, unseen-entity "
        "generalization, or external-dataset confirmation.",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-zip", default=DEFAULT_RESULTS_ARCHIVE)
    parser.add_argument("--output-dir", default="results/analysis/official_finred_test")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    archive_path = repository_path(args.results_zip)
    output_dir = repository_path(args.output_dir)
    report = analyze_archive(archive_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "official_finred_test_analysis_10seeds.json"
    markdown_path = output_dir / "official_finred_test_analysis_10seeds.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Audit passed: {json_path}")
    print(f"Report written: {markdown_path}")
    print(f"ZIP SHA-256: {report['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())