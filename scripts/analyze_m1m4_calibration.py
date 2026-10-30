from __future__ import annotations

import hashlib
import json
import math
import tarfile
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT_PATH = Path("results/analysis/entity_disjoint/m1_m4_test_analysis_10seeds.json")
TEST_PATH = Path("data/features/finred_entity_disjoint_v1/C2/test.jsonl")
OUTPUT_JSON = Path("results/analysis/calibration_analysis_10seeds.json")
OUTPUT_FIGURE = Path("results/analysis/reliability_diagram_10seeds.pdf")
BIN_COUNT = 10
ARMS = ("BCE", "NegDW-BCE")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(data: bytes, description: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate(data.splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {description}:{line_number}: {exc}") from exc
            check(isinstance(row, dict), f"Expected object at {description}:{line_number}")
            rows.append(row)
    return rows


def archive_members(path: Path) -> dict[str, bytes]:
    blobs = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            check(not member_path.is_absolute() and ".." not in member_path.parts,
                  f"Unsafe archive path: {member.name}")
            check(member.isfile() and member.name not in blobs,
                  f"Unexpected or duplicate archive member: {member.name}")
            stream = archive.extractfile(member)
            check(stream is not None, f"Could not read archive member: {member.name}")
            blobs[member.name] = stream.read()
    return blobs


def fixed_width_ece(targets: np.ndarray, probabilities: np.ndarray,
                    bin_count: int = BIN_COUNT) -> tuple[float, list[dict]]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    check(targets.shape == probabilities.shape and targets.size > 0,
          "ECE requires nonempty, aligned targets and probabilities")
    check(np.isfinite(probabilities).all() and np.logical_and(probabilities >= 0, probabilities <= 1).all(),
          "Probabilities must be finite values in [0, 1]")
    bin_indices = np.minimum((probabilities * bin_count).astype(np.int64), bin_count - 1)
    bins = []
    ece = 0.0
    for index in range(bin_count):
        mask = bin_indices == index
        count = int(mask.sum())
        if not count:
            continue
        mean_probability = float(probabilities[mask].mean())
        observed_frequency = float(targets[mask].mean())
        ece += count / targets.size * abs(mean_probability - observed_frequency)
        bins.append({
            "bin_index": index,
            "lower_edge": index / bin_count,
            "upper_edge": (index + 1) / bin_count,
            "count": count,
            "mean_predicted_probability": mean_probability,
            "observed_recorded_positive_frequency": observed_frequency,
        })
    return float(ece), bins


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    labels = report["labels"]
    seeds = [int(seed) for seed in report["seeds"]]
    test_bytes = TEST_PATH.read_bytes()
    check(sha256(test_bytes) == report["test_input"]["sha256"],
          "Entity-disjoint test file hash differs from the audited report")
    source_rows = read_jsonl(test_bytes, str(TEST_PATH))
    check(len(source_rows) == report["test_input"]["candidate_count"] == 701,
          "Unexpected entity-disjoint test candidate count")

    archive_blobs = {}
    for archive_name, metadata in report["archives"].items():
        archive_path = Path(archive_name)
        archive_bytes = archive_path.read_bytes()
        check(sha256(archive_bytes) == metadata["sha256"],
              f"Archive hash mismatch: {archive_name}")
        archive_blobs.update(archive_members(archive_path))

    run_hashes = report["run_hashes"]
    check(len(run_hashes) == len(ARMS) * len(seeds) == 20,
          "Expected ten paired seeds for two loss arms")
    runs: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        for seed in seeds:
            run_name = f"{arm}_seed{seed}"
            member_name = f"{run_name}/test_predictions.jsonl"
            check(member_name in archive_blobs, f"Missing archived predictions: {member_name}")
            prediction_bytes = archive_blobs[member_name]
            check(sha256(prediction_bytes) == run_hashes[run_name]["test_predictions_sha256"],
                  f"Prediction hash mismatch: {run_name}")
            predictions = read_jsonl(prediction_bytes, member_name)
            check(len(predictions) == len(source_rows), f"Prediction count mismatch: {run_name}")
            gold = np.zeros((len(source_rows), len(labels)), dtype=np.uint8)
            scores = np.zeros((len(source_rows), len(labels)), dtype=np.float64)
            for row_index, (source, prediction) in enumerate(zip(source_rows, predictions)):
                check(prediction.get("candidate_id") == source.get("id"),
                      f"Candidate order mismatch: {run_name}:{row_index}")
                check(prediction.get("candidate_index") == row_index,
                      f"Candidate index mismatch: {run_name}:{row_index}")
                check(prediction.get("record_id") == source.get("record_id"),
                      f"Record ID mismatch: {run_name}:{row_index}")
                target = [int(label in source.get("labels", [])) for label in labels]
                check(prediction.get("gold_targets") == target,
                      f"Gold target mismatch: {run_name}:{row_index}")
                raw_scores = prediction.get("scores")
                check(isinstance(raw_scores, list) and len(raw_scores) == len(labels),
                      f"Score width mismatch: {run_name}:{row_index}")
                check(all(isinstance(value, (int, float)) and math.isfinite(float(value))
                          and 0 <= float(value) <= 1 for value in raw_scores),
                      f"Invalid probability: {run_name}:{row_index}")
                gold[row_index] = target
                scores[row_index] = raw_scores
            runs[run_name] = {"gold": gold, "scores": scores}

    method_results = {}
    pooled_bins = {}
    per_seed_micro: dict[str, list[float]] = {arm: [] for arm in ARMS}
    per_seed_macro: dict[str, list[float]] = {arm: [] for arm in ARMS}
    per_seed_values: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARMS}

    for arm in ARMS:
        pooled_gold = []
        pooled_scores = []
        for seed in seeds:
            run_name = f"{arm}_seed{seed}"
            gold = runs[run_name]["gold"]
            scores = runs[run_name]["scores"]
            micro_ece, _ = fixed_width_ece(gold, scores)
            relation_eces = [fixed_width_ece(gold[:, label_index], scores[:, label_index])[0]
                             for label_index in range(len(labels))]
            macro_ece = float(np.mean(relation_eces))
            per_seed_micro[arm].append(micro_ece)
            per_seed_macro[arm].append(macro_ece)
            per_seed_values[arm][str(seed)] = {
                "micro_ece": micro_ece,
                "macro_relation_ece": macro_ece,
                "mean_predicted_probability": float(scores.mean()),
            }
            pooled_gold.append(gold.reshape(-1))
            pooled_scores.append(scores.reshape(-1))

        all_gold = np.concatenate(pooled_gold)
        all_scores = np.concatenate(pooled_scores)
        pooled_ece, bins = fixed_width_ece(all_gold, all_scores)
        pooled_bins[arm] = bins
        method_results[arm] = {
            "micro_ece_mean_across_seeds": float(np.mean(per_seed_micro[arm])),
            "micro_ece_sample_sd_across_seeds": float(np.std(per_seed_micro[arm], ddof=1)),
            "macro_relation_ece_mean_across_seeds": float(np.mean(per_seed_macro[arm])),
            "macro_relation_ece_sample_sd_across_seeds": float(np.std(per_seed_macro[arm], ddof=1)),
            "pooled_decision_ece_for_plot": pooled_ece,
            "mean_predicted_probability_across_seeds": float(all_scores.mean()),
            "observed_recorded_positive_rate": float(all_gold.mean()),
            "pooled_decision_count": int(all_gold.size),
            "seed_results": per_seed_values[arm],
            "reliability_bins_pooled_across_seeds": bins,
        }

    paired_micro = [per_seed_micro["NegDW-BCE"][i] - per_seed_micro["BCE"][i]
                    for i in range(len(seeds))]
    paired_macro = [per_seed_macro["NegDW-BCE"][i] - per_seed_macro["BCE"][i]
                    for i in range(len(seeds))]
    result = {
        "status": "posthoc_exploratory",
        "dataset_split": "FinRED entity-disjoint test; 701 candidates; previously inspected",
        "labels": labels,
        "seeds": seeds,
        "binning": {
            "method": "10 fixed-width bins over positive-label sigmoid probabilities",
            "edges": [index / BIN_COUNT for index in range(BIN_COUNT + 1)],
            "micro_ece": "sum_b (n_b/N) * abs(mean_probability_b - mean_recorded_target_b), pooling the 29 label-candidate decisions within each seed",
            "macro_relation_ece": "unweighted mean of the 29 relation-specific 10-bin ECE values, calculated within each seed",
            "reporting": "mean and sample SD over ten seeds; no calibration transform or threshold selection",
            "reliability_plot": "bin points pool score-target pairs from the ten seed-specific models; repeated candidate targets across seeds are descriptive and not independent observations",
        },
        "methods": method_results,
        "paired_difference_NegDW_minus_BCE": {
            "micro_ece_mean": float(np.mean(paired_micro)),
            "micro_ece_sample_sd": float(np.std(paired_micro, ddof=1)),
            "macro_relation_ece_mean": float(np.mean(paired_macro)),
            "macro_relation_ece_sample_sd": float(np.std(paired_macro, ddof=1)),
            "per_seed_micro_ece": {str(seed): paired_micro[index] for index, seed in enumerate(seeds)},
        },
        "input_provenance": {
            "test_sha256": report["test_input"]["sha256"],
            "prediction_archive_sha256": report["archives"]["results/archives/m1m4_test_predictions.tar.gz"]["sha256"],
            "extra_seed_archive_sha256": report["archives"]["results/archives/m1m4_extra_seed_artifacts_7seeds.tar.gz"]["sha256"],
            "prediction_hashes_checked_against": str(REPORT_PATH),
        },
        "output_figure": str(OUTPUT_FIGURE),
    }

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#666666",
            label="Perfect calibration")
    palette = {"BCE": "#1768AC", "NegDW-BCE": "#D1495B"}
    for arm in ARMS:
        bins = pooled_bins[arm]
        x = [item["mean_predicted_probability"] for item in bins]
        y = [item["observed_recorded_positive_frequency"] for item in bins]
        ece = method_results[arm]["micro_ece_mean_across_seeds"]
        sd = method_results[arm]["micro_ece_sample_sd_across_seeds"]
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=5.5,
                color=palette[arm], label=f"{arm} (ECE {ece:.3f} $\\pm$ {sd:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
           ylabel="Observed recorded-positive frequency",
           title="Reliability on the entity-disjoint test")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    plt.close(fig)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for arm in ARMS:
        metrics = method_results[arm]
        print(
            f"{arm}: micro-ECE={metrics['micro_ece_mean_across_seeds']:.6f} "
            f"(SD {metrics['micro_ece_sample_sd_across_seeds']:.6f}); "
            f"macro relation-ECE={metrics['macro_relation_ece_mean_across_seeds']:.6f} "
            f"(SD {metrics['macro_relation_ece_sample_sd_across_seeds']:.6f}); "
            f"mean p={metrics['mean_predicted_probability_across_seeds']:.6f}; "
            f"observed={metrics['observed_recorded_positive_rate']:.6f}"
        )
    print(
        "NegDW-BCE minus BCE micro-ECE: "
        f"{result['paired_difference_NegDW_minus_BCE']['micro_ece_mean']:.6f} "
        f"(SD {result['paired_difference_NegDW_minus_BCE']['micro_ece_sample_sd']:.6f})"
    )
    print(f"Wrote {OUTPUT_JSON} and {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()