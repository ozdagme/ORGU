from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path, PurePosixPath

from orgu.metrics import multilabel_metrics


ARCHIVE = Path("results/archives/m1m4_validation_artifacts.tar.gz")
AGGREGATE = Path("results/archives/m1m4_validation_manifest_2.json")
TRAIN = Path("data/features/finred_entity_disjoint_v1/C2/train.jsonl")
VALIDATION = Path("data/features/finred_entity_disjoint_v1/C2/validation.jsonl")
SCHEMA = Path("data/features/finred_entity_disjoint_v1/label_schema.json")
CONTEXT = Path("data/features/finred_entity_disjoint_v1/context_manifest.json")
TOKENIZER = Path("data/features/finred_entity_disjoint_v1/tokenizer.json")
OUT = Path("results/analysis/validation")
SEEDS = (13, 42, 87)
ARMS = ("BCE", "NegDW-BCE")
TOL = 1e-12
RHO = 0.1
THEORY_THRESHOLD = RHO / (1 + RHO)


def check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(data: bytes, where: str) -> dict:
    value = json.loads(data.decode("utf-8"))
    check(isinstance(value, dict), f"Expected JSON object: {where}")
    return value


def load_jsonl(data: bytes, where: str) -> list[dict]:
    rows = []
    for index, line in enumerate(data.splitlines(), 1):
        if line.strip():
            rows.append(load_json(line, f"{where}:{index}"))
    return rows


def ordered_id_hash(rows: list[dict]) -> str:
    return digest("\n".join(str(row.get("id", "")) for row in rows).encode("utf-8"))


def compare_nested(expected, observed, where: str) -> None:
    if isinstance(expected, dict):
        check(isinstance(observed, dict) and expected.keys() == observed.keys(),
              f"Metric keys differ: {where}")
        for key in expected:
            compare_nested(expected[key], observed[key], f"{where}.{key}")
    elif isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if isinstance(expected, int) and isinstance(observed, int):
            check(expected == observed, f"Metric differs: {where}")
        else:
            check(math.isclose(float(expected), float(observed), rel_tol=TOL, abs_tol=TOL),
                  f"Metric differs: {where}: {expected} != {observed}")
    else:
        check(expected == observed, f"Metric differs: {where}")


def read_archive(aggregate: dict, source_rows: list[dict], labels: list[str]):
    expected_names = {
        f"runs/m1m4_entity_disjoint_v1/{arm}_seed{seed}/{filename}"
        for arm in ARMS for seed in SEEDS
        for filename in ("run_manifest.json", "validation_predictions.jsonl", "validation_metrics.json")
    }
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            check(not path.is_absolute() and ".." not in path.parts,
                  f"Unsafe archive path: {member.name}")
            check(member.isfile(), f"Unexpected non-file archive entry: {member.name}")
            check(member.name not in members, f"Duplicate archive entry: {member.name}")
            members[member.name] = member
        check(set(members) == expected_names, "Unexpected archive contents")
        blobs = {}
        for name, member in members.items():
            stream = archive.extractfile(member)
            check(stream is not None, f"Cannot read archive member: {name}")
            blobs[name] = stream.read()

    run_entries = {(entry["loss"], entry["seed"]): entry for entry in aggregate["runs"]}
    check(len(run_entries) == 6 and set(run_entries) == {(a, s) for a in ARMS for s in SEEDS},
          "Aggregate manifest does not identify six unique arm/seed runs")
    check(aggregate.get("status") == "completed", "Aggregate manifest is not completed")
    check(aggregate.get("all_six_candidate_and_gold_targets_aligned") is True,
          "Aggregate manifest does not confirm candidate/gold alignment")
    check(aggregate.get("test_file_read") is False and aggregate.get("official_test_read") is False,
          "Aggregate manifest does not preserve test lock")

    all_runs, reports = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            entry = run_entries[(arm, seed)]
            folder = f"runs/m1m4_entity_disjoint_v1/{arm}_seed{seed}"
            run_name, pred_name, metric_name = (
                f"{folder}/run_manifest.json",
                f"{folder}/validation_predictions.jsonl",
                f"{folder}/validation_metrics.json",
            )
            run_bytes, pred_bytes, metric_bytes = (blobs[n] for n in (run_name, pred_name, metric_name))
            run = load_json(run_bytes, run_name)
            stored_metrics = load_json(metric_bytes, metric_name)
            predictions = load_jsonl(pred_bytes, pred_name)

            check(digest(run_bytes) == entry["run_manifest_sha256"], f"Run-manifest hash mismatch: {run_name}")
            check(digest(pred_bytes) == entry["validation_predictions_sha256"]
                  == run["validation_predictions_sha256"], f"Prediction hash mismatch: {pred_name}")
            check(digest(metric_bytes) == run["validation_metrics_sha256"], f"Metrics hash mismatch: {metric_name}")
            check(run.get("status") == "completed" and run.get("loss_method") == arm
                  and run.get("seed") == seed, f"Run identity/status mismatch: {run_name}")
            check(run.get("labels") == labels == aggregate["labels_in_score_vector_order"],
                  f"Label order mismatch: {run_name}")
            check(run.get("train_sha256") == aggregate["train_input"]["sha256"],
                  f"Training input hash mismatch: {run_name}")
            check(run.get("validation_sha256") == aggregate["validation_input"]["sha256"],
                  f"Validation input hash mismatch: {run_name}")
            check(run.get("validation_candidate_order_sha256")
                  == aggregate["validation_input"]["candidate_order_sha256"],
                  f"Candidate-order hash mismatch: {run_name}")
            check(run.get("validation_prediction_rows") == len(predictions) == len(source_rows) == 720,
                  f"Prediction count mismatch: {run_name}")
            check(run.get("test_file_read") is False and run.get("official_test_read") is False,
                  f"Test lock mismatch: {run_name}")
            check(entry["run_manifest"] == run_name and entry["validation_predictions"] == pred_name,
                  f"Aggregate paths mismatch: {run_name}")

            gold_sets, thresholded_sets, score_rows, targets = [], [], [], []
            for index, (source, prediction) in enumerate(zip(source_rows, predictions)):
                check(prediction.get("candidate_index") == index, f"Candidate order mismatch: {pred_name}:{index}")
                check(prediction.get("candidate_id") == source.get("id"), f"Candidate ID mismatch: {pred_name}:{index}")
                check(prediction.get("record_id") == source.get("record_id"), f"Record ID mismatch: {pred_name}:{index}")
                check(prediction.get("split") == source.get("split") == "validation",
                      f"Split mismatch: {pred_name}:{index}")
                check(prediction.get("split_sha256") == aggregate["validation_input"]["sha256"],
                      f"Split hash mismatch: {pred_name}:{index}")
                gold = set(source.get("labels", []))
                check(set(prediction.get("gold_labels", [])) == gold, f"Gold labels mismatch: {pred_name}:{index}")
                target = [int(label in gold) for label in labels]
                check(prediction.get("gold_targets") == target, f"Gold target vector mismatch: {pred_name}:{index}")
                scores = prediction.get("scores")
                check(isinstance(scores, list) and len(scores) == len(labels),
                      f"Score dimension mismatch: {pred_name}:{index}")
                check(all(isinstance(x, (int, float)) and math.isfinite(float(x)) and 0 <= x <= 1 for x in scores),
                      f"Invalid probability: {pred_name}:{index}")
                check(prediction.get("threshold") == 0.5, f"Recorded threshold mismatch: {pred_name}:{index}")
                predicted = {label for label, score in zip(labels, scores) if float(score) >= 0.5}
                check(set(prediction.get("predicted_labels", [])) == predicted,
                      f"Thresholded labels mismatch: {pred_name}:{index}")
                gold_sets.append(gold)
                thresholded_sets.append(predicted)
                score_rows.append([float(x) for x in scores])
                targets.append(target)

            metrics = multilabel_metrics(gold_sets, thresholded_sets, labels)
            compare_nested(stored_metrics["all_candidates"], metrics, f"{arm}/seed{seed}.all_candidates")
            compare_nested(stored_metrics["per_class"], metrics["per_class"], f"{arm}/seed{seed}.per_class")
            for key, filename in (("training_code_sha256", "train_encoder.py"),
                                  ("neural_code_sha256", "neural.py"),
                                  ("metrics_code_sha256", "metrics.py")):
                check(entry.get(key) == run.get(key) == aggregate["training_code_hashes"][filename],
                      f"Code-hash provenance mismatch: {run_name}:{key}")
            all_runs[(arm, seed)] = {"gold": gold_sets, "scores": score_rows, "targets": targets}
            reports.append({
                "loss": arm,
                "seed": seed,
                "run_manifest_sha256": digest(run_bytes),
                "validation_predictions_sha256": digest(pred_bytes),
                "validation_metrics_sha256": digest(metric_bytes),
                "rows": len(predictions),
                "selected_epoch": run["best_checkpoint"]["epoch"],
                "selected_checkpoint_sha256_recorded_but_blob_absent": run["best_checkpoint"]["sha256"],
                "metrics_at_0_5": metrics,
            })
    return all_runs, reports


def average_precision(scores: list[float], targets: list[int]) -> float:
    support = sum(targets)
    if not support:
        return 0.0
    ranked = sorted(zip(scores, targets), key=lambda item: item[0], reverse=True)
    tp = fp = ap = index = 0
    while index < len(ranked):
        score = ranked[index][0]
        group_size = group_positive = 0
        while index < len(ranked) and ranked[index][0] == score:
            group_positive += ranked[index][1]
            group_size += 1
            index += 1
        tp += group_positive
        fp += group_size - group_positive
        ap += (group_positive / support) * (tp / (tp + fp))
    return ap


def class_curves(run: dict, labels: list[str]) -> dict:
    curves = {}
    for j, label in enumerate(labels):
        ordered = sorted((row[j], target[j]) for row, target in zip(run["scores"], run["targets"]))
        cumulative = [0]
        for _, target in ordered:
            cumulative.append(cumulative[-1] + target)
        curves[label] = {"scores": [x[0] for x in ordered], "cum_pos": cumulative,
                         "support": cumulative[-1]}
    return curves


def f1_at(curve: dict, threshold: float) -> float:
    i = bisect.bisect_left(curve["scores"], threshold)
    predicted = len(curve["scores"]) - i
    tp = curve["cum_pos"][-1] - curve["cum_pos"][i]
    return 2 * tp / (predicted + curve["support"]) if predicted + curve["support"] else 0.0


def choose_threshold(runs: dict, labels: list[str], arm: str, relation: str | None = None):
    curves = [class_curves(runs[(arm, seed)], labels) for seed in SEEDS]
    if relation is None:
        candidates = {0.0}
        for seed_curves in curves:
            for curve in seed_curves.values():
                candidates.update(curve["scores"])
        best_score = -1.0
        best = 0.0
        for threshold in sorted(candidates | {math.nextafter(max(candidates), math.inf)}):
            value = statistics.mean(
                statistics.mean(f1_at(seed_curves[label], threshold) for label in labels)
                for seed_curves in curves
            )
            if value > best_score or (value == best_score and threshold > best):
                best, best_score = threshold, value
        return best

    candidates = {0.0}
    for seed_curves in curves:
        candidates.update(seed_curves[relation]["scores"])
    best_score = -1.0
    best = 0.0
    for threshold in sorted(candidates | {math.nextafter(max(candidates), math.inf)}):
        value = statistics.mean(f1_at(seed_curves[relation], threshold) for seed_curves in curves)
        if value > best_score or (value == best_score and threshold > best):
            best, best_score = threshold, value
    return best


def metrics_at(run: dict, labels: list[str], thresholds: list[float]) -> dict:
    predicted = [
        {label for label, score, threshold in zip(labels, scores, thresholds) if score >= threshold}
        for scores in run["scores"]
    ]
    result = multilabel_metrics(run["gold"], predicted, labels)
    aps = {
        label: average_precision([row[j] for row in run["scores"]], [row[j] for row in run["targets"]])
        for j, label in enumerate(labels)
    }
    result["macro_average_precision_fixed_inventory"] = statistics.mean(aps.values())
    result["per_class_average_precision"] = aps
    return result


def analyze_thresholds(runs: dict, labels: list[str]) -> dict:
    global_t = {arm: choose_threshold(runs, labels, arm) for arm in ARMS}
    relation_t = {arm: {label: choose_threshold(runs, labels, arm, label) for label in labels} for arm in ARMS}
    policies = {
        "fixed_0_5": {arm: [0.5] * len(labels) for arm in ARMS},
        "BCE_at_theoretical_1_over_11": {"BCE": [THEORY_THRESHOLD] * len(labels)},
        "global_tuned_on_BCE_validation": {arm: [global_t["BCE"]] * len(labels) for arm in ARMS},
        "global_tuned_on_NegDW_validation": {arm: [global_t["NegDW-BCE"]] * len(labels) for arm in ARMS},
        "per_relation_tuned_on_BCE_validation": {
            arm: [relation_t["BCE"][label] for label in labels] for arm in ARMS
        },
        "per_relation_tuned_on_NegDW_validation": {
            arm: [relation_t["NegDW-BCE"][label] for label in labels] for arm in ARMS
        },
    }
    result = {}
    for policy, arms in policies.items():
        result[policy] = {}
        for arm, thresholds in arms.items():
            by_seed = {str(seed): metrics_at(runs[(arm, seed)], labels, thresholds) for seed in SEEDS}
            summary = {}
            for metric in ("macro_f1_fixed_inventory", "micro_f1", "micro_precision", "micro_recall",
                           "exact_set_match", "macro_average_precision_fixed_inventory"):
                values = [by_seed[str(seed)][metric] for seed in SEEDS]
                summary[metric] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
                                   "by_seed": dict(zip(map(str, SEEDS), values))}
            result[policy][arm] = {
                "thresholds": thresholds if len(set(thresholds)) == 1 else dict(zip(labels, thresholds)),
                "summary": summary,
                "by_seed": by_seed,
            }
    return {"global_thresholds_selected_on_validation": global_t,
            "per_relation_thresholds_selected_on_validation": relation_t,
            "policies": result}


def save_pr_curves(path: Path, runs: dict, labels: list[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["loss", "seed", "relation", "threshold", "precision", "recall", "f1", "support", "predicted"])
        for arm in ARMS:
            for seed in SEEDS:
                run = runs[(arm, seed)]
                for j, label in enumerate(labels):
                    scores = [row[j] for row in run["scores"]]
                    targets = [row[j] for row in run["targets"]]
                    support = sum(targets)
                    thresholds = sorted(set(scores) | {math.nextafter(max(scores), math.inf)}, reverse=True)
                    for threshold in thresholds:
                        tp = sum(int(score >= threshold and target) for score, target in zip(scores, targets))
                        predicted = sum(score >= threshold for score in scores)
                        precision = tp / predicted if predicted else 0.0
                        recall = tp / support if support else 0.0
                        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                        writer.writerow([arm, seed, label, format(threshold, ".17g"), format(precision, ".17g"),
                                         format(recall, ".17g"), format(f1, ".17g"), support, predicted])
                        count += 1
    return count


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# M1/M4 Validation Artifact Audit", "",
        "**Durum:** Altı validation tahmin dosyası, yerel 720 satırlı validation girdisine karşı bağımsız doğrulandı. Betik test dosyasını açmaz; aşağıdaki eşik sonuçları validation içi tuning özetidir.", "",
        "## Bütünlük ve provenans", "",
        f"- Arşiv SHA-256: `{payload['archive_sha256']}`.",
        f"- Birleşik manifest SHA-256: `{payload['aggregate_manifest_sha256']}`.",
        f"- Denetim betiği SHA-256: `{payload['analysis_script_sha256']}`.",
        "- 6/6 koşu, her birinde 720 satır: aday indeksi/kimliği/sırası, record ID, split/hash, gold etiket/hedef, 29 olasılık, @0.5 tahmini ve run/prediction/metrics hashleri geçti.",
        "- Kaydedilmiş validation ölçütleri ham tahminlerden yeniden hesaplandı ve eşleşti.",
        "- Seçilmiş checkpoint SHA-256 değerleri manifestte kayıtlı; checkpoint blob'ları bu arşivde olmadığı için tekrar hashlenemedi.",
        f"- Eğitim/model/metrik kaynak hashleri eşleşti; runner kod hash'i: `{payload['runner_code_sha256']}` ({'eşleşti' if payload['runner_hash_matches'] else 'eşleşmedi'}). Aggregate/run manifestleri testin açılmadığını bildiriyor.", "",
        "## Sabitlenen validation eşik protokolü", "",
        "- Global eşik: üç BCE seed'inde ortalama fixed-inventory macro-F1'i ençoklayan tek değer. NegDW-BCE için paralel bir eşik ayrıca seçilir.",
        "- İlişki eşiği: her etiket için üç seed üzerindeki ortalama sınıf F1'ini ençoklayan değer.",
        "- Adaylar: 0, tüm gözlenen olasılıklar ve maksimumun bir sonraki temsil edilebilir float değeri; karar `score >= threshold`.",
        "- Eşit amaç değerinde daha yüksek eşik seçilir. Her iki yöntem de BCE-seçilmiş ve NegDW-seçilmiş eşiklerle çapraz değerlendirilir.",
        "- BCE için teorik `1/11 = 0.090909...` ve fixed-inventory macro-AP de raporlanır. AP hesaplamasında eşit skorlar aynı kesme noktasında gruplanır.",
        "- Eşikler validation üzerinde seçilip aynı validation üzerinde özetlendiğinden bu F1'ler seçim-içi (in-sample) değerlerdir; test sonucu veya tarafsız genelleme tahmini değildir.", "",
        "## Validation özetleri", "",
        "Değerler üç seed ortalaması (örneklem SD). BCE@1/11 satırında NegDW-BCE değerlendirilmez.", "",
        "| Politika | BCE macro-F1 | NegDW-BCE macro-F1 | BCE macro-AP | NegDW-BCE macro-AP |", "|---|---:|---:|---:|---:|",
    ]
    for policy, arms in payload["policies"].items():
        cells = {}
        for arm in ARMS:
            cells[arm] = {}
            for metric in ("macro_f1_fixed_inventory", "macro_average_precision_fixed_inventory"):
                if arm in arms:
                    item = arms[arm]["summary"][metric]
                    cells[arm][metric] = f"{item['mean']:.4f} ({item['sample_sd']:.4f})"
                else:
                    cells[arm][metric] = "—"
        lines.append(f"| `{policy}` | {cells['BCE']['macro_f1_fixed_inventory']} | "
                     f"{cells['NegDW-BCE']['macro_f1_fixed_inventory']} | "
                     f"{cells['BCE']['macro_average_precision_fixed_inventory']} | "
                     f"{cells['NegDW-BCE']['macro_average_precision_fixed_inventory']} |")
    lines.extend([
        "", "### Validation'da seçilen eşikler", "",
        f"- BCE global: `{payload['global_thresholds_selected_on_validation']['BCE']:.17g}`.",
        f"- NegDW-BCE global: `{payload['global_thresholds_selected_on_validation']['NegDW-BCE']:.17g}`.",
        "- 29 ilişki için her iki eşik kaynağının değerleri JSON çıktısında kayıtlı.",
        f"- Tam PR eğri noktaları: `{OUT / 'm1_m4_validation_pr_curves.csv'}` ({payload['pr_curve_rows']} veri satırı).", "",
        "## Sınırlar ve sonraki adım", "",
        "- Bu denetim M1/M4 test analizini tamamlamaz ve makale sonuçlarının değiştirilmesini desteklemez.",
        "- Test daha önce prototip çalışmalarda incelenmiştir; kilitli eşikler ve evaluator ile yapılacak skorlamayı keşifsel olarak işaretle, test sonucuna göre eşik değiştirme.",
        "- M4 için en az 5 (tercihen 10) eşlenmiş seed'in test tahminleri ve seed+cluster belirsizlik analizi hâlâ gereklidir.", "",
        "## Sabit 0.5: koşu bazlı ölçütler", "",
        "| Kayıp | Seed | Macro-F1 | Micro-F1 | Precision | Recall | Exact match | Macro-AP |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    fixed = payload["policies"]["fixed_0_5"]
    for arm in ARMS:
        for seed in SEEDS:
            m = fixed[arm]["by_seed"][str(seed)]
            lines.append(f"| {arm} | {seed} | {m['macro_f1_fixed_inventory']:.6f} | {m['micro_f1']:.6f} | "
                         f"{m['micro_precision']:.6f} | {m['micro_recall']:.6f} | {m['exact_set_match']:.6f} | "
                         f"{m['macro_average_precision_fixed_inventory']:.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    aggregate_bytes = AGGREGATE.read_bytes()
    aggregate = load_json(aggregate_bytes, str(AGGREGATE))
    train_bytes, validation_bytes = TRAIN.read_bytes(), VALIDATION.read_bytes()
    train_rows, validation_rows = load_jsonl(train_bytes, str(TRAIN)), load_jsonl(validation_bytes, str(VALIDATION))
    labels = json.loads(SCHEMA.read_text(encoding="utf-8"))["labels"]
    check(digest(train_bytes) == aggregate["train_input"]["sha256"] and len(train_rows) == aggregate["train_input"]["rows"],
          "Training input hash/row count mismatch")
    check(digest(validation_bytes) == aggregate["validation_input"]["sha256"], "Validation input hash mismatch")
    check(len(validation_rows) == aggregate["validation_input"]["rows"] == 720, "Validation row count mismatch")
    check(ordered_id_hash(validation_rows) == aggregate["validation_input"]["candidate_order_sha256"],
          "Validation candidate-order hash mismatch")
    check(labels == aggregate["labels_in_score_vector_order"], "Schema label order mismatch")
    for path, key in ((CONTEXT, "context_manifest_sha256"), (SCHEMA, "label_schema_sha256"),
                      (TOKENIZER, "tokenizer_sha256")):
        check(file_digest(path) == aggregate[key], f"Feature hash mismatch: {path}")

    code_files = {"train_encoder.py": Path("src/orgu/train_encoder.py"),
                  "neural.py": Path("src/orgu/neural.py"), "metrics.py": Path("src/orgu/metrics.py")}
    for name, path in code_files.items():
        check(file_digest(path) == aggregate["training_code_hashes"][name], f"Source hash mismatch: {path}")

    runner_path = Path("run_m1m4_validation.py")
    runner_hash_matches = runner_path.is_file() and file_digest(runner_path) == aggregate["runner_code_sha256"]
    check(runner_hash_matches, "Runner source-code hash mismatch or missing runner source")

    runs, run_reports = read_archive(aggregate, validation_rows, labels)
    analysis = analyze_thresholds(runs, labels)
    OUT.mkdir(parents=True, exist_ok=True)
    pr_rows = save_pr_curves(OUT / "m1_m4_validation_pr_curves.csv", runs, labels)
    payload = {
        "status": "passed",
        "scope": "validation only; test input not opened",
        "archive_sha256": file_digest(ARCHIVE),
        "aggregate_manifest_sha256": digest(aggregate_bytes),
        "analysis_script_sha256": file_digest(Path("scripts/audit_m1m4_validation_archive.py")),
        "train_input_sha256": digest(train_bytes),
        "validation_input_sha256": digest(validation_bytes),
        "validation_candidate_order_sha256": ordered_id_hash(validation_rows),
        "validation_rows": len(validation_rows),
        "run_count": len(run_reports),
        "test_file_read": False,
        "checkpoint_blobs_in_archive": False,
        "runner_source_rechecked": True,
        "runner_code_sha256": file_digest(runner_path),
        "runner_hash_matches": runner_hash_matches,
        "runs": run_reports,
        "threshold_protocol": {
            "objective": "mean fixed-inventory macro-F1 over seeds 13, 42, 87",
            "threshold_candidates": "0, every observed probability, nextafter(max,+infinity)",
            "decision": "score >= threshold",
            "tie_break": "largest threshold among exact ties",
            "relation_threshold_objective": "mean per-relation F1 over three seeds",
            "validation_tuning_is_in_sample": True,
            "theoretical_bce_threshold": THEORY_THRESHOLD,
            "fixed_inventory_macro_ap": "equal-score items grouped at each cutoff",
        },
        **analysis,
        "pr_curve_csv": str(OUT / "m1_m4_validation_pr_curves.csv"),
        "pr_curve_rows": pr_rows,
    }
    (OUT / "results/analysis/validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(OUT / "results/analysis/validation.md", payload)
    print(f"[audit] PASS: {len(run_reports)} runs, {len(validation_rows)} validation rows each")
    print(f"[audit] report: {OUT / 'results/analysis/validation.md'}")
    print(f"[audit] PR points: {pr_rows}; test file opened: no")


if __name__ == "__main__":
    main()