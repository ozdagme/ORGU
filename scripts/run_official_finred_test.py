#!/usr/bin/env python3
"""Run post-hoc inference on the official FinRED test with frozen checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import statistics
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SEEDS = (13, 42, 87, 101, 202, 303, 404, 505, 606, 707)
ARMS = ("BCE", "NegDW-BCE")
THRESHOLD_POLICIES = {
    "shared_0_5": {"BCE": 0.5, "NegDW-BCE": 0.5},
    "validation_macro_f1": {
        "BCE": 0.06187598779797554,
        "NegDW-BCE": 0.188323974609375,
    },
    "validation_recall_matched": {
        "BCE": 0.113973468542099,
        "NegDW-BCE": 0.5,
    },
}
EXPECTED_SOURCE_COUNTS = {"records": 1068, "triples": 1488, "candidates": 1277}
EXPECTED_SOURCE_HASHES = {
    "relations.txt": "bdfe1a6372d44bfdc8d5d0ff857fa4be6b611ce80b44e2b2b6a6dab88d6873ce",
    "test.txt": "820011b888e3fe08e42c0a6660fda3f3935fc20a6c956bef679901a96eec39e5",
    "test.sent": "08bd9c2e30dbf9ba951e260050e2fa6833e5d596d65745bc1cc8fec57bc35c19",
    "test.tup": "388a9e0ad38938d50f13a73a3acf9125e1b82285770c3d3830e41c54b0f82ef1",
    "test.pointer": "883b5c805def9733172d1b9564505716a1af022c761306a3223a64c49bf74e7f",
}


class EvaluationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [line.rstrip("\r\n") for line in handle]


def _normal_text(value: str) -> str:
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _relation_aliases(names: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for display in names:
        display = display.strip()
        if not display:
            continue
        underscored = "_".join(display.split())
        canonical = underscored.replace("/", "_or_")
        for spelling in {
            display,
            underscored,
            canonical,
            underscored.replace("/", "_"),
        }:
            key = spelling.casefold().replace("_/_", "/")
            if key in aliases and aliases[key] != canonical:
                raise EvaluationError(f"Ambiguous relation name in relations.txt: {spelling!r}")
            aliases[key] = canonical
    _require(bool(aliases), "relations.txt has no relation labels")
    return aliases


def _canonical_relation(value: str, aliases: dict[str, str]) -> str:
    key = value.strip().casefold().replace("_/_", "/")
    if key not in aliases:
        raise EvaluationError(f"Unknown relation in official test: {value!r}")
    return aliases[key]


def _check_txt_alignment(
    sentence: str, combined: str, tuple_line: str, aliases: dict[str, str], source_line: int
) -> None:
    normalized_sentence = _normal_text(sentence)
    normalized_combined = _normal_text(combined)
    _require(bool(normalized_sentence), f"Empty .sent line {source_line}")
    _require(normalized_combined.startswith(normalized_sentence),
             f".txt sentence does not match .sent on line {source_line}")
    remainder = normalized_combined[len(normalized_sentence):].strip()
    if remainder:
        _require(remainder.startswith("|"), f".txt/.sent boundary mismatch on line {source_line}")
        remainder = remainder[1:].strip()
    txt_parts = remainder.split("|") if remainder else []
    tup_parts = tuple_line.split("|") if tuple_line.strip() else []
    _require(len(txt_parts) == len(tup_parts), f".txt/.tup count mismatch on line {source_line}")
    for txt_part, tup_part in zip(txt_parts, tup_parts, strict=True):
        txt_fields = [field.strip() for field in txt_part.split(";")]
        tup_fields = [field.strip() for field in tup_part.split(";")]
        _require(
            len(txt_fields) == 3 and len(tup_fields) == 3
            and all(txt_fields) and all(tup_fields),
            f"Invalid .txt/.tup triple on line {source_line}",
        )
        _require(
            _canonical_relation(txt_fields[2], aliases)
            == _canonical_relation(tup_fields[2], aliases),
            f".txt/.tup relation mismatch on line {source_line}",
        )


def parse_official_test(
    raw_dir: Path,
    labels: list[str],
    relation_inventory_path: Path | None = None,
) -> dict[str, Any]:
    inventory_path = relation_inventory_path or raw_dir / "relations.txt"
    paths = {suffix: raw_dir / f"test.{suffix}" for suffix in ("txt", "sent", "tup", "pointer")}
    for path in (inventory_path, *paths.values()):
        _require(path.is_file(), f"Required official FinRED file is missing: {path}")

    source_hashes = {"relations.txt": sha256_file(inventory_path)}
    source_hashes.update({f"test.{suffix}": sha256_file(path) for suffix, path in paths.items()})
    for filename, expected_hash in EXPECTED_SOURCE_HASHES.items():
        actual_hash = source_hashes[filename]
        _require(
            actual_hash == expected_hash,
            f"{filename} SHA-256 differs from the audited source: expected {expected_hash}, got {actual_hash}",
        )

    aliases = _relation_aliases(_read_lines(inventory_path))
    canonical_inventory = {_canonical_relation(name, aliases) for name in _read_lines(inventory_path)}
    _require(canonical_inventory == set(labels), "Official relation inventory differs from the frozen model labels")
    source_lines = {suffix: _read_lines(path) for suffix, path in paths.items()}
    counts = {suffix: len(values) for suffix, values in source_lines.items()}
    _require(len(set(counts.values())) == 1, f"Official test files have different line counts: {counts}")

    candidates: list[dict[str, Any]] = []
    triple_count = 0
    label_to_index = {label: index for index, label in enumerate(labels)}
    for record_index, (combined, sentence, tuple_line, pointer_line) in enumerate(
        zip(source_lines["txt"], source_lines["sent"], source_lines["tup"],
            source_lines["pointer"], strict=True)
    ):
        source_line = record_index + 1
        _check_txt_alignment(sentence, combined, tuple_line, aliases, source_line)
        tokens = sentence.split()
        _require(bool(tokens), f"Empty token sequence on official test line {source_line}")
        tuple_parts = tuple_line.split("|") if tuple_line.strip() else []
        pointer_parts = pointer_line.split("|") if pointer_line.strip() else []
        _require(len(tuple_parts) == len(pointer_parts),
                 f".pointer/.tup count mismatch on line {source_line}")

        grouped: dict[tuple[int, int, int, int], set[str]] = {}
        for pointer_part, tuple_part in zip(pointer_parts, tuple_parts, strict=True):
            pointer_fields = pointer_part.strip().split(maxsplit=4)
            triple_fields = [field.strip() for field in tuple_part.split(";")]
            _require(
                len(pointer_fields) == 5 and len(triple_fields) == 3 and all(triple_fields),
                f"Invalid pointer/triple format on line {source_line}",
            )
            try:
                head_start, head_end_inclusive, tail_start, tail_end_inclusive = map(
                    int, pointer_fields[:4]
                )
            except ValueError as exc:
                raise EvaluationError(f"Noninteger pointer on line {source_line}") from exc
            relation = _canonical_relation(pointer_fields[4], aliases)
            _require(relation == _canonical_relation(triple_fields[2], aliases),
                     f"Pointer/.tup relation mismatch on line {source_line}")
            _require(relation in label_to_index, f"Unknown model relation {relation!r}")
            for start, end, source_name, role in (
                (head_start, head_end_inclusive, triple_fields[0], "head"),
                (tail_start, tail_end_inclusive, triple_fields[1], "tail"),
            ):
                _require(0 <= start <= end < len(tokens),
                         f"Out-of-range {role} pointer on line {source_line}: {start}:{end}")
                surface = " ".join(tokens[start:end + 1])
                _require(_normal_text(surface) == _normal_text(source_name),
                         f"Pointer/name mismatch on line {source_line}: {surface!r} != {source_name!r}")
            triple_count += 1
            grouped.setdefault(
                (head_start, head_end_inclusive + 1, tail_start, tail_end_inclusive + 1), set()
            ).add(relation)

        record_id = f"finred:test:{source_line:06d}"
        for (head_start, head_end, tail_start, tail_end), candidate_labels in grouped.items():
            _require(head_end <= tail_start or tail_end <= head_start,
                     f"Overlapping relation arguments on line {source_line}; cannot mark unambiguously")
            identifier_payload = json.dumps(
                [record_id, head_start, head_end, tail_start, tail_end],
                separators=(",", ":"),
            ).encode("utf-8")
            candidate_id = hashlib.sha256(identifier_payload).hexdigest()
            candidates.append({
                "id": candidate_id,
                "record_id": record_id,
                "record_index": record_index,
                "tokens": tokens,
                "head_start": head_start,
                "head_end": head_end,
                "tail_start": tail_start,
                "tail_end": tail_end,
                "labels": sorted(candidate_labels),
            })

    observed = {"records": counts["sent"], "triples": triple_count, "candidates": len(candidates)}
    _require(observed == EXPECTED_SOURCE_COUNTS,
             f"Official test counts differ from the audited source: {observed}")
    _require(len({row["id"] for row in candidates}) == len(candidates),
             "Duplicate candidate IDs after directed-pair grouping")
    return {
        "candidates": candidates,
        "source_hashes": source_hashes,
        "counts": observed,
        "source_bundle_sha256": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _marked_tokens(candidate: dict[str, Any]) -> list[str]:
    tokens = candidate["tokens"]
    head_start, head_end = candidate["head_start"], candidate["head_end"]
    tail_start, tail_end = candidate["tail_start"], candidate["tail_end"]
    boundaries: dict[int, list[tuple[int, str]]] = {}
    for start, end, opening, closing, opening_order, closing_order in (
        (head_start, head_end, "[SUBJ]", "[/SUBJ]", 11, 1),
        (tail_start, tail_end, "[OBJ]", "[/OBJ]", 12, 2),
    ):
        boundaries.setdefault(start, []).append((opening_order, opening))
        boundaries.setdefault(end, []).append((closing_order, closing))
    marked: list[str] = []
    for token_index in range(len(tokens) + 1):
        marked.extend(marker for _, marker in sorted(boundaries.get(token_index, [])))
        if token_index < len(tokens):
            marked.append(str(tokens[token_index]))
    return marked


def _encode_marked_tokens(
    marked_tokens: list[str], tokenizer: Any, max_length: int, gap_token_id: int
) -> tuple[list[int], dict[str, int], int]:
    marker_ids = {
        "head_start": tokenizer.token_to_id("[SUBJ]"),
        "head_end": tokenizer.token_to_id("[/SUBJ]"),
        "tail_start": tokenizer.token_to_id("[OBJ]"),
        "tail_end": tokenizer.token_to_id("[/OBJ]"),
    }
    _require(None not in marker_ids.values(), "Feature tokenizer lacks entity marker tokens")
    _require(len(set(marker_ids.values())) == 4, "Entity marker token IDs are not distinct")
    encoding = tokenizer.encode(marked_tokens, is_pretokenized=True)
    full_ids = [int(value) for value in encoding.ids]
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    _require(len(full_ids) >= 2 and full_ids[0] == cls_id and full_ids[-1] == sep_id,
             "Unexpected tokenizer special-token layout")
    body = full_ids[1:-1]
    full_length = len(full_ids)
    positions = {}
    for name, marker_id in marker_ids.items():
        found = [index for index, token_id in enumerate(body) if token_id == marker_id]
        _require(len(found) == 1, f"Expected one {name} marker after tokenization")
        positions[name] = found[0]
    _require(positions["head_start"] < positions["head_end"], "Invalid subject marker order")
    _require(positions["tail_start"] < positions["tail_end"], "Invalid object marker order")

    body_budget = max_length - 2
    segments: list[tuple[int, int]]
    use_gap = False
    if len(body) <= body_budget:
        segments = [(0, len(body))]
    else:
        head_start = positions["head_start"]
        head_end = positions["head_end"] + 1
        tail_start = positions["tail_start"]
        tail_end = positions["tail_end"] + 1
        required_start = min(head_start, tail_start)
        required_end = max(head_end, tail_end)
        if required_end - required_start <= body_budget:
            pair_length = required_end - required_start
            left_context = (body_budget - pair_length) // 2
            window_start = required_start - left_context
            window_start = max(0, min(window_start, len(body) - body_budget))
            segments = [(window_start, window_start + body_budget)]
        else:
            use_gap = True
            _require(body_budget >= 4, "C2 token budget is too small for dual-window context")
            mention_spans = sorted(
                ((head_start, head_end), (tail_start, tail_end)), key=lambda span: span[0]
            )
            context_budget = body_budget - 1 - sum(
                mention_end - mention_start
                for mention_start, mention_end in mention_spans
            )
            _require(context_budget >= 0,
                     "Entity marker spans exceed the dual-window token budget")
            first_context = context_budget // 2
            context_allocations = (first_context, context_budget - first_context)
            segments = []
            for (mention_start, mention_end), context in zip(
                mention_spans, context_allocations, strict=True
            ):
                mention_length = mention_end - mention_start
                window_length = mention_length + context
                left_context = context // 2
                window_start = mention_start - left_context
                window_start = max(0, min(window_start, len(body) - window_length))
                window_end = window_start + window_length
                if mention_start < window_start or mention_end > window_end:
                    window_start = max(0, min(mention_start, len(body) - window_length))
                    window_end = window_start + window_length
                _require(mention_start >= window_start and mention_end <= window_end,
                         "Unable to preserve entity markers in dual-window context")
                segments.append((window_start, window_end))

    if use_gap:
        selected_body = body[segments[0][0]:segments[0][1]] + [gap_token_id]
        selected_body += body[segments[1][0]:segments[1][1]]
    else:
        selected_body = body[segments[0][0]:segments[0][1]]
    input_ids = [full_ids[0], *selected_body, full_ids[-1]]
    _require(len(input_ids) <= max_length,
             f"C2 crop exceeds max_length: {len(input_ids)} > {max_length}")
    output_positions = {}
    for name, marker_id in marker_ids.items():
        found = [index for index, token_id in enumerate(input_ids) if token_id == marker_id]
        _require(len(found) == 1, f"C2 crop lost or duplicated {name} marker")
        output_positions[name] = found[0]
    return input_ids, output_positions, full_length


def validate_training_preprocessing(
    artifacts: dict[str, Any], max_length: int, gap_token_id: int
) -> dict[str, int]:
    tokenizer = artifacts["tokenizer"]
    rows = artifacts["train_rows"]
    _require(bool(rows), "No frozen C2 training rows available for preprocessing validation")
    checked = long_rows = dual_rows = 0
    mismatches = []
    for row in rows:
        marked = row.get("marked_tokens")
        _require(isinstance(marked, list) and marked,
                 f"Training candidate lacks marked_tokens: {row.get('id')}")
        rebuilt_ids, rebuilt_positions, full_length = _encode_marked_tokens(
            marked, tokenizer, max_length, gap_token_id
        )
        expected_ids = [int(value) for value in row.get("model_inputs", row)["input_ids"]]
        expected_positions = row.get("input_marker_positions", {})
        checked += 1
        long_rows += int(full_length > max_length)
        dual_rows += int(len(row.get("source_segments", [])) > 1)
        if rebuilt_ids != expected_ids or any(
            int(rebuilt_positions.get(name, -1)) != int(expected_positions.get(name, -2))
            for name in ("head_start", "head_end", "tail_start", "tail_end")
        ):
            mismatches.append(str(row.get("id")))
            if len(mismatches) >= 5:
                break
    _require(not mismatches,
             "Reconstructed C2 preprocessing differs from frozen training features; "
             f"first mismatches: {mismatches}. Refusing official-test inference.")
    return {
        "training_candidates_checked": checked,
        "long_training_candidates": long_rows,
        "dual_window_training_candidates": dual_rows,
    }


def encode_candidates(
    parsed: dict[str, Any], tokenizer: Any, labels: list[str], max_length: int,
    gap_token_id: int,
) -> tuple[list[dict[str, Any]], list[int], int]:
    marker_ids = {
        "head_start": tokenizer.token_to_id("[SUBJ]"),
        "head_end": tokenizer.token_to_id("[/SUBJ]"),
        "tail_start": tokenizer.token_to_id("[OBJ]"),
        "tail_end": tokenizer.token_to_id("[/OBJ]"),
    }
    _require(None not in marker_ids.values(), "Feature tokenizer lacks entity marker tokens")
    _require(len(set(marker_ids.values())) == 4, "Entity marker token IDs are not distinct")
    label_set = set(labels)
    rows = []
    lengths = []
    full_length_overflow = 0
    for candidate in parsed["candidates"]:
        ids, positions, full_length = _encode_marked_tokens(
            _marked_tokens(candidate), tokenizer, max_length, gap_token_id
        )
        full_length_overflow += int(full_length > max_length)
        candidate_labels = candidate["labels"]
        _require(bool(candidate_labels) and set(candidate_labels) <= label_set,
                 f"Invalid gold labels for candidate {candidate['id']}")
        rows.append({
            "id": candidate["id"],
            "record_id": candidate["record_id"],
            "record_index": candidate["record_index"],
            "labels": candidate_labels,
            "model_inputs": {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "token_type_ids": [0] * len(ids),
            },
            "input_marker_positions": positions,
        })
        lengths.append(len(ids))
    return rows, lengths, full_length_overflow


def build_gold_matrix(rows: list[dict[str, Any]], labels: list[str], np: Any):
    label_to_index = {label: index for index, label in enumerate(labels)}
    gold = np.zeros((len(rows), len(labels)), dtype=np.uint8)
    for row_index, row in enumerate(rows):
        for label in row["labels"]:
            gold[row_index, label_to_index[label]] = 1
    return gold


def _average_precision(gold_column: Any, score_column: Any, np: Any) -> float:
    support = int(gold_column.sum())
    if support == 0:
        return 0.0
    order = np.argsort(-score_column, kind="mergesort")
    ranked_scores = score_column[order]
    cumulative_positives = np.cumsum(gold_column[order], dtype=np.int64)
    ends = np.concatenate((np.flatnonzero(np.diff(ranked_scores)), [len(ranked_scores) - 1]))
    true_positives = cumulative_positives[ends]
    predicted_positives = ends + 1
    precision = true_positives / predicted_positives
    recall = true_positives / support
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def _calculate_metrics(gold: Any, scores: Any, threshold: float, labels: list[str], np: Any) -> dict[str, Any]:
    predicted = scores >= threshold
    supports = gold.sum(axis=0).astype(int)
    per_label: dict[str, Any] = {}
    f1_values = []
    supported_f1_values = []
    tp_total = fp_total = fn_total = 0
    for index, label in enumerate(labels):
        gold_column = gold[:, index].astype(bool)
        predicted_column = predicted[:, index]
        tp = int(np.logical_and(gold_column, predicted_column).sum())
        fp = int(np.logical_and(~gold_column, predicted_column).sum())
        fn = int(np.logical_and(gold_column, ~predicted_column).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": int(supports[index]),
            "predicted": tp + fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
        if supports[index] >= 5:
            supported_f1_values.append(f1)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    micro_precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )
    return {
        "threshold": threshold,
        "candidate_count": int(gold.shape[0]),
        "accepted_labels": int(predicted.sum()),
        "macro_f1_fixed_inventory": float(statistics.mean(f1_values)),
        "macro_f1_support_at_least_5": float(statistics.mean(supported_f1_values)),
        "relations_with_support_at_least_5": len(supported_f1_values),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "exact_set_match": float(np.all(gold.astype(bool) == predicted, axis=1).mean()),
        "per_label": per_label,
    }


def _run_summary(
    gold: Any, all_scores: Any, runs: list[dict[str, Any]], labels: list[str], np: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    for run_index, run in enumerate(runs):
        scores = all_scores[run_index]
        per_label_ap = {
            label: _average_precision(gold[:, label_index], scores[:, label_index], np)
            for label_index, label in enumerate(labels)
        }
        results[run["name"]] = {
            "arm": "BCE" if run["loss_arm"] == "L0" else "NegDW-BCE",
            "seed": run["seed"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "run_manifest_sha256": run["manifest_sha256"],
            "threshold_independent": {
                "macro_average_precision_fixed_inventory": float(statistics.mean(per_label_ap.values())),
                "per_label_average_precision": per_label_ap,
            },
            "operating_points": {},
        }
        arm = results[run["name"]]["arm"]
        for policy_name, thresholds in THRESHOLD_POLICIES.items():
            results[run["name"]]["operating_points"][policy_name] = _calculate_metrics(
                gold, scores, thresholds[arm], labels, np
            )

    paired: dict[str, Any] = {}
    metric_names = (
        "macro_f1_fixed_inventory", "macro_f1_support_at_least_5",
        "micro_precision", "micro_recall", "micro_f1", "exact_set_match",
    )
    for policy_name in THRESHOLD_POLICIES:
        paired[policy_name] = {}
        for metric in metric_names:
            by_arm = {arm: [] for arm in ARMS}
            by_seed = {}
            for seed in SEEDS:
                pair = {}
                for arm in ARMS:
                    run_name = f"{arm}_seed{seed}"
                    value = float(results[run_name]["operating_points"][policy_name][metric])
                    pair[arm] = value
                    by_arm[arm].append(value)
                by_seed[str(seed)] = pair["NegDW-BCE"] - pair["BCE"]
            paired[policy_name][metric] = {
                "BCE_mean": float(statistics.mean(by_arm["BCE"])),
                "NegDW_BCE_mean": float(statistics.mean(by_arm["NegDW-BCE"])),
                "paired_difference_mean_NegDW_minus_BCE": float(statistics.mean(by_seed.values())),
                "paired_differences_by_seed": by_seed,
            }
    ap_differences = {}
    for seed in SEEDS:
        bce = results[f"BCE_seed{seed}"]["threshold_independent"]["macro_average_precision_fixed_inventory"]
        negdw = results[f"NegDW-BCE_seed{seed}"]["threshold_independent"]["macro_average_precision_fixed_inventory"]
        ap_differences[str(seed)] = negdw - bce
    paired["threshold_independent_macro_ap"] = {
        "BCE_mean": float(statistics.mean(
            results[f"BCE_seed{seed}"]["threshold_independent"]["macro_average_precision_fixed_inventory"]
            for seed in SEEDS
        )),
        "NegDW_BCE_mean": float(statistics.mean(
            results[f"NegDW-BCE_seed{seed}"]["threshold_independent"]["macro_average_precision_fixed_inventory"]
            for seed in SEEDS
        )),
        "paired_difference_mean_NegDW_minus_BCE": float(statistics.mean(ap_differences.values())),
        "paired_differences_by_seed": ap_differences,
    }
    return results, paired


def _write_result_zip(
    output: Path,
    overwrite: bool,
    summary: dict[str, Any],
    arrays: dict[str, Any],
    np: Any,
) -> None:
    _require(not output.exists() or overwrite, f"Output already exists: {output} (use --overwrite)")
    output.parent.mkdir(parents=True, exist_ok=True)
    npz_buffer = io.BytesIO()
    np.savez_compressed(npz_buffer, **arrays)
    readme = (
        "Official FinRED test evaluation result bundle\n"
        "Our team had not used the official test before this study; this is its first evaluation here. The analysis remains exploratory because seven seeds followed earlier results on the previously inspected entity-disjoint test.\n"
        "No training, checkpoint selection, or threshold selection was performed here.\n"
        "scores.npz contains raw probabilities, binary gold labels, labels, paired run IDs,\n"
        "candidate IDs and source-record indices. It contains no sentence text or entity names.\n"
        "Use this ZIP with the documented analysis scripts; do not distribute checkpoint weights.\n"
    )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            archive.writestr("scores.npz", npz_buffer.getvalue())
            archive.writestr("README.txt", readme)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the 20 frozen BCE/NegDW checkpoints on official FinRED test data, then create a small shareable ZIP."
    )
    parser.add_argument("--finred-dir", default="data/raw/finred_official",
                        help="Repository-relative directory containing the official test.* files (the official data directory).")
    parser.add_argument("--relations-file", default="data/raw/finred_official/relations.txt",
                        help="Repository-relative FinRED relation inventory path (default: data/raw/finred_official/relations.txt).")
    parser.add_argument("--feature-dir", default="data/features/finred_entity_disjoint_v1",
                        help="Repository-relative frozen C2 feature directory matching the run manifests.")
    parser.add_argument("--run-root", default="runs/m1_m4_loss_family_v1",
                        help="Repository-relative directory containing the ten-seed loss-family runs.")
    parser.add_argument("--output", default="results/official_finred_test_results.zip",
                        help="Repository-relative result ZIP path.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"), default="auto")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate data, tokenizer, run manifests, and checkpoint hashes without model inference.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing result ZIP.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        os.chdir(ROOT)
        source_root = str(ROOT / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        try:
            import numpy as np
            import torch
            from safetensors.torch import load_file
            from orgu import neural, train_encoder
            from orgu.export_m1m4_test_predictions import (
                _preflight_runs,
                _resolve_device,
                _score_rows,
            )
        except ImportError as exc:
            raise EvaluationError(
                "ORGU inference dependencies are unavailable. Run this in the existing ORGU environment with torch, transformers, tokenizers, and safetensors installed."
            ) from exc

        raw_dir = train_encoder.project_path(args.finred_dir, "FinRED data directory")
        relation_inventory_path = train_encoder.project_path(
            args.relations_file, "FinRED relation inventory"
        )
        feature_dir = train_encoder.project_path(args.feature_dir, "Feature directory")
        run_root = train_encoder.project_path(args.run_root, "Run root")
        output_path = train_encoder.project_path(args.output, "Output ZIP")
        artifacts = train_encoder.load_feature_data(feature_dir)
        _require(artifacts["labels"] and len(artifacts["labels"]) == 29,
                 "Expected the frozen 29-label FinRED model inventory")
        _require(artifacts["manifest"].get("model_id") == "google-bert/bert-base-cased",
                 "Unexpected encoder in the frozen feature manifest")
        max_length = int(artifacts["manifest"].get("max_length", 0))
        _require(max_length == 512, f"Unexpected C2 max_length: {max_length}")
        gap_token = artifacts["manifest"].get("gap_token")
        gap_token_id = artifacts["manifest"].get("gap_token_id")
        _require(gap_token == "[unused1]" and isinstance(gap_token_id, int),
                 "Unexpected C2 gap-token configuration")
        _require(artifacts["tokenizer"].token_to_id(gap_token) == gap_token_id,
                 "C2 gap-token ID differs from the frozen tokenizer")
        preprocessing_check = validate_training_preprocessing(
            artifacts, max_length, gap_token_id
        )
        expected_dual_windows = int(
            artifacts["manifest"]["arms"]["C2"]["files"]["train"]["dual_window_candidates"]
        )
        _require(
            preprocessing_check["dual_window_training_candidates"] == expected_dual_windows,
            "Observed C2 dual-window count differs from the frozen feature manifest",
        )
        parsed = parse_official_test(raw_dir, artifacts["labels"], relation_inventory_path)
        test_rows, sequence_lengths, overflow_count = encode_candidates(
            parsed, artifacts["tokenizer"], artifacts["labels"], max_length, gap_token_id
        )
        order_hash = train_encoder.ordered_ids_sha256(test_rows)
        test_hash = parsed["source_bundle_sha256"]
        runs = _preflight_runs(run_root, artifacts, SEEDS)

        print(
            f"[preflight] official test: {parsed['counts']['records']} records, "
            f"{parsed['counts']['triples']} triples, {len(test_rows)} directed-pair candidates; "
            f"max prepared length {max(sequence_lengths)}/{max_length}; "
            f"{overflow_count} full sequences use C2 context windows",
            flush=True,
        )
        print(
            f"[preflight] C2 preprocessing reproduced for "
            f"{preprocessing_check['training_candidates_checked']} frozen training candidates "
            f"({preprocessing_check['long_training_candidates']} long, "
            f"{preprocessing_check['dual_window_training_candidates']} dual-window)",
            flush=True,
        )
        print(f"[preflight] source bundle SHA-256: {test_hash}", flush=True)
        print(f"[preflight] candidate-order SHA-256: {order_hash}", flush=True)
        print(f"[preflight] validated {len(runs)} run manifests and checkpoint hashes", flush=True)
        if args.check_only:
            print("[preflight] --check-only: no checkpoints loaded and no scores produced", flush=True)
            return 0

        device = _resolve_device(args.device)
        gold = build_gold_matrix(test_rows, artifacts["labels"], np)
        all_scores = np.empty((len(runs), len(test_rows), len(artifacts["labels"])), dtype=np.float32)
        run_records = []
        for run_index, run in enumerate(runs):
            manifest = run["manifest"]
            model = neural.EntityMarkerEncoder(
                manifest["pretrained_model"],
                artifacts["tokenizer"].get_vocab_size(with_added_tokens=True),
                len(artifacts["labels"]),
                revision=manifest["pretrained_model_revision"],
            ).to(device)
            state = load_file(str(run["checkpoint_path"]), device="cpu")
            model.load_state_dict(state, strict=True)
            predictions = _score_rows(
                model, test_rows, artifacts, device, run["eval_batch_size"], test_hash
            )
            all_scores[run_index] = np.asarray([row["scores"] for row in predictions], dtype=np.float32)
            run_records.append(run)
            del state
            del model
            del predictions
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[inference] {run['name']} complete ({len(test_rows)} candidates)", flush=True)

        per_run, paired = _run_summary(gold, all_scores, run_records, artifacts["labels"], np)
        gold_support = {
            label: int(gold[:, index].sum()) for index, label in enumerate(artifacts["labels"])
        }
        summary = {
            "format": "orgu-official-finred-test-evaluation-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "interpretation": "first-time official-test evaluation in this study; exploratory because ten-seed expansion followed entity-disjoint test results",
            "score_interpretation": "agreement with recorded annotations, not independently verified relation truth",
            "test_previously_inspected": False,
            "test_used_for_training_checkpoint_or_threshold_selection": False,
            "prior_team_use_author_confirmation_date": "2026-10-30",
            "operation": "inference with 20 existing selected checkpoints; no training",
            "selection_protocol": "checkpoints were selected on validation micro-F1 at threshold 0.5; the three threshold policies below were fixed before this official-test inference",
            "evaluator_code_sha256": sha256_file(Path(__file__)),
            "device": str(device),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "dataset": {
                "name": "FinRED official test",
                **parsed["counts"],
                "candidate_definition": "one candidate per sentence-local directed head/tail span pair; observed relations for a pair are merged into one multilabel target",
                "pointer_convention": "zero-based inclusive token offsets",
                "source_hashes": parsed["source_hashes"],
                "source_bundle_sha256": test_hash,
                "candidate_order_sha256": order_hash,
                "sequence_length": {
                    "maximum_encoded_subtokens": max(sequence_lengths),
                    "model_max_length": max_length,
                    "full_sequences_using_context_windows": overflow_count,
                },
                "preprocessing_reproduction_check": preprocessing_check,
                "positive_candidate_support_by_label": gold_support,
            },
            "feature_assets": {
                "feature_manifest_sha256": train_encoder.sha256_file(artifacts["manifest_path"]),
                "tokenizer_sha256": train_encoder.sha256_file(artifacts["tokenizer_path"]),
                "label_schema_sha256": train_encoder.sha256_file(artifacts["schema_path"]),
                "labels_in_score_order": artifacts["labels"],
            },
            "checkpoint_runs": {
                run["name"]: {
                    "seed": run["seed"],
                    "arm": "BCE" if run["loss_arm"] == "L0" else "NegDW-BCE",
                    "run_manifest_sha256": run["manifest_sha256"],
                    "checkpoint_sha256": run["checkpoint_sha256"],
                }
                for run in run_records
            },
            "threshold_policies": THRESHOLD_POLICIES,
            "per_run_results": per_run,
            "paired_seed_summary": paired,
            "archive_contents": {
                "scores.npz": "raw sigmoid scores [run, candidate, label], binary gold matrix, labels, run IDs, candidate IDs and source-record indices; no raw text/entity strings",
                "summary.json": "source/checkpoint hashes, all metrics and paired seed summaries",
            },
        }
        arrays = {
            "scores": all_scores,
            "gold": gold,
            "labels": np.asarray(artifacts["labels"], dtype="U"),
            "run_names": np.asarray([run["name"] for run in run_records], dtype="U"),
            "seeds": np.asarray([run["seed"] for run in run_records], dtype=np.int64),
            "arms": np.asarray(["BCE" if run["loss_arm"] == "L0" else "NegDW-BCE" for run in run_records], dtype="U"),
            "candidate_ids": np.asarray([row["id"] for row in test_rows], dtype="U"),
            "record_indices": np.asarray([row["record_index"] for row in test_rows], dtype=np.int16),
        }
        _write_result_zip(output_path, args.overwrite, summary, arrays, np)
        print(f"[complete] results: {output_path}", flush=True)
        print(f"[complete] upload only this ZIP; checkpoint archive is not needed", flush=True)
        return 0
    except Exception as exc:
        print(f"[official-test-eval] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())