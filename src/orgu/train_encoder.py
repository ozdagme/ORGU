from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import tokenizers
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from . import neural
from .metrics import multilabel_metrics


DEFAULT_ENCODER = "google-bert/bert-base-cased"
DEFAULT_ENCODER_REVISION = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
UNKNOWN_LABEL_WEIGHT = 0.1
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
ASL_GAMMA_POS = 0.0
ASL_GAMMA_NEG = 4.0
ASL_CLIP = 0.05
ASL_EPSILON = 1e-8
LOSS_METHOD_NAMES = {
    "L0": "BCE",
    "L1": "NegDW-BCE",
    "FOCAL": "Focal",
    "ASL": "ASL",
    "POS_WEIGHT": "pos_weight-BCE",
}
EXPECTED_COUNTS = {
    "finred_pair_disjoint_v1": (5752, 665),
    "finred_entity_disjoint_v1": (5647, 720),
    "finred_context_textfiltered_v1": (7068, 1318),
}
MARKER_NAMES = {
    "head_start": "[SUBJ]",
    "head_end": "[/SUBJ]",
    "tail_start": "[OBJ]",
    "tail_end": "[/OBJ]",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: str | Path, description: str) -> Path:
    path = Path(value)
    require(not path.is_absolute(), f"{description} must be repository-relative: {path}")
    require(".." not in path.parts, f"{description} cannot contain '..': {path}")
    resolved = path.resolve()
    repository = Path.cwd().resolve()
    require(resolved.is_relative_to(repository), f"{description} escapes the repository: {path}")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def ordered_ids_sha256(rows: list[dict]) -> str:
    payload = "\n".join(str(row.get("id", "")) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def label_inventory(path: Path) -> list[str]:
    schema = read_json(path)
    labels = schema.get("labels")
    require(isinstance(labels, list) and labels, "Label schema must contain a nonempty labels list")
    if isinstance(labels[0], dict):
        labels = [item.get("name", item.get("label")) for item in labels]
    labels = [str(label) for label in labels]
    require(len(labels) == len(set(labels)), "Label schema contains duplicate labels")
    return labels


def load_feature_data(feature_dir: Path):
    feature_dir = project_path(feature_dir, "Feature directory")
    manifest_path = feature_dir / "context_manifest.json"
    tokenizer_path = feature_dir / "tokenizer.json"
    schema_path = feature_dir / "label_schema.json"
    for path in (manifest_path, tokenizer_path, schema_path):
        require(path.is_file(), f"Missing required feature asset: {path}")

    manifest = read_json(manifest_path)
    require(manifest.get("test_inference_run") is False, "Feature manifest does not preserve the test lock")
    arm_files = manifest.get("arms", {}).get("C2", {}).get("files", {})
    require("train" in arm_files and "validation" in arm_files, "C2 train/validation entries are missing")

    split_rows = {}
    split_paths = {}
    for split in ("train", "validation"):
        info = arm_files[split]
        path = project_path(info.get("path", ""), f"{split} feature file")
        require(path.is_file(), f"Missing {split} feature file: {path}")
        actual_hash = sha256_file(path)
        require(actual_hash == info.get("sha256"), f"{split} feature hash mismatch: {path}")
        split_rows[split] = read_jsonl(path)
        split_paths[split] = path
        require(split_rows[split], f"{split} feature file is empty")

    train_rows = [row for row in split_rows["train"] if row.get("eligible_for_training", True)]
    validation_rows = split_rows["validation"]
    if feature_dir.name in EXPECTED_COUNTS:
        expected_train, expected_validation = EXPECTED_COUNTS[feature_dir.name]
        require(len(train_rows) == expected_train,
                f"Unexpected eligible train count: {len(train_rows)} != {expected_train}")
        require(len(validation_rows) == expected_validation,
                f"Unexpected validation count: {len(validation_rows)} != {expected_validation}")
    labels = label_inventory(schema_path)
    label_set = set(labels)
    for split_name, rows in (("train", train_rows), ("validation", validation_rows)):
        for row in rows:
            row_labels = row.get("labels")
            require(isinstance(row_labels, list), f"{split_name} row lacks a labels list")
            unknown = set(row_labels) - label_set
            require(not unknown, f"Unknown labels in {split_name}: {sorted(unknown)}")
            require(isinstance(row.get("model_inputs", row).get("input_ids"), list),
                    f"{split_name} row lacks encoded input_ids")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    marker_ids = {name: tokenizer.token_to_id(token) for name, token in MARKER_NAMES.items()}
    require(None not in marker_ids.values(), "Feature tokenizer is missing entity markers")
    require(len(set(marker_ids.values())) == len(MARKER_NAMES), "Feature entity marker IDs are not distinct")
    require(tokenizer.token_to_id("[PAD]") is not None, "Feature tokenizer is missing [PAD]")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "tokenizer_path": tokenizer_path,
        "schema_path": schema_path,
        "train_path": split_paths["train"],
        "validation_path": split_paths["validation"],
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "labels": labels,
        "tokenizer": tokenizer,
        "marker_ids": marker_ids,
    }


def resolve_model_revision(
    encoder_name: str,
    feature_manifest: dict,
    requested_revision: str | None = None,
) -> str | None:
    if requested_revision is not None:
        return requested_revision
    if encoder_name == feature_manifest.get("model_id"):
        return feature_manifest.get("model_revision") or DEFAULT_ENCODER_REVISION
    return None


def validate_tokenizer_compatibility(
    tokenizer: Tokenizer,
    encoder_name: str,
    revision: str | None = None,
) -> None:
    local_only = Path(encoder_name).exists()
    base_tokenizer = AutoTokenizer.from_pretrained(
        encoder_name,
        revision=revision,
        use_fast=True,
        local_files_only=local_only,
        trust_remote_code=False,
    )
    base_vocab = base_tokenizer.get_vocab()
    feature_vocab = tokenizer.get_vocab(with_added_tokens=True)
    missing = set(base_vocab) - set(feature_vocab)
    mismatched = [
        token
        for token, token_id in base_vocab.items()
        if feature_vocab.get(token) != token_id
    ]
    require(not missing and not mismatched,
            "Feature tokenizer vocabulary does not match the selected pretrained encoder; "
            f"missing={len(missing)}, mismatched={len(mismatched)}")
    for token in MARKER_NAMES.values():
        require(token in feature_vocab, f"Feature tokenizer lacks marker token {token}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def target_matrix(rows: list[dict], labels: list[str], device: torch.device) -> torch.Tensor:
    label_to_id = {label: index for index, label in enumerate(labels)}
    targets = torch.zeros((len(rows), len(labels)), dtype=torch.float32, device=device)
    for row_index, row in enumerate(rows):
        for label in row["labels"]:
            targets[row_index, label_to_id[label]] = 1.0
    return targets


@torch.inference_mode()
def evaluate(
    model,
    rows: list[dict],
    labels: list[str],
    tokenizer,
    marker_ids: dict[str, int],
    device: torch.device,
    batch_size: int,
    common_mask=None,
    split_sha256: str | None = None,
) -> tuple[dict, list[dict]]:
    model.eval()
    predictions = []
    gold_sets = []
    predicted_sets = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs, positions = neural.make_batch(batch, tokenizer, marker_ids, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs, positions)
        probabilities = torch.sigmoid(logits.float()).cpu().tolist()
        for row, scores in zip(batch, probabilities):
            gold = set(row.get("labels", []))
            predicted = {
                labels[index] for index, score in enumerate(scores) if float(score) >= 0.5
            }
            gold_sets.append(gold)
            predicted_sets.append(predicted)
            predictions.append(
                {
                    "candidate_index": len(predictions),
                    "candidate_id": row.get("id"),
                    "record_id": row.get("record_id"),
                    "split": row.get("split", "validation"),
                    "split_sha256": split_sha256,
                    "head": row.get("head"),
                    "tail": row.get("tail"),
                    "gold_labels": sorted(gold),
                    "gold_targets": [int(label in gold) for label in labels],
                    "predicted_labels": sorted(predicted),
                    "scores": [float(score) for score in scores],
                    "threshold": 0.5,
                }
            )
    report = {"all_candidates": multilabel_metrics(gold_sets, predicted_sets, labels)}
    if common_mask is not None:
        mask = np.asarray(common_mask, dtype=bool)
        require(mask.shape == (len(rows),), "Common-subset mask has the wrong size")
        report["common_candidates"] = multilabel_metrics(
            [gold for gold, include in zip(gold_sets, mask) if include],
            [predicted for predicted, include in zip(predicted_sets, mask) if include],
            labels,
        )
    report["per_class"] = report["all_candidates"]["per_class"]
    return report, predictions


def compute_pos_weight(targets: torch.Tensor) -> torch.Tensor:
    require(targets.ndim == 2 and targets.shape[0] > 0, "Training targets must be a nonempty matrix")
    target_values = targets.float()
    require(
        bool(torch.all((target_values == 0.0) | (target_values == 1.0)).item()),
        "Training targets must be binary",
    )
    positives = target_values.sum(dim=0)
    require(bool(torch.all(positives > 0).item()), "pos_weight requires at least one training positive per relation")
    negatives = target_values.shape[0] - positives
    return negatives / positives


def loss_parameters(args: argparse.Namespace, pos_weight: torch.Tensor | None) -> dict:
    return {
        "negative_target_weight": UNKNOWN_LABEL_WEIGHT if args.loss == "L1" else None,
        "focal": {
            "alpha": args.focal_alpha,
            "gamma": args.focal_gamma,
        } if args.loss == "FOCAL" else None,
        "asl": {
            "gamma_positive": args.asl_gamma_positive,
            "gamma_negative": args.asl_gamma_negative,
            "negative_clip": args.asl_clip,
            "epsilon": ASL_EPSILON,
            "focusing_weight_gradients": "disabled",
        } if args.loss == "ASL" else None,
        "pos_weight": {
            "formula": "training_negative_count / training_positive_count per relation",
            "values_by_label": {
                label: float(value)
                for label, value in zip(args.labels, pos_weight.detach().cpu().tolist())
            },
        } if pos_weight is not None else None,
    }


def batch_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_arm: str,
    pos_weight: torch.Tensor | None = None,
    focal_alpha: float = FOCAL_ALPHA,
    focal_gamma: float = FOCAL_GAMMA,
    asl_gamma_positive: float = ASL_GAMMA_POS,
    asl_gamma_negative: float = ASL_GAMMA_NEG,
    asl_clip: float = ASL_CLIP,
) -> torch.Tensor:
    if loss_arm in {"L0", "L1"}:
        losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if loss_arm == "L1":
            weights = torch.where(
                targets > 0,
                torch.ones_like(targets),
                torch.full_like(targets, UNKNOWN_LABEL_WEIGHT),
            )
            return (losses * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return losses.mean(dim=1)

    logits_float = logits.float()
    targets_float = targets.float()
    probabilities = torch.sigmoid(logits_float)
    if loss_arm == "FOCAL":
        require(0.0 <= focal_alpha <= 1.0 and focal_gamma >= 0.0,
                "Focal alpha must be in [0, 1] and gamma must be nonnegative")
        bce = F.binary_cross_entropy_with_logits(logits_float, targets_float, reduction="none")
        probability_of_target = (
            probabilities * targets_float
            + (1.0 - probabilities) * (1.0 - targets_float)
        )
        alpha_of_target = (
            focal_alpha * targets_float
            + (1.0 - focal_alpha) * (1.0 - targets_float)
        )
        return (alpha_of_target * (1.0 - probability_of_target).pow(focal_gamma) * bce).mean(dim=1)

    if loss_arm == "ASL":
        require(asl_gamma_positive >= 0.0 and asl_gamma_negative >= 0.0,
                "ASL gammas must be nonnegative")
        require(0.0 <= asl_clip < 1.0, "ASL negative clip must be in [0, 1)")
        negative_probabilities = (1.0 - probabilities + asl_clip).clamp(max=1.0)
        epsilon = ASL_EPSILON
        log_likelihood = (
            targets_float * probabilities.clamp_min(epsilon).log()
            + (1.0 - targets_float) * negative_probabilities.clamp_min(epsilon).log()
        )
        with torch.no_grad():
            probability_of_target = (
                probabilities * targets_float
                + negative_probabilities * (1.0 - targets_float)
            )
            gamma = (
                asl_gamma_positive * targets_float
                + asl_gamma_negative * (1.0 - targets_float)
            )
            focusing_weight = (1.0 - probability_of_target).pow(gamma)
        return (-focusing_weight * log_likelihood).mean(dim=1)

    if loss_arm == "POS_WEIGHT":
        require(pos_weight is not None, "pos_weight vector must be computed from training labels")
        require(pos_weight.numel() == logits.shape[-1], "pos_weight vector does not match relation inventory")
        weighted_bce = F.binary_cross_entropy_with_logits(
            logits_float,
            targets_float,
            pos_weight=pos_weight.to(device=logits.device, dtype=logits_float.dtype),
            reduction="none",
        )
        return weighted_bce.mean(dim=1)

    raise RuntimeError(f"Unsupported loss arm: {loss_arm}")


def save_checkpoint(path: Path, model: torch.nn.Module) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(state, str(path), metadata={"format": "pt"})
    return sha256_file(path)


def train(args: argparse.Namespace) -> dict:
    require(args.arm == "C2", "Only the C2 feature arm is implemented")
    require(args.loss in LOSS_METHOD_NAMES, f"Unsupported loss arm: {args.loss}")
    require(args.epochs > 0 and args.batch_size > 0 and args.accumulation > 0,
            "Training budget values must be positive")
    require(args.eval_batch_size > 0, "Evaluation batch size must be positive")
    require(0.0 <= args.focal_alpha <= 1.0 and args.focal_gamma >= 0.0,
            "Focal alpha must be in [0, 1] and gamma must be nonnegative")
    require(args.asl_gamma_positive >= 0.0 and args.asl_gamma_negative >= 0.0,
            "ASL gammas must be nonnegative")
    require(0.0 <= args.asl_clip < 1.0, "ASL negative clip must be in [0, 1)")
    run_started = time.monotonic()
    artifacts = load_feature_data(args.feature_dir)
    train_rows = artifacts["train_rows"]
    validation_rows = artifacts["validation_rows"]
    labels = artifacts["labels"]
    tokenizer = artifacts["tokenizer"]
    marker_ids = artifacts["marker_ids"]
    model_revision = resolve_model_revision(
        args.pretrained_model,
        artifacts["manifest"],
        args.pretrained_model_revision,
    )
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda":
            require(torch.cuda.is_available(), "CUDA was selected but is not available in this PyTorch runtime")

    pos_weight = None
    if args.loss == "POS_WEIGHT":
        training_targets = target_matrix(train_rows, labels, torch.device("cpu"))
        pos_weight = compute_pos_weight(training_targets).to(device)
    args.labels = labels
    configured_loss_parameters = loss_parameters(args, pos_weight)

    if args.check_only:
        print(json.dumps({
            "status": "INPUTS_VALID",
            "feature_dir": str(args.feature_dir),
            "training_candidates": len(train_rows),
            "validation_candidates": len(validation_rows),
            "relation_labels": len(labels),
            "pretrained_model": args.pretrained_model,
            "pretrained_model_revision": model_revision,
            "pretrained_model_loaded": False,
            "device": str(device),
            "loss_method": LOSS_METHOD_NAMES[args.loss],
            "loss_parameters": configured_loss_parameters,
            "official_test_read": False,
        }, indent=2))
        return {"status": "INPUTS_VALID"}

    output_dir = project_path(args.output_dir, "Output directory")
    report_path = output_dir / "run_manifest.json"
    if report_path.is_file():
        existing = read_json(report_path)
        existing_config = existing.get("config", {})
        loss_config_matches = existing_config.get("loss_parameters") == configured_loss_parameters
        if args.loss in {"L0", "L1"} and "loss_parameters" not in existing_config:
            loss_config_matches = existing.get("unknown_label_weight") == (
                UNKNOWN_LABEL_WEIGHT if args.loss == "L1" else None
            )
        require(
            existing.get("status") == "completed"
            and existing.get("seed") == args.seed
            and existing.get("arm") == args.arm
            and existing.get("loss_arm") == args.loss
            and existing.get("train_candidates_eligible") == len(train_rows)
            and existing.get("validation_candidates") == len(validation_rows)
            and existing.get("train_sha256") == sha256_file(artifacts["train_path"])
            and existing.get("validation_sha256") == sha256_file(artifacts["validation_path"])
            and existing.get("pretrained_model") == args.pretrained_model
            and existing.get("pretrained_model_revision") == model_revision
            and existing_config.get("epochs") == args.epochs
            and existing_config.get("batch_size") == args.batch_size
            and existing_config.get("accumulation") == args.accumulation
            and existing_config.get("eval_batch_size") == args.eval_batch_size
            and existing_config.get("encoder_lr") == args.encoder_lr
            and existing_config.get("weight_decay") == args.weight_decay
            and existing_config.get("gradient_clip") == args.gradient_clip
            and loss_config_matches,
            f"Existing run does not match this request: {output_dir}",
        )
        print(f"[train] completed run preserved: {report_path}")
        return existing
    require(not output_dir.exists(), f"Partial output directory exists; preserve and inspect it: {output_dir}")

    validate_tokenizer_compatibility(tokenizer, args.pretrained_model, model_revision)
    set_seed(args.seed)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    model = neural.EntityMarkerEncoder(
        args.pretrained_model,
        vocab_size,
        len(labels),
        revision=model_revision,
    ).to(device)
    model_max_length = int(getattr(model.encoder.config, "max_position_embeddings", 0))
    for row in train_rows + validation_rows:
        input_ids = row.get("model_inputs", row)["input_ids"]
        require(len(input_ids) <= model_max_length,
                f"Feature sequence exceeds encoder limit ({model_max_length}): {row.get('id')}")

    output_dir.mkdir(parents=True)
    model.encoder.config.to_json_file(str(output_dir / "encoder_config.json"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.encoder_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = None
    best_predictions = None
    history = []
    update_count = 0
    use_amp = device.type == "cuda"
    training_started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ordering = np.random.permutation(len(train_rows)).tolist()
        epoch_losses = []
        window_size = args.batch_size * args.accumulation
        for window_start in range(0, len(ordering), window_size):
            window = ordering[window_start : window_start + window_size]
            optimizer.zero_grad(set_to_none=True)
            window_count = len(window)
            for batch_start in range(0, len(window), args.batch_size):
                indices = window[batch_start : batch_start + args.batch_size]
                batch = [train_rows[index] for index in indices]
                inputs, positions = neural.make_batch(batch, tokenizer, marker_ids, device)
                targets = target_matrix(batch, labels, device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(inputs, positions)
                    per_candidate = batch_loss(
                        logits,
                        targets,
                        args.loss,
                        pos_weight=pos_weight,
                        focal_alpha=args.focal_alpha,
                        focal_gamma=args.focal_gamma,
                        asl_gamma_positive=args.asl_gamma_positive,
                        asl_gamma_negative=args.asl_gamma_negative,
                        asl_clip=args.asl_clip,
                    )
                    loss = per_candidate.sum() / window_count
                scaler.scale(loss).backward()
                epoch_losses.extend(per_candidate.detach().float().cpu().tolist())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            update_count += 1
            done = min(window_start + window_count, len(ordering))
            if update_count == 1 or update_count % 250 == 0:
                print(f"[train] epoch {epoch}: {done}/{len(ordering)} candidates", flush=True)

        validation_report, validation_predictions = evaluate(
            model,
            validation_rows,
            labels,
            tokenizer,
            marker_ids,
            device,
            args.eval_batch_size,
            split_sha256=sha256_file(artifacts["validation_path"]),
        )
        metrics = validation_report["all_candidates"]
        checkpoint_path = output_dir / f"epoch{epoch}.safetensors"
        checkpoint_hash = save_checkpoint(checkpoint_path, model)
        entry = {
            "epoch": epoch,
            "update": update_count,
            "mean_train_loss": statistics.mean(epoch_losses),
            "validation_micro_f1": metrics["micro_f1"],
            "validation_macro_f1_fixed_inventory": metrics["macro_f1_fixed_inventory"],
            "validation_metrics": validation_report,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
        }
        history.append(entry)
        if best is None or entry["validation_micro_f1"] > best["validation_micro_f1"]:
            best = entry
            best_predictions = validation_predictions
            shutil.copy2(checkpoint_path, output_dir / "best.safetensors")
        print(
            f"[train] epoch {epoch}: loss={entry['mean_train_loss']:.6f}, "
            f"validation micro-F1={entry['validation_micro_f1']:.6f}, "
            f"macro-F1={entry['validation_macro_f1_fixed_inventory']:.6f}",
            flush=True,
        )

    require(best is not None and best_predictions is not None, "No checkpoint was selected")
    best_path = output_dir / "best.safetensors"
    selected_metrics = best["validation_metrics"]
    metrics_path = output_dir / "validation_metrics.json"
    write_json(metrics_path, selected_metrics)
    prediction_path = output_dir / "validation_predictions.jsonl"
    require(len(best_predictions) == len(validation_rows), "Validation prediction count mismatch")
    require(
        [row["candidate_id"] for row in best_predictions]
        == [row.get("id") for row in validation_rows],
        "Validation predictions do not preserve candidate order",
    )
    write_jsonl(prediction_path, best_predictions)
    manifest = {
        "status": "completed",
        "date": date.today().isoformat(),
        "dataset": "FinRED",
        "seed": args.seed,
        "arm": args.arm,
        "loss_arm": args.loss,
        "loss_method": LOSS_METHOD_NAMES[args.loss],
        "loss_parameters": configured_loss_parameters,
        "device": str(device),
        "runtime": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "transformers_version": transformers.__version__,
            "tokenizers_version": tokenizers.__version__,
            "safetensors_version": safetensors.__version__,
            "accelerator_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "training_loop_seconds": round(time.monotonic() - training_started, 3),
            "run_wall_seconds": round(time.monotonic() - run_started, 3),
        },
        "training_code_sha256": sha256_file(Path(__file__)),
        "neural_code_sha256": sha256_file(Path(__file__).with_name("neural.py")),
        "metrics_code_sha256": sha256_file(Path(__file__).with_name("metrics.py")),
        "pretrained_model": args.pretrained_model,
        "pretrained_model_revision": model_revision or getattr(model.encoder.config, "_commit_hash", None),
        "feature_dir": str(args.feature_dir),
        "feature_manifest_path": str(artifacts["manifest_path"]),
        "feature_manifest_sha256": sha256_file(artifacts["manifest_path"]),
        "tokenizer_path": str(artifacts["tokenizer_path"]),
        "tokenizer_sha256": sha256_file(artifacts["tokenizer_path"]),
        "label_schema_path": str(artifacts["schema_path"]),
        "label_schema_sha256": sha256_file(artifacts["schema_path"]),
        "train_path": str(artifacts["train_path"]),
        "train_sha256": sha256_file(artifacts["train_path"]),
        "validation_path": str(artifacts["validation_path"]),
        "validation_sha256": sha256_file(artifacts["validation_path"]),
        "validation_predictions_path": str(prediction_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "validation_prediction_rows": len(best_predictions),
        "validation_candidate_order_sha256": ordered_ids_sha256(validation_rows),
        "validation_metrics_path": str(metrics_path),
        "validation_metrics_sha256": sha256_file(metrics_path),
        "validation_prediction_format": {
            "scores": "probabilities in the run_manifest labels order",
            "gold_targets": "binary targets in the run_manifest labels order",
            "candidate_id": "feature row id",
            "candidate_index": "zero-based order in the validation input file",
            "split": "validation",
            "split_sha256": "SHA-256 of the validation input file",
        },
        "train_candidates_eligible": len(train_rows),
        "validation_candidates": len(validation_rows),
        "labels": labels,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "accumulation": args.accumulation,
            "eval_batch_size": args.eval_batch_size,
            "encoder_lr": args.encoder_lr,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "unknown_label_weight": UNKNOWN_LABEL_WEIGHT if args.loss == "L1" else None,
            "loss_parameters": configured_loss_parameters,
            "selection_metric": "validation_micro_f1_at_fixed_threshold_0.5",
            "selection_tie_break": "earliest_epoch_on_exact_tie",
            "threshold": 0.5,
            "mixed_precision": use_amp,
        },
        "unknown_label_weight": UNKNOWN_LABEL_WEIGHT if args.loss == "L1" else None,
        "test_file_read": False,
        "official_test_read": False,
        "stress_test_read": False,
        "best_checkpoint": {
            "path": str(best_path),
            "sha256": sha256_file(best_path),
            "epoch": best["epoch"],
            "update": best["update"],
            "validation_micro_f1": best["validation_micro_f1"],
            "validation_macro_f1_fixed_inventory": best[
                "validation_macro_f1_fixed_inventory"
            ],
        },
        "history": history,
    }
    write_json(report_path, manifest)
    print(f"[train] run manifest: {report_path}")
    print(f"[train] checkpoint SHA-256: {manifest['best_checkpoint']['sha256']}")
    print("[train] FinRED official test and stress-test inputs were not read.")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an ORGU C2 entity-marker encoder")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--arm", choices=("C2",), required=True)
    parser.add_argument("--loss", choices=tuple(LOSS_METHOD_NAMES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--focal-alpha", type=float, default=FOCAL_ALPHA)
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--asl-gamma-positive", type=float, default=ASL_GAMMA_POS)
    parser.add_argument("--asl-gamma-negative", type=float, default=ASL_GAMMA_NEG)
    parser.add_argument("--asl-clip", type=float, default=ASL_CLIP)
    parser.add_argument(
        "--pretrained-model",
        default=os.environ.get("ORGU_ENCODER_MODEL", DEFAULT_ENCODER),
    )
    parser.add_argument("--pretrained-model-revision")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"), default="auto")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train(args)
    except Exception as exc:
        print(f"[train] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())