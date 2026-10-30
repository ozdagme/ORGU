# ORGU reproducibility code

This repository contains the code used for the FinRED relation-extraction experiments reported in **Operating Points Matter: Revisiting Negative-Target Downweighting for Financial Relation Extraction under Entity Shift**.

## Scope

The repository contains training, inference, evaluation, and analysis code only. It does not contain FinRED data, annotations, split membership lists, prediction archives, result tables, or model checkpoints. The `data/` and `results/` directories contain only `.gitkeep` placeholders. Add authorized data and results locally when they are available; do not commit them unless the authors have cleared their release.

The experiment runners consume frozen, entity-marker-encoded feature files and their SHA-256 manifests. This release does not reconstruct those private preprocessing artifacts from raw data. A complete reproduction therefore requires the exact frozen feature inputs described below, in addition to the relevant raw official-test files for the official-test analysis.

## Environment

The manuscript's recorded training stack is Python 3.11, PyTorch 2.9.1+cu128, Transformers 4.57.1, tokenizers 0.22.1, and safetensors 0.8.0. Install the PyTorch build matching your hardware, then install the remaining pinned packages:

```bash
python -m pip install -r requirements.txt
```

On a CUDA system, select the matching PyTorch wheel from the official PyTorch installation selector. The manuscript run used CUDA 12.8. For CPU-only use, install the CPU PyTorch build instead. Record the installed versions with each run; the training manifests capture runtime and input provenance.

From the repository root, expose the source package:

```bash
export PYTHONPATH=src
```

In PowerShell:

```powershell
$env:PYTHONPATH = "src"
```

## Required local inputs

The default entity-disjoint feature directory is `data/features/finred_entity_disjoint_v1/`. It must contain:

- `context_manifest.json`, pinning the model ID/revision and repository-relative train and validation files under the `C2` arm, with file hashes.
- `label_schema.json`, containing the ordered 29-label FinRED inventory.
- `tokenizer.json`, compatible with `google-bert/bert-base-cased` revision `cd5ef92a9fb2f889e972770a36d4ed042daf221e` and including `[SUBJ]`, `[/SUBJ]`, `[OBJ]`, and `[/OBJ]`.
- The manifest-referenced train and validation JSONL files. Each row must provide `id`, `record_id`, `labels`, and `model_inputs.input_ids`; optional attention masks, token-type IDs, and marker positions are supported.
- For entity-disjoint test evaluation, a held-out encoded JSONL file with the same row structure. Supply it only to the evaluation stage after validation thresholds are frozen.

The feature manifest must identify the model as `google-bert/bert-base-cased` and pin the revision above. Training uses the `[CLS]`, subject-start, object-start, and absolute-difference representation with dropout 0.1.

For the manually annotated official FinRED test, place the authorized `relations.txt`, `test.txt`, `test.sent`, `test.tup`, and `test.pointer` files in `data/raw/finred_official/`. The evaluation script verifies the official files and their alignment before inference.

## Main five-loss experiment

This runs BCE, NegDW-BCE, focal loss, ASL, and positive-weighted BCE on ten paired seeds. Run the training preflight before training:

Defaults match the paper: seeds 13, 42, 87, 101, 202, 303, 404, 505, 606, and 707; three epochs; batch size 2; gradient accumulation 8; evaluation batch size 8; learning rate 2e-5; weight decay 0.01; and gradient clipping 1.0. Checkpoint selection uses validation micro-F1 at 0.5, with the earliest epoch winning an exact tie. The loss settings are NegDW-BCE rho=0.1, focal alpha=0.25/gamma=2, ASL gamma-positive=0/gamma-negative=4/clip=0.05, and unclipped per-relation positive weights computed from training labels only.

```bash
python scripts/run_m1m4_loss_family_comparison.py --stage train --check-only --device cuda
python scripts/run_m1m4_loss_family_comparison.py --stage train --device cuda
python scripts/run_m1m4_loss_family_comparison.py --stage freeze-thresholds
python scripts/run_m1m4_loss_family_comparison.py --stage evaluate-test --test-file data/features/finred_entity_disjoint_v1/C2/test.jsonl
```

The test file is opened only in the final stage. The runner writes checkpoints and manifests under `runs/`. It records validation-selected thresholds before held-out inference.

To bundle the generated prediction artifacts and calculate the two-way bootstrap summaries:

```bash
python scripts/package_m1m4_loss_family_review_bundle.py
python scripts/analyze_loss_family_macro_ap.py
python scripts/analyze_loss_family_macro_f1_bootstrap.py
```

The bootstrap scripts use paired seed and source-record-cluster resampling. Their defaults are 10,000 replicates; options for the input bundle, output path, batch size, and RNG seed are available through `--help`.

## Controlled label removal

This experiment uses paired seeds 13, 42, and 87 and nested removal rates of 0%, 10%, 30%, and 50%. It modifies only training positives and keeps the same mask for BCE and NegDW-BCE:

```bash
python scripts/run_controlled_label_removal_local.py --check-only --device cuda
python scripts/run_controlled_label_removal_local.py --device cuda
```

For held-out evaluation, freeze the validation thresholds first, then supply a separate encoded file:

```bash
python scripts/run_controlled_label_removal_local.py --stage evaluate --evaluation-file data/features/finred_entity_disjoint_v1/C2/test.jsonl --device cuda
```

## Official-test evaluation

The official-test stage requires the 20 BCE/NegDW checkpoints from the ten-seed loss-family run and the raw official-test files. It writes a score archive, not checkpoint weights:

```bash
python scripts/run_official_finred_test.py --check-only --device cuda
python scripts/run_official_finred_test.py --device cuda
python scripts/analyze_official_finred_test_results.py
```

Calibration and historical-artifact audit scripts are included for the corresponding authorized local inputs. See each script's `--help` and input constants before running them. The official and entity-disjoint test results are exploratory and evaluate agreement with recorded annotations, not verified relation truth.

## Synthetic checks

Run the in-memory tests without loading any dataset:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```


## Repository layout

- `src/orgu/`: model, loss functions, trainer, metrics, and prediction utilities.
- `scripts/`: experiment drivers, held-out evaluation, and analysis programs.
- `tests/`: in-memory synthetic checks; no dataset fixtures are included.
- `data/`, `results/`: empty placeholders; local data and outputs are ignored by Git.
- `docs/GITHUB_RELEASE.md`: repository creation, release, and later Zenodo notes.

No license is included pending author approval. Choose and add a license before inviting third-party reuse.