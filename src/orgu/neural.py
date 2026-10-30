from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel


class EntityMarkerEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str | Path,
        vocab_size: int,
        num_labels: int,
        dropout: float = 0.1,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        source = str(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(
            source,
            revision=revision,
            local_files_only=Path(source).exists(),
            trust_remote_code=False,
        )
        if self.encoder.get_input_embeddings().num_embeddings != vocab_size:
            self.encoder.resize_token_embeddings(vocab_size)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 4, num_labels)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        marker_positions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        encoder_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in inputs and int(
            getattr(self.encoder.config, "type_vocab_size", 1)
        ) > 1:
            encoder_inputs["token_type_ids"] = inputs["token_type_ids"]
        hidden = self.encoder(**encoder_inputs).last_hidden_state
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        head = hidden[batch_indices, marker_positions["head_start"]]
        tail = hidden[batch_indices, marker_positions["tail_start"]]
        pooled = torch.cat(
            (hidden[:, 0], head, tail, torch.abs(head - tail)),
            dim=-1,
        )
        return self.classifier(self.dropout(pooled))


def make_batch(
    rows: list[dict],
    tokenizer,
    marker_ids: dict[str, int],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not rows:
        raise ValueError("Cannot construct an empty batch")
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise ValueError("Feature tokenizer has no [PAD] token")
    if len(set(marker_ids.values())) != 4:
        raise ValueError("Entity marker IDs must be present and distinct")

    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    type_ids: list[list[int]] = []
    positions: dict[str, list[int]] = {name: [] for name in marker_ids}
    for row in rows:
        model_inputs = row.get("model_inputs", row)
        ids = [int(value) for value in model_inputs["input_ids"]]
        if not ids:
            raise ValueError(f"Empty token sequence in candidate {row.get('id')}")
        attention = [
            int(value)
            for value in model_inputs.get("attention_mask", [1] * len(ids))
        ]
        token_types = [
            int(value)
            for value in model_inputs.get("token_type_ids", [0] * len(ids))
        ]
        if len(attention) != len(ids) or len(token_types) != len(ids):
            raise ValueError(f"Mismatched input lengths in candidate {row.get('id')}")

        row_positions = row.get("input_marker_positions") or {}
        for name, marker_id in marker_ids.items():
            found = [index for index, value in enumerate(ids) if value == marker_id]
            position = int(row_positions[name]) if name in row_positions else None
            if position is None:
                if len(found) != 1:
                    raise ValueError(
                        f"Expected one {name} marker in candidate {row.get('id')}"
                    )
                position = found[0]
            if position not in found:
                raise ValueError(
                    f"Incorrect {name} position in candidate {row.get('id')}"
                )
            positions[name].append(position)
        if not (
            positions["head_start"][-1] < positions["head_end"][-1]
            and positions["tail_start"][-1] < positions["tail_end"][-1]
        ):
            raise ValueError(f"Invalid entity marker order in candidate {row.get('id')}")
        sequences.append(ids)
        masks.append(attention)
        type_ids.append(token_types)

    width = max(map(len, sequences))
    padded_ids = [sequence + [pad_id] * (width - len(sequence)) for sequence in sequences]
    padded_masks = [mask + [0] * (width - len(mask)) for mask in masks]
    padded_types = [types + [0] * (width - len(types)) for types in type_ids]
    inputs = {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(padded_masks, dtype=torch.long, device=device),
        "token_type_ids": torch.tensor(padded_types, dtype=torch.long, device=device),
    }
    marker_positions = {
        name: torch.tensor(values, dtype=torch.long, device=device)
        for name, values in positions.items()
    }
    return inputs, marker_positions