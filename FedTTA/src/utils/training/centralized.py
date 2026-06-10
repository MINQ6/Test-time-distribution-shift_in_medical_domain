"""Centralized baseline (FedDPA paper Table 1).

A single model is trained on the *concatenation* of all clients' training data —
no federation, no per-client adapter. The resulting LoRA is replicated as
per-client dual_lora_adapter checkpoints so the existing inference pipeline can
load it for each client and evaluate with `--*_adapter_mode global_only`.

Training compute is matched roughly to the total per-client compute of an FL
round (=local_epochs), since the combined dataset is `num_clients × num_samples`
large already; effective optimizer-step count therefore scales with num_clients.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from datasets import Dataset as HFDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from data.loader import _load_local_feddpa_examples, build_generate_and_tokenize_prompt
from models.dual_lora_model import setup_model_with_dual_lora
from utils.training.common import clone_state_dict_to_cpu, load_hf_token, logger, set_seed
from utils.training.local_round import _run_stage


def _checkpoint_dir() -> str:
    ts = os.environ.get("RUN_TIMESTAMP")
    if not ts:
        raise RuntimeError("RUN_TIMESTAMP is not set; refusing to write checkpoints to a non-TS path")
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "checkpoints", ts))


def _load_combined_dataset(server) -> HFDataset:
    """Concatenate every client's training examples into a single HFDataset."""
    combined: list[dict[str, Any]] = []
    for client_id in server.client_ids:
        examples = _load_local_feddpa_examples(
            dataset_name=server.dataset_name,
            client_id=client_id,
            split="train",
            limit=server.num_samples,
        )
        combined.extend(examples)
    if not combined:
        raise ValueError("Centralized: combined dataset is empty")
    logger.info(
        "[Centralized] combined %d examples from %d clients (%d each)",
        len(combined), len(server.client_ids), server.num_samples,
    )
    return HFDataset.from_list(combined)


def _build_centralized_dataloader(server, tokenizer):
    dataset_obj = _load_combined_dataset(server)
    generate_and_tokenize_prompt = build_generate_and_tokenize_prompt(
        tokenizer,
        max_length=server.max_length,
        train_on_inputs=server.train_on_inputs,
    )
    tokenized = dataset_obj.map(generate_and_tokenize_prompt)
    tokenized = tokenized.filter(lambda ex: ex["has_target_tokens"])
    tokenized = tokenized.remove_columns(
        [c for c in tokenized.column_names if c not in {"input_ids", "attention_mask", "labels"}]
    )
    logger.info("[Centralized] tokenized %d supervised examples (max_length=%d)", len(tokenized), server.max_length)
    return DataLoader(
        tokenized,
        batch_size=server.micro_batch_size,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )


def run_centralized_training(server) -> dict[str, str]:
    """Train one model on the union of all clients' data, save per-client copies for inference."""
    checkpoints_dir = _checkpoint_dir()
    os.makedirs(checkpoints_dir, exist_ok=True)

    hf_token = load_hf_token()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for centralized training")
    device = torch.device("cuda")
    set_seed(server.seed)

    tokenizer = AutoTokenizer.from_pretrained(server.model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataloader = _build_centralized_dataloader(server, tokenizer)

    model = setup_model_with_dual_lora(
        server.model_id,
        hf_token,
        rank=server.rank,
        alpha=server.lora_alpha,
        dropout=server.lora_dropout,
        target_modules=server.lora_target_modules,
    )

    # Train only the global slot — local stays at zeros so inference with
    # global_only is well-defined.
    model.dual_lora_adapter.configure_global_training()
    trainable = [p for p in model.dual_lora_adapter.global_parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=server.learning_rate)
    model.train()

    # Compute scale to roughly match one FL client's per-round compute:
    # combined dataset is num_clients × per-client. Train for local_epochs so
    # total optimizer steps ≈ num_clients × local_epochs × per-client-steps,
    # comparable to one FL round's aggregate.
    epochs = server.local_epochs
    logger.info(
        "[Centralized] training %d epochs over combined dataset (=%d FL-round local_epochs equivalent)",
        epochs, server.local_epochs,
    )

    _run_stage(
        model=model,
        dataloader=dataloader,
        device=device,
        optimizer=optimizer,
        trainable_params=trainable,
        round_idx=0,
        client_id="centralized",
        stage_name="Centralized",
        epochs=epochs,
        gradient_accumulation_steps=server.gradient_accumulation_steps,
    )

    final_state = clone_state_dict_to_cpu(model.dual_lora_adapter.state_dict())

    # save as per-client copies so existing inference script (which loads
    # dual_lora_adapter_<client_id>.pth) just works with --*_adapter_mode global_only
    saved: dict[str, str] = {}
    for client_id in server.client_ids:
        save_path = f"{checkpoints_dir}/dual_lora_adapter_{client_id}.pth"
        torch.save(final_state, save_path)
        saved[client_id] = save_path
    logger.info("[Centralized] saved single trained model to %d per-client checkpoint files in %s", len(saved), checkpoints_dir)

    del model, optimizer
    torch.cuda.empty_cache()
    return saved
