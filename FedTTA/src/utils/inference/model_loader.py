from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from models.dual_lora_model import setup_model_with_dual_lora


def _normalize_generation_config(model) -> None:
    """Deterministic evaluation에 맞게 generation warning을 유발하는 옵션을 정리한다."""
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    generation_config.do_sample = False
    generation_config.max_length = None
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None


def load_base_model(hf_token: str, model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _normalize_generation_config(model)
    model.eval()
    return model


def load_adapter_model(
    hf_token: str,
    checkpoint_path: Path,
    device: torch.device,
    model_id: str,
    mixer_path: Path | None = None,
):
    """Dual LoRA adapter (local/global) checkpoint를 로드한다.

    mixer_path가 주어지고 존재하면 FedALT mixer 가중치도 함께 로드한다(use_mixer는
    호출부에서 set_use_mixer로 켠다). mixer 로드 여부는 호출부에서 mixer_path 존재로
    판단할 수 있다. 반환값: model (기존 호출부 호환).
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    rank_key = next(
        key for key in state_dict
        if key.endswith(".global_lora.lora_A.weight")
    )
    rank = int(state_dict[rank_key].shape[0])

    model = setup_model_with_dual_lora(
        model_id,
        hf_token,
        rank=rank,
    )
    model.dual_lora_adapter.load_state_dict(state_dict, strict=True)

    if mixer_path is not None and Path(mixer_path).exists():
        mixer_state = torch.load(mixer_path, map_location=device)
        model.dual_lora_adapter.load_mixer_state_dict(mixer_state)

    _normalize_generation_config(model)
    model.eval()
    return model
