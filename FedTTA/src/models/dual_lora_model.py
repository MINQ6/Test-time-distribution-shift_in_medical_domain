import logging
import os

import torch
from transformers import AutoModelForCausalLM

from .dual_lora_adapter import DualLoRAAdapter

logging.basicConfig(level=logging.INFO, format="%(filename)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_model_with_dual_lora(
    model_id: str,
    token: str,
    rank: int | None = None,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | tuple[str, ...] | None = None,
):
    """
    backbone의 모든 attention/FFN projection에 FedDPA 스타일 dual LoRA를 주입한다.
    local LoRA와 global LoRA를 동시에 두고, backbone은 freeze한다.
    """
    logger.info(f"Downloading & Loading base model: {model_id}")

    use_8bit = os.environ.get("LOAD_IN_8BIT", "0") == "1"
    if use_8bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("LOAD_IN_8BIT=1 → int8 (bitsandbytes) + fp16 compute (matches FedDPA paper original)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )

    for param in model.parameters():
        param.requires_grad = False

    dual_lora_adapter = DualLoRAAdapter(
        model=model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_module_suffixes=tuple(target_modules) if target_modules is not None else None,
    )

    model.dual_lora_adapter = dual_lora_adapter

    model.dual_lora_adapter.configure_training()

    logger.info("Base model parameters frozen.")
    logger.info(
        f"Attached Dual LoRA Adapter to {len(dual_lora_adapter.target_module_names)} attention/FFN projections "
        f"(rank={rank}, dtype={model.dtype})."
    )
    return model
