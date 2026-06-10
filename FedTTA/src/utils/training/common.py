import logging
import os
import random
from typing import Dict

import numpy as np
import torch
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    """재현성을 위해 모든 난수 생성기의 seed를 맞춘다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def checkpoint_suffix(client_id: str) -> str:
    return client_id.split("client_")[-1] if client_id.startswith("client_") else client_id


def load_hf_token() -> str | None:
    """학습/추론에 사용할 Hugging Face 토큰을 로드한다. 없으면 None(public 모델용)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(current_dir, "..", "..", ".env"),
        os.path.join(current_dir, "..", ".env"),
    ]
    loaded_path = None
    for dotenv_path in candidate_paths:
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            loaded_path = dotenv_path
            break

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        # Qwen3 등 public 모델은 토큰 없이 로드 가능. gated 모델만 필요.
        logger.warning(
            "HF_TOKEN is not set; proceeding without it (OK for public models like Qwen3). "
            "Set it in src/.env only if you need a gated model."
        )
        return None
    if loaded_path:
        logger.info("Loaded HF token from %s", os.path.abspath(loaded_path))
    else:
        logger.info("HF token loaded from process environment.")
    return hf_token


def clone_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """GPU state dict를 CPU tensor로 안전하게 복사해 서버/디스크로 넘기기 쉽게 만든다."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}
