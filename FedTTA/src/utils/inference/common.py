import collections
import logging
import random
from pathlib import Path

import numpy as np
import torch


logging.basicConfig(level=logging.INFO, format="%(filename)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "meta-llama/Llama-2-7b-hf"
CHECKPOINT_DIR = Path("/home/0630rb/research/src/checkpoints")
DATA_ROOT = Path("/home/0630rb/research/src/data")


def checkpoint_suffix(client_id: str) -> str:
    return client_id.split("client_")[-1] if client_id.startswith("client_") else client_id


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

PROMPT_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"
