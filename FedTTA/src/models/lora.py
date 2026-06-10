from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRA(nn.Module):
    """A/B 행렬을 명시적으로 가진 표준 LoRA block."""

    def __init__(
        self,
        hidden_size: int | None = None,
        rank: int | None = None,
        alpha: int = 16,
        dropout: float = 0.05,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ):
        super().__init__()
        if input_dim is None:
            if hidden_size is None:
                raise ValueError("Either hidden_size or input_dim must be provided.")
            input_dim = hidden_size
        if output_dim is None:
            if hidden_size is None:
                raise ValueError("Either hidden_size or output_dim must be provided.")
            output_dim = hidden_size

        self.input_dim = input_dim
        self.output_dim = output_dim
        rank_base = hidden_size or min(input_dim, output_dim)
        self.rank = rank or max(1, rank_base // 16)
        self.alpha = alpha
        self.dropout_p = dropout
        self.scaling = self.alpha / self.rank

        self.lora_A = nn.Linear(self.input_dim, self.rank, bias=False)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lora_B = nn.Linear(self.rank, self.output_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # PEFT 기본 LoRA 초기화와 유사하게 A는 kaiming_uniform, B는 zero-init으로 둔다.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        input_dtype = x.dtype
        x = self.lora_A(x)
        x = self.dropout(x)
        x = self.lora_B(x)
        return (x * self.scaling).to(dtype=input_dtype)
