import logging
from typing import Dict

import torch

logging.basicConfig(level=logging.INFO, format="%(filename)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class FederatedClient:
    """
    각 client의 상태만 보관하는 lightweight container.

    학습 실행 로직은 utils/train.py로 옮기고,
    이 클래스는 local state / personalized global LoRA 상태를 들고 있는 역할에 집중한다.
    """

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.cluster_id: int | None = None
        # 각 client가 자기 데이터로 학습해 유지하는 Local LoRA 파라미터
        self.local_lora_state: Dict[str, torch.Tensor] | None = None
        # 서버 집계 결과로 round마다 갱신되는 personalized Global LoRA 파라미터
        self.global_lora_state: Dict[str, torch.Tensor] | None = None
        # FedALT: client별로 로컬 보관(집계 X)하는 mixer 파라미터
        self.mixer_state: Dict[str, torch.Tensor] | None = None

    def set_initial_state(self, state_dict: Dict[str, torch.Tensor]):
        # 초기 broadcast state를 local/global LoRA 두 부분으로 나눠 저장한다.
        self.set_adapter_state(state_dict)

    def set_adapter_state(self, state_dict: Dict[str, torch.Tensor]):
        # 모델 전체 adapter state dict에서 local LoRA 부분만 따로 떼어 보관한다.
        self.local_lora_state = {
            key: value.detach().cpu().clone()
            for key, value in state_dict.items()
            if ".local_lora." in key
        }
        # 모델 전체 adapter state dict에서 global LoRA 부분만 따로 떼어 보관한다.
        self.global_lora_state = {
            key: value.detach().cpu().clone()
            for key, value in state_dict.items()
            if ".global_lora." in key
        }

    def get_adapter_state(self) -> Dict[str, torch.Tensor]:
        if self.local_lora_state is None or self.global_lora_state is None:
            raise ValueError(f"{self.client_id} is missing local/global LoRA state.")

        # 학습/추론 모델에 다시 load할 수 있도록 local/global 상태를 하나의 state dict로 합친다.
        return {
            **{
                key: value.detach().cpu().clone()
                for key, value in self.local_lora_state.items()
            },
            **{
                key: value.detach().cpu().clone()
                for key, value in self.global_lora_state.items()
            },
        }

    def set_global_lora_state(self, global_lora_state: Dict[str, torch.Tensor] | None):
        if global_lora_state is None:
            self.global_lora_state = None
            return

        # 서버가 내려준 personalized global LoRA만 별도로 덮어쓴다.
        self.global_lora_state = {
            key: value.detach().cpu().clone() for key, value in global_lora_state.items()
        }

    def set_cluster_id(self, cluster_id: int | None):
        self.cluster_id = cluster_id
