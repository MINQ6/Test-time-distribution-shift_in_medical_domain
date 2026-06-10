import logging
import os
from typing import Dict

import torch
from dotenv import load_dotenv

from core.client import FederatedClient

logging.basicConfig(level=logging.INFO, format="%(filename)s - %(message)s")
logger = logging.getLogger(__name__)


class FederatedServer:
    """
    federated learning에 필요한 전역 상태를 보관하는 server container.

    실제 round orchestration과 aggregation 로직은 utils/train.py에서 수행한다.
    이 클래스는 실험 설정, client 목록, personalized global LoRA 상태를 담는 역할에 집중한다.
    """

    def __init__(
        self,
        model_id: str,
        num_clients: int,
        num_rounds: int,
        local_epochs: int,
        dataset_name: str,
        batch_size: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        num_samples: int,
        max_length: int,
        seed: int,
        rank: int,
        lora_alpha: int,
        lora_dropout: float,
        learning_rate: float,
        train_on_inputs: bool,
        lora_target_modules: list[str],
        stage1_only: bool = False,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        wandb_entity: str | None = None,
        save_epoch_snapshots: bool = False,
        warmup_epochs: int = 2,
        personalization_epochs: int | None = None,
        cluster_similarity_factor: str = "B",
        cluster_orthogonalization: bool = True,
    ):
        # HF_TOKEN을 환경변수에서 읽어 모델 로딩에 사용한다.
        project_src_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )
        load_dotenv(os.path.join(project_src_dir, ".env"))
        load_dotenv()
        # Qwen3 등 public/로컬 모델은 토큰 없이 로드 가능. None이면 그대로 진행.
        self.hf_token = os.getenv("HF_TOKEN")

        # 아래 값들은 round orchestration에서 공통 설정으로 사용된다.
        self.model_id = model_id
        self.client_ids = [f"client_{idx}" for idx in range(1, num_clients + 1)]
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_samples = num_samples
        self.max_length = max_length
        self.seed = seed
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.train_on_inputs = train_on_inputs
        self.lora_target_modules = lora_target_modules
        self.stage1_only = stage1_only
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_entity = wandb_entity
        self.save_epoch_snapshots = save_epoch_snapshots
        self.warmup_epochs = warmup_epochs
        self.personalization_epochs = personalization_epochs or local_epochs
        self.cluster_similarity_factor = cluster_similarity_factor
        self.cluster_orthogonalization = cluster_orthogonalization

        # 서버는 각 client별 client container를 하나씩 들고 있다.
        self.clients: Dict[str, FederatedClient] = {
            client_id: FederatedClient(client_id) for client_id in self.client_ids
        }
        # target client마다 다른 personalized global LoRA를 만들기 때문에 dict로 관리한다.
        self.global_lora_states: Dict[str, Dict[str, torch.Tensor] | None] = {
            client_id: None for client_id in self.client_ids
        }
        self.cluster_assignments: Dict[str, int] = {}
        self.cluster_global_lora_states: Dict[int, Dict[str, torch.Tensor]] = {}
