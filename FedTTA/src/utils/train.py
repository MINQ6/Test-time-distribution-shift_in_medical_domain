"""Training facade.

실제 구현은 utils/training 하위 모듈로 분리하고,
기존 import 경로 호환성을 위해 여기서 다시 노출한다.
"""

from utils.training.federated import run_federated_training
from utils.training.feddpa_f import run_feddpa_f_training
from utils.training.local_finetuned import run_local_finetuned_training, train_local_client
from utils.training.local_round import run_local_federated_round, run_client_round

__all__ = [
    "run_feddpa_f_training",
    "run_federated_training",
    "run_local_federated_round",
    "run_local_finetuned_training",
    "run_client_round",
    "train_local_client",
]
