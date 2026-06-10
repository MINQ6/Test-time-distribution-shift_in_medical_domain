import argparse
METHOD_CHOICES = ["ours", "feddpa_f", "feddpa_t", "fedalt", "lora", "local_finetuned", "centralized"]
DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B-Base"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the federated learning pipeline for the FedDPA-style dual LoRA model."
    )
    parser.add_argument("--method", type=str, default="ours", choices=METHOD_CHOICES)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset_name", type=str, default="dataset1", choices=["dataset1", "dataset2", "medmcqa"])
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument(
        "--max_length",
        "--cutoff_len",
        dest="max_length",
        type=int,
        default=512,
        help="Maximum tokenized sequence length. Equivalent to FedDPA cutoff_len.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument(
        "--train_on_inputs",
        action="store_true",
        help="If set, include prompt/input tokens in the loss instead of masking them with -100.",
    )
    parser.add_argument("--lora_target_modules", type=str, default="q_proj, v_proj", help="Comma-separated module suffixes to receive LoRA, e.g. q_proj,v_proj",)
    parser.add_argument("--stage1_only", action="store_true", help="Run only FedDPA-F Stage 1 (global LoRA training) and skip local adapter training.",)
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="research_personalization")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_epoch_snapshots", action="store_true", help="Save and analyze per-epoch global LoRA snapshots for each client.",)
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="Number of local epochs used in the initial warm-up round before clustering.",
    )
    parser.add_argument(
        "--personalization_epochs",
        type=int,
        default=None,
        help="Number of local-only personalization epochs after phase 1. Defaults to local_epochs.",
    )
    parser.add_argument(
        "--cluster_similarity_factor",
        type=str,
        default="B",
        choices=["B", "BA"],
        help="Global LoRA factor used for warm-up clustering.",
    )
    parser.add_argument(
        "--disable_cluster_orthogonalization",
        action="store_true",
        help="Disable cluster LoRA BA (= Delta W) orthogonalization after warm-up clustering and after the final round.",
    )
    return parser
