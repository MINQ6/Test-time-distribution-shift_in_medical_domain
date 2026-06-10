import argparse

from core.server import FederatedServer
from utils.train import run_local_finetuned_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train local-finetuned dual-LoRA baselines for all clients.")
    parser.add_argument("--model_id", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--local_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--checkpoint_prefix", default="local_finetuned_dual_lora")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    server = FederatedServer(
        model_id=args.model_id,
        num_clients=args.num_clients,
        num_rounds=1,
        local_epochs=args.local_epochs,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
        rank=args.rank,
    )

    saved_paths = run_local_finetuned_training(
        server=server,
        checkpoint_prefix=args.checkpoint_prefix,
    )

    print("Local-finetuned checkpoints:")
    for client_id, path in saved_paths.items():
        print(f"  {client_id}: {path}")


if __name__ == "__main__":
    main()
