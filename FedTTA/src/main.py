from args import build_parser
from core.server import FederatedServer
from utils.train import run_federated_training


def main() -> None:
    args = build_parser().parse_args()
    gradient_accumulation_steps = args.gradient_accumulation_steps
    if gradient_accumulation_steps is None:
        if args.batch_size % args.micro_batch_size != 0:
            raise ValueError(
                f"batch_size ({args.batch_size}) must be divisible by micro_batch_size ({args.micro_batch_size}) "
                "when gradient_accumulation_steps is not provided."
            )
        gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    server = FederatedServer(
        model_id=args.model_id,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        train_on_inputs=args.train_on_inputs,
        lora_target_modules=[item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
        stage1_only=args.stage1_only,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        save_epoch_snapshots=args.save_epoch_snapshots,
        warmup_epochs=args.warmup_epochs,
        personalization_epochs=args.personalization_epochs,
        cluster_similarity_factor=args.cluster_similarity_factor,
        cluster_orthogonalization=not args.disable_cluster_orthogonalization,
    )

    if args.method == "ours":
        run_federated_training(server)
    elif args.method == "feddpa_f":
        from utils.training.feddpa_f import run_feddpa_f_training

        run_feddpa_f_training(server)
    elif args.method == "feddpa_t":
        from utils.training.feddpa_t import run_feddpa_t_training

        run_feddpa_t_training(server)
    elif args.method == "fedalt":
        from utils.training.fedalt import run_fedalt_training

        run_fedalt_training(server)
    elif args.method == "lora":
        from methods.lora.federated import run_federated_lora_training

        run_federated_lora_training(server)
    elif args.method == "local_finetuned":
        from utils.training.local_finetuned import run_local_finetuned_training

        run_local_finetuned_training(server)
    elif args.method == "centralized":
        from utils.training.centralized import run_centralized_training

        run_centralized_training(server)
    else:
        raise ValueError(f"Unsupported method: {args.method}")


if __name__ == "__main__":
    main()
