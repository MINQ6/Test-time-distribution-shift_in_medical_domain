import argparse
import logging

from utils.analysis import (
    analyze_global_lora_sources,
    analyze_upload_payloads,
    extract_global_lora_from_checkpoints,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze final-stage client Global LoRA and generate pairwise similarity heatmaps/dendrograms."
    )
    parser.add_argument("--uploads_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--source_name", type=str, default="global_lora_analysis")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[],
        help="Optional extra analysis methods to run in addition to layerwise flatten similarity. Choose from: svd qr both",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if bool(args.uploads_path) == bool(args.checkpoint_dir):
        raise ValueError("Provide exactly one of --uploads_path or --checkpoint_dir.")

    if args.uploads_path is not None:
        import torch

        uploads = torch.load(args.uploads_path, map_location="cpu", weights_only=False)
        artifact_paths = analyze_upload_payloads(
            uploads=uploads,
            output_dir=args.output_dir,
            source_name=args.source_name,
            methods=args.methods,
        )
    else:
        artifact_paths = analyze_global_lora_sources(
            client_global_lora=extract_global_lora_from_checkpoints(args.checkpoint_dir),
            output_dir=args.output_dir,
            source_name=args.source_name,
            methods=args.methods,
        )

    print("Saved analysis artifacts:")
    for name, path in sorted(artifact_paths.items()):
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
