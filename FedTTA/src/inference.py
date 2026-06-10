import argparse

from utils.inference import run_inference
from utils.inference.common import MODEL_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--dataset_name", default="dataset1")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--compare_base", action="store_true")
    parser.add_argument("--disable_local_finetuned", action="store_true")
    parser.add_argument("--compare_lora", action="store_true")
    parser.add_argument("--lora_checkpoint_prefix", default="lora_adapter")
    parser.add_argument("--disable_ttp", action="store_true")
    parser.add_argument("--ttp_num_samples", type=int, default=None)
    parser.add_argument("--result_json", default=None)
    parser.add_argument("--inference_batch_size", type=int, default=8)
    args = parser.parse_args()

    run_inference(
        target_client=args.client,
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        num_samples=args.num_samples,
        compare_base=args.compare_base,
        compare_local_finetuned=not args.disable_local_finetuned,
        compare_lora=args.compare_lora,
        run_ttp=not args.disable_ttp,
        ttp_num_samples=args.ttp_num_samples,
        result_json=args.result_json,
        inference_batch_size=args.inference_batch_size,
        lora_checkpoint_prefix=args.lora_checkpoint_prefix,
    )


if __name__ == "__main__":
    main()
