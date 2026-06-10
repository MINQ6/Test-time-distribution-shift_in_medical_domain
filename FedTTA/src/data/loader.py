import os
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizer
from datasets import Dataset as HFDataset
from dotenv import load_dotenv
from typing import Dict, List
import logging
from pathlib import Path

# =========================================================================
# 1. 로깅 및 환경 설정
# =========================================================================
logging.basicConfig(level=logging.INFO, format="%(filename)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING) # 불필요한 HTTP INFO 로그 차단

current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path)
token = os.getenv("HF_TOKEN")
LOCAL_DATA_ROOT = Path(current_dir)
MEDMCQA_DIR = LOCAL_DATA_ROOT / "medmcqa"
# client_1..4 순서 (CLAUDE.md §3.1). prepare_medmcqa.py의 SUBJECTS와 일치해야 함.
MEDMCQA_SUBJECTS = ["Medicine", "Skin", "Surgery", "Orthopaedics"]

PROMPT_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"
IGNORE_INDEX = -100


def _build_alpaca_short_prompt(instruction: str, input_text: str = "", response: str | None = None) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        prompt = PROMPT_INPUT.format(instruction=instruction, input=input_text)
    else:
        prompt = PROMPT_NO_INPUT.format(instruction=instruction)
    if response is not None:
        prompt = f"{prompt}{response.strip()}"
    return prompt


def _tokenize_text(
    tokenizer: PreTrainedTokenizer,
    text: str,
    *,
    max_length: int,
    add_eos_token: bool,
) -> Dict[str, List[int]]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    if (
        add_eos_token
        and encoded["input_ids"]
        and encoded["input_ids"][-1] != tokenizer.eos_token_id
        and len(encoded["input_ids"]) < max_length
    ):
        encoded["input_ids"].append(tokenizer.eos_token_id)
        encoded["attention_mask"].append(1)
    return encoded


def build_generate_and_tokenize_prompt(
    tokenizer: PreTrainedTokenizer,
    *,
    max_length: int,
    train_on_inputs: bool = False,
):
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id must be set before tokenization.")

    skipped_stats = {"count": 0}

    def generate_and_tokenize_prompt(example: Dict[str, str]) -> Dict[str, List[int] | bool]:
        instruction = (example.get("instruction") or example.get("inputs") or "").strip()
        input_text = (example.get("input") or "").strip()
        response = (example.get("output") or example.get("targets") or "").strip()

        prompt_text = _build_alpaca_short_prompt(
            instruction=instruction,
            input_text=input_text,
            response=None,
        )
        full_text = _build_alpaca_short_prompt(
            instruction=instruction,
            input_text=input_text,
            response=response,
        )

        full_encoding = _tokenize_text(
            tokenizer,
            full_text,
            max_length=max_length,
            add_eos_token=True,
        )
        prompt_encoding = _tokenize_text(
            tokenizer,
            prompt_text,
            max_length=max_length,
            add_eos_token=False,
        )

        labels = list(full_encoding["input_ids"])
        if not train_on_inputs:
            prompt_token_count = min(len(prompt_encoding["input_ids"]), max_length)
            labels[:prompt_token_count] = [IGNORE_INDEX] * prompt_token_count
        has_target_tokens = any(token != IGNORE_INDEX for token in labels)
        if not has_target_tokens:
            skipped_stats["count"] += 1

        return {
            "input_ids": full_encoding["input_ids"],
            "attention_mask": full_encoding["attention_mask"],
            "labels": labels,
            "has_target_tokens": has_target_tokens,
        }

    generate_and_tokenize_prompt.skipped_stats = skipped_stats
    return generate_and_tokenize_prompt


def _get_client_idx(client_id: str) -> int:
    if not client_id.startswith("client_"):
        raise ValueError(f"client_id must look like 'client_1', got: {client_id}")
    return int(client_id.split("_")[-1]) - 1


def _subject_for_client(client_id: str) -> str:
    idx = _get_client_idx(client_id)
    if idx < 0 or idx >= len(MEDMCQA_SUBJECTS):
        raise ValueError(
            f"MedMCQA expects client_1..client_{len(MEDMCQA_SUBJECTS)}, got: {client_id}"
        )
    return MEDMCQA_SUBJECTS[idx]


def _load_medmcqa_examples(client_id: str, split: str, limit: int) -> List[Dict[str, str]]:
    subject = _subject_for_client(client_id)
    file_split = "test" if split in {"test", "validation"} else "train"
    file_path = MEDMCQA_DIR / f"{subject}_{file_split}.json"
    if not file_path.exists():
        raise FileNotFoundError(
            f"MedMCQA file not found: {file_path}. Run src/data/prepare_medmcqa.py first."
        )
    with file_path.open("r", encoding="utf-8") as f:
        examples = json.load(f)
    return examples[:limit]


def _to_mcqa_eval_record(example: Dict, client_id: str | None = None) -> Dict[str, str]:
    """MedMCQA record -> 평가용 dict. answer_letter/options/cop 메타를 보존한다."""
    subject = example.get("subject_name") or (
        _subject_for_client(client_id) if client_id else ""
    )
    return {
        "question": (example.get("instruction") or "").strip(),
        "input": (example.get("input") or "").strip(),
        "reference": example["answer_letter"],
        "answer_letter": example["answer_letter"],
        "cop": example["cop"],
        "options": example["options"],
        "subject_name": example.get("subject_name", ""),
        "task_name": subject,
        "task_type": "mcqa",
        "client_id": client_id,
        "id": example.get("id", ""),
    }


def _load_local_feddpa_examples(
    dataset_name: str,
    client_id: str,
    split: str,
    limit: int,
) -> List[Dict[str, str]]:
    if dataset_name == "medmcqa":
        return _load_medmcqa_examples(client_id=client_id, split=split, limit=limit)

    dataset_dir = LOCAL_DATA_ROOT / dataset_name
    client_idx = _get_client_idx(client_id)

    if split == "train":
        file_path = dataset_dir / "8" / f"local_training_{client_idx}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"FedDPA local training file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as f:
            examples = json.load(f)
        return examples[:limit]

    if split == "test":
        file_path = dataset_dir / "flan_test_200_selected_nstrict_1.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"FedDPA local test file not found: {file_path}")
        task_name = _get_client_task_info(dataset_name, client_id)["task_name"]
        examples = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["task"] != task_name:
                    continue
                examples.append(record)
                if len(examples) >= limit:
                    break
        return examples

    raise ValueError(f"Unsupported split: {split}")


def _load_local_feddpa_hf_dataset(
    dataset_name: str,
    client_id: str,
    split: str,
    limit: int,
) -> HFDataset:
    examples = _load_local_feddpa_examples(
        dataset_name=dataset_name,
        client_id=client_id,
        split=split,
        limit=limit,
    )
    if not examples:
        raise ValueError(f"No examples found for {dataset_name}/{client_id}/{split}")
    return HFDataset.from_list(examples)


def _get_client_task_info(dataset_name: str, client_id: str) -> Dict[str, str]:
    if dataset_name == "medmcqa":
        return {"task_name": _subject_for_client(client_id), "task_type": "mcqa"}

    train_examples = _load_local_feddpa_examples(
        dataset_name=dataset_name,
        client_id=client_id,
        split="train",
        limit=1,
    )
    if not train_examples:
        raise ValueError(f"No training examples found for {dataset_name}/{client_id}")

    example = train_examples[0]
    return {
        "task_name": example["task"],
        "task_type": example["category"],
    }


# =========================================================================
# 3. 데이터 로딩 함수 (Client 학습, General Query, Evaluation)
# =========================================================================

def get_client_dataloader(
    client_id: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    num_samples: int = 1000,
    dataset_name: str = "dataset1",
    max_length: int = 512,
    train_on_inputs: bool = False,
) -> DataLoader:
    return get_task_specific_client_dataloader(
        client_id=client_id,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_samples=num_samples,
        max_length=max_length,
        train_on_inputs=train_on_inputs,
    )


def get_task_specific_client_dataloader(
    client_id: str,
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "dataset1",
    batch_size: int = 4,
    num_samples: int = 300,
    split: str = "train",
    max_length: int = 512,
    train_on_inputs: bool = False,
) -> DataLoader:
    """FedDPA task-specific client loader using local dataset files."""
    local_split = "test" if split in {"test", "validation"} else "train"
    logger.info(
        f"[FedDPA Local] Using {dataset_name}/{client_id} {local_split} data "
        f"from {LOCAL_DATA_ROOT}"
    )
    dataset_obj = _load_local_feddpa_hf_dataset(
        dataset_name=dataset_name,
        client_id=client_id,
        split=local_split,
        limit=num_samples,
    )
    generate_and_tokenize_prompt = build_generate_and_tokenize_prompt(
        tokenizer,
        max_length=max_length,
        train_on_inputs=train_on_inputs,
    )
    tokenized_dataset = dataset_obj.map(generate_and_tokenize_prompt)
    tokenized_dataset = tokenized_dataset.filter(lambda example: example["has_target_tokens"])
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in tokenized_dataset.column_names if col not in {"input_ids", "attention_mask", "labels"}]
    )

    skipped_no_target = generate_and_tokenize_prompt.skipped_stats["count"]
    logger.info(
        "[FedDPA Local] Built %d supervised examples (skipped %d with no target tokens, max_length=%d, train_on_inputs=%s)",
        len(tokenized_dataset),
        skipped_no_target,
        max_length,
        train_on_inputs,
    )
    if len(tokenized_dataset) == 0:
        raise ValueError(
            "No valid supervised training examples remained after prompt/target masking. "
            f"max_length={max_length} may be too small for this dataset."
        )

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
    )


def get_client_eval_dataset(
    client_id: str,
    dataset_name: str = "dataset1",
    num_samples: int = 200,
) -> List[Dict[str, str]]:
    if dataset_name == "medmcqa":
        examples = _load_medmcqa_examples(client_id=client_id, split="test", limit=num_samples)
        logger.info(
            "[MedMCQA] eval %s (%s) %d samples", client_id, _subject_for_client(client_id), len(examples)
        )
        return [_to_mcqa_eval_record(ex, client_id=client_id) for ex in examples]

    task_info = _get_client_task_info(dataset_name, client_id)
    local_examples = _load_local_feddpa_examples(
        dataset_name=dataset_name,
        client_id=client_id,
        split="test",
        limit=num_samples,
    )
    logger.info(
        f"[FedDPA Local] Using eval data for {dataset_name}/{client_id} "
        f"from {LOCAL_DATA_ROOT}"
    )
    return [
        {
            "question": (example.get("instruction") or example.get("inputs") or "").strip(),
            "input": (example.get("input") or "").strip(),
            "reference": (example.get("output") or "").strip(),
            "task_name": example.get("task", task_info["task_name"]),
            "task_type": example.get("category", task_info["task_type"]),
            "client_id": client_id,
        }
        for example in local_examples
    ]


def get_full_test_eval_dataset(
    dataset_name: str = "dataset1",
    num_samples: int | None = None,
) -> List[Dict[str, str]]:
    if dataset_name == "medmcqa":
        file_path = MEDMCQA_DIR / "all_test.json"
        if not file_path.exists():
            raise FileNotFoundError(
                f"MedMCQA full test file not found: {file_path}. Run src/data/prepare_medmcqa.py first."
            )
        with file_path.open("r", encoding="utf-8") as f:
            examples = json.load(f)
        if num_samples is not None:
            examples = examples[:num_samples]
        logger.info("[MedMCQA] full cross-domain test: %d samples", len(examples))
        return [_to_mcqa_eval_record(ex, client_id=None) for ex in examples]

    file_path = LOCAL_DATA_ROOT / dataset_name / "flan_test_200_selected_nstrict_1.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"FedDPA full test file not found: {file_path}")

    examples: List[Dict[str, str]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            examples.append(
                {
                    "question": (record.get("instruction") or record.get("inputs") or "").strip(),
                    "input": (record.get("input") or "").strip(),
                    "reference": (record.get("output") or "").strip(),
                    "task_name": record.get("task", ""),
                    "task_type": record.get("category", ""),
                }
            )
            if num_samples is not None and len(examples) >= num_samples:
                break

    logger.info(
        f"[FedDPA Local] Using full test eval data for {dataset_name} "
        f"from {LOCAL_DATA_ROOT}"
    )
    return examples


# =========================================================================
# 4. 모듈 테스트 (직접 실행 시 정상 작동하는지 확인)
# =========================================================================
if __name__ == "__main__":
    logger.info(">>> 토크나이저 로드 중...")
    tz = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=token)
    tz.pad_token = tz.eos_token


    # 4. dataset1 / dataset2 각각 client_1의 train/eval 샘플 5개 확인
    for dataset_name in ["dataset1", "dataset2"]:
        client_id = "client_1"
        task_info = _get_client_task_info(dataset_name, client_id)

        train_examples = _load_local_feddpa_examples(
            dataset_name=dataset_name,
            client_id=client_id,
            split="train",
            limit=5,
        )
        eval_examples = get_client_eval_dataset(
            client_id=client_id,
            dataset_name=dataset_name,
            num_samples=5,
        )

        logger.info("\n" + "=" * 80)
        logger.info(f"[{dataset_name} / {client_id}] task={task_info['task_name']} | category={task_info['task_type']}")
        logger.info("-" * 80)

        logger.info("Train samples:")
        for idx, example in enumerate(train_examples, start=1):
            logger.info(f"[train {idx}] instruction: {example['instruction'][:140]}...")
            logger.info(f"[train {idx}] output: {example['output'][:140]}...")

        logger.info("-" * 80)
        logger.info("Eval samples:")
        for idx, example in enumerate(eval_examples, start=1):
            logger.info(f"[eval {idx}] question: {example['question'][:140]}...")
            logger.info(f"[eval {idx}] reference: {example['reference'][:140]}...")

    logger.info("=" * 80)
