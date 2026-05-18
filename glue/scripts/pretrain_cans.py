"""
Pretrain a RoBERTa masked-language model with CANS-based whitening.

This version supports caching the expensive dataset preprocessing phase:
raw text -> tokenized examples -> fixed-length grouped blocks.

Typical two-stage usage:

1) Preprocess once and save the grouped/tokenized dataset:

python -m glue.scripts.pretrain_cans \
  --dataset-name allenai/c4 \
  --dataset-config en \
  --dataset-split 'train[:0.1%]' \
  --processed-dataset-dir /path/to/c4_roberta_grouped_512 \
  --preprocess-only \
  --trust-remote-code

2) Train from the already processed dataset:

python -m glue.scripts.pretrain_cans \
  --processed-dataset-dir /path/to/c4_roberta_grouped_512 \
  --output-dir ./cans_pretrained \
  --num-train-epochs 1 \
  --iterations 2 \
  --momentum 0.1
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import torch

try:
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )
except ImportError as e:
    raise ImportError(
        "pretrain_cans.py requires the 'transformers' and 'datasets' libraries. "
        "Install them before running this script."
    ) from e

from models.layers.cans_whitening import WhiteningCANS2d


LOGGER = logging.getLogger(__name__)


def replace_layer_norm_with_cans(
    model,
    iterations: int = 2,
    momentum: float = 0.1,
    affine: bool = True,
):
    """Recursively replace all torch.nn.LayerNorm modules with WhiteningCANS2d."""
    import torch.nn as nn

    for name, module in model.named_children():
        replace_layer_norm_with_cans(module, iterations=iterations, momentum=momentum, affine=affine)

        if isinstance(module, nn.LayerNorm):
            num_features = module.normalized_shape[0]
            cans_layer = WhiteningCANS2d(
                num_features=num_features,
                iterations=iterations,
                momentum=momentum,
                affine=affine,
                track_running_stats=True,
                use_running_stats_train=True,
                use_only_running_stats_eval=True,
            )

            if affine:
                with torch.no_grad():
                    cans_layer.weight.data.copy_(module.weight.data)
                    cans_layer.bias.data.copy_(module.bias.data)

            setattr(model, name, cans_layer)


class WhiteningMetricsCallback(TrainerCallback):
    """Log coarse whitening statistics from all WhiteningCANS2d layers."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return

        sigma_norms = []
        identity_devs = []
        h_norms = []
        wh_norms = []

        for module in model.modules():
            if isinstance(module, WhiteningCANS2d):
                sigma = module.running_covariance.detach()
                eye = torch.eye(sigma.size(0), device=sigma.device, dtype=sigma.dtype)
                sigma_norms.append(torch.norm(sigma, p="fro").item())
                identity_devs.append(torch.norm(sigma - eye, p="fro").item())
                if getattr(module, "running_H", None) is not None:
                    h_norms.append(torch.norm(module.running_H.detach(), p="fro").item())
                if getattr(module, "running_whitening", None) is not None:
                    wh_norms.append(torch.norm(module.running_whitening.detach(), p="fro").item())

        if logs is not None and sigma_norms:
            logs["whitening/sigma_fro"] = sum(sigma_norms) / len(sigma_norms)
            logs["whitening/identity_dev"] = sum(identity_devs) / len(identity_devs)
            if h_norms:
                logs["whitening/H_fro"] = sum(h_norms) / len(h_norms)
            if wh_norms:
                logs["whitening/inv_sqrt_fro"] = sum(wh_norms) / len(wh_norms)


def _normalise_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in {"none", "null"}:
        return None
    return value


def get_or_create_lm_dataset(args, tokenizer):
    """Load a cached grouped/tokenized dataset, or create and save one.

    The cached dataset is saved after grouping, i.e. after examples have fixed
    length ``args.block_size``.  MLM labels are intentionally not cached because
    DataCollatorForLanguageModeling dynamically samples masks during training.
    """
    processed_path = Path(args.processed_dataset_dir) if args.processed_dataset_dir else None

    if processed_path is not None and processed_path.exists() and not args.force_preprocess:
        LOGGER.info("Loading already tokenized+grouped dataset from %s", processed_path)
        return load_from_disk(str(processed_path))

    if processed_path is not None and processed_path.exists() and args.force_preprocess:
        LOGGER.info("--force-preprocess is set; removing existing dataset at %s", processed_path)
        shutil.rmtree(processed_path)

    dataset_name = _normalise_optional_string(args.dataset_name)
    dataset_config = _normalise_optional_string(args.dataset_config)

    if dataset_name is None:
        raise ValueError(
            "No --dataset-name was provided and --processed-dataset-dir does not point to an existing dataset."
        )

    LOGGER.info(
        "Loading raw dataset name=%s config=%s split=%s",
        dataset_name,
        dataset_config,
        args.dataset_split,
    )

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=args.dataset_split,
        trust_remote_code=args.trust_remote_code,
    )
    LOGGER.info("Loaded raw dataset: %s", dataset)

    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"Text column '{args.text_column}' not found. Available columns: {dataset.column_names}"
        )

    def tokenize_function(examples):
        return tokenizer(
            examples[args.text_column],
            return_special_tokens_mask=True,
            truncation=False,
        )

    LOGGER.info("Tokenizing raw dataset")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=args.preprocessing_batch_size,
        num_proc=args.preprocessing_num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing raw text",
    )

    block_size = args.block_size
    if tokenizer.model_max_length and tokenizer.model_max_length < int(1e20):
        block_size = min(block_size, tokenizer.model_max_length)

    if block_size <= 0:
        raise ValueError(f"--block-size must be positive, got {block_size}")

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        return {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }

    LOGGER.info("Grouping tokenized examples into blocks of %d", block_size)
    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        batch_size=args.preprocessing_batch_size,
        num_proc=args.preprocessing_num_proc,
        desc=f"Grouping tokenized text into blocks of {block_size}",
    )

    if processed_path is not None:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving tokenized+grouped dataset to %s", processed_path)
        lm_dataset.save_to_disk(str(processed_path))

        metadata = {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_split": args.dataset_split,
            "text_column": args.text_column,
            "tokenizer_name_or_path": args.tokenizer_name_or_path,
            "block_size": block_size,
            "columns": list(lm_dataset.column_names),
        }
        with open(processed_path / "preprocess_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    return lm_dataset


def split_train_eval_dataset(args, lm_dataset) -> Tuple[Dataset, Optional[Dataset]]:
    """Create train/eval datasets from a Dataset or use existing DatasetDict splits."""
    if isinstance(lm_dataset, DatasetDict):
        train_dataset = lm_dataset.get("train")
        eval_dataset = (
            lm_dataset.get("validation")
            or lm_dataset.get("eval")
            or lm_dataset.get("test")
        )
        if train_dataset is None:
            raise ValueError(
                "Loaded DatasetDict does not contain a 'train' split. "
                f"Available splits: {list(lm_dataset.keys())}"
            )
        return train_dataset, eval_dataset

    dataset_len = len(lm_dataset)
    if dataset_len == 0:
        raise ValueError("The tokenized+grouped dataset is empty. Check --dataset-split and --block-size.")

    if dataset_len == 1 or args.validation_fraction <= 0 or args.max_validation_samples <= 0:
        LOGGER.warning("No evaluation split will be created because the processed dataset is too small or validation is disabled.")
        return lm_dataset, None

    val_size = int(args.validation_fraction * dataset_len)
    val_size = max(1, val_size)
    val_size = min(args.max_validation_samples, val_size, dataset_len - 1)

    shuffled = lm_dataset.shuffle(seed=args.shuffle_seed)
    eval_dataset = shuffled.select(range(val_size))
    train_dataset = shuffled.select(range(val_size, dataset_len))

    LOGGER.info(
        "Created train/eval split from processed dataset: train=%d eval=%d",
        len(train_dataset),
        len(eval_dataset),
    )
    return train_dataset, eval_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain RoBERTa with CANS whitening")

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="allenai/c4",
        help="HF Hub dataset name. Ignored if --processed-dataset-dir exists and --force-preprocess is not set.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="en",
        help="Dataset config/subset. Use 'none' or an empty string if the dataset has no config.",
    )
    parser.add_argument("--dataset-split", type=str, default="train[:0.1%]", help="Raw dataset split to preprocess.")
    parser.add_argument("--text-column", type=str, default="text", help="Raw text column name.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of dataset loading scripts from the HF Hub.",
    )

    parser.add_argument(
        "--processed-dataset-dir",
        type=str,
        default=None,
        help=(
            "Path to tokenized+grouped HF dataset saved with save_to_disk(). "
            "If it exists, preprocessing is skipped and this dataset is loaded directly. "
            "If it does not exist, preprocessing is run and the result is saved there."
        ),
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Ignore an existing --processed-dataset-dir and rebuild it from raw text.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Run tokenization/grouping/save_to_disk and exit before model allocation/training.",
    )
    parser.add_argument("--block-size", type=int, default=512, help="Grouped sequence length.")
    parser.add_argument(
        "--preprocessing-num-proc",
        type=int,
        default=8,
        help="Number of processes for dataset.map() during tokenization/grouping.",
    )
    parser.add_argument(
        "--preprocessing-batch-size",
        type=int,
        default=1000,
        help="Batch size for batched dataset.map() during preprocessing.",
    )

    parser.add_argument("--model-name-or-path", type=str, default="roberta-base", help="Model checkpoint/name.")
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None, help="Tokenizer checkpoint/name. Defaults to model name.")
    parser.add_argument("--output-dir", type=str, default="./cans_pretrained", help="Directory where checkpoints are saved.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum optimizer steps. Overrides epochs if > 0.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--logging-steps", type=int, default=500, help="Logging interval.")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Evaluation interval.")
    parser.add_argument("--save-steps", type=int, default=2000, help="Checkpoint interval.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--mlm-probability", type=float, default=0.15, help="MLM masking probability.")

    parser.add_argument("--iterations", type=int, default=2, help="Number of CANS iterations for whitening.")
    parser.add_argument("--momentum", type=float, default=0.1, help="EMA momentum for whitening statistics.")

    parser.add_argument("--validation-fraction", type=float, default=0.01, help="Fraction of processed dataset used for eval.")
    parser.add_argument("--max-validation-samples", type=int, default=10000, help="Cap on eval examples.")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed used for train/eval split.")

    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training in Trainer.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training in Trainer.")
    parser.add_argument("--report-to", type=str, default="none", help="Trainer report_to value, e.g. 'wandb' or 'none'.")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    lm_dataset = get_or_create_lm_dataset(args, tokenizer)

    if args.preprocess_only:
        LOGGER.info("Preprocessing complete; exiting because --preprocess-only was set.")
        return

    train_dataset, eval_dataset = split_train_eval_dataset(args, lm_dataset)

    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    replace_layer_norm_with_cans(
        model,
        iterations=args.iterations,
        momentum=args.momentum,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    report_to = [] if args.report_to.lower() == "none" else [x.strip() for x in args.report_to.split(",") if x.strip()]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        prediction_loss_only=True,
        remove_unused_columns=False,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[WhiteningMetricsCallback()],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    LOGGER.info("Training complete. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
