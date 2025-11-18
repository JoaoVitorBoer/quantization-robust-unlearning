"""
Run 4-bit quantization-aware fine-tuning (QAT) for unlearning steps.

The script injects fake-quantization noise (torchao) during training so the
resulting checkpoint remains a standard, non-quantized HF model that is more
robust to later PTQ. Quantized weights are never saved.
"""

import argparse
import pathlib
import sys
from typing import Iterable, Mapping

import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import AutoModelForCausalLM

# Allow imports from the existing baselines package
BASELINE_PATH = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(BASELINE_PATH))

from baselines.dataset import DefaultDataset
from baselines.utils import load_model_and_tokenizer

try:
    from torchao.quantization import (
        Int8DynamicActivationInt4WeightConfig,
        quantize_,
    )
    from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig
except Exception as exc:  # pragma: no cover - import guard for helpful failure
    raise ImportError(
        "torchao is required for QAT. Install with `pip install torchao` "
        "inside your training environment."
    ) from exc


def prepare_qat_model(
    model: AutoModelForCausalLM,
    group_size: int = 32,
    quantize_embeddings: bool = False,
) -> Int8DynamicActivationInt4WeightConfig:
    """
    Insert fake-quant modules so training sees 4-bit noise.

    Args:
        model: HF causal LM model to wrap.
        group_size: Group size for weight-only int4 quantization.
        quantize_embeddings: If True, also apply weight-only fake quantization to
            embedding layers.
    Returns:
        The PTQ base config used for QAT (needed only for reference/logging).
    """
    base_config = Int8DynamicActivationInt4WeightConfig(group_size=group_size)

    # Prepare all Linear layers (activations: int8 dyn per-token, weights: int4 per-group).
    quantize_(model, QATConfig(base_config, step="prepare"))

    if quantize_embeddings:
        weight_cfg = IntxFakeQuantizeConfig(torch.int4, group_size=group_size)
        quantize_(
            model,
            QATConfig(weight_config=weight_cfg, step="prepare"),
            filter_fn=lambda module, _: isinstance(module, nn.Embedding),
        )

    return base_config


def export_fp_checkpoint(
    qat_model: AutoModelForCausalLM,
    reference_model_dir: str,
    save_dir: str,
    tokenizer=None,
) -> None:
    """
    Strip fake-quant wrappers by loading a clean float model and transplanting
    the trained weights from the QAT model.
    """
    float_model = AutoModelForCausalLM.from_pretrained(
        reference_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    base_state: Mapping[str, torch.Tensor] = float_model.state_dict()
    qat_state: Mapping[str, torch.Tensor] = qat_model.state_dict()

    merged_state = dict(base_state)
    for name, tensor in qat_state.items():
        if name in merged_state:
            merged_state[name] = tensor.to(merged_state[name].dtype)

    missing: Iterable[str] = set(merged_state.keys()) - set(qat_state.keys())
    if missing:
        print(f"[qat-export] Keeping {len(missing)} untouched fp weights (not present in QAT state).")

    float_model.load_state_dict(merged_state, strict=False)
    float_model.save_pretrained(save_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)


def run_qat_finetune(args: argparse.Namespace):
    model, tokenizer = load_model_and_tokenizer(
        args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
    )

    dataset: Dataset = DefaultDataset(
        args.data_file,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    prepare_qat_model(
        model,
        group_size=args.group_size,
        quantize_embeddings=args.qat_embeddings,
    )

    training_args = transformers.TrainingArguments(
        output_dir=args.temp_qat_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=True,
        report_to="none",
    )

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    export_fp_checkpoint(
        qat_model=model,
        reference_model_dir=args.model_dir if args.reference_model_dir is None else args.reference_model_dir,
        save_dir=args.out_dir,
        tokenizer=tokenizer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4-bit QAT fine-tuning for unlearning (torchao).")
    parser.add_argument("--model_dir", type=str, required=True, help="HF folder of the target model.")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Optional tokenizer path.")
    parser.add_argument("--data_file", type=str, required=True, help="Forget set used for fine-tuning.")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to save the float checkpoint after QAT.")
    parser.add_argument(
        "--reference_model_dir",
        type=str,
        default=None,
        help="Optional float checkpoint to clone architecture/state dict layout from when stripping QAT. "
             "Defaults to `--model_dir`.",
    )
    parser.add_argument("--max_len", type=int, default=4096, help="Max sequence length for tokenization.")
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--group_size", type=int, default=32, help="Group size for int4 weight quantization.")
    parser.add_argument(
        "--qat_embeddings",
        action="store_true",
        help="Also apply weight-only fake quantization to embedding layers.",
    )
    parser.add_argument(
        "--temp_qat_dir",
        type=str,
        default="qat_tmp",
        help="Where to place intermediate trainer checkpoints (kept in QAT form).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume Trainer from the checkpoint in `--temp_qat_dir` if present.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_qat_finetune(parse_args())
