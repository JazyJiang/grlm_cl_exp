"""Training entry-point for GRLM with PKM injection.

Usage:
    python -m routing.train_with_pkm \
        --model_name_or_path models/Qwen3-0.6B \
        --dataset grlm_indomain_videogames_cl_D0 \
        --pkm_layer_indices 14 \
        --pkm_position parallel \
        --pkm_n_keys 128 \
        --pkm_gate_init 0.0 \
        --pkm_lr 1e-3 \
        --output_dir checkpoints/vg_pkm_C2_parallel_mid \
        [... other LlamaFactory SFT args ...]

Adds PKM-residual modules to specified Qwen3 layers (output += gate * PKM(input)).
At gate=0 (init), behaves identically to baseline. Gate is learnable; PKM keys/values
also trainable.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List

import torch
from transformers import HfArgumentParser

# Add LlamaFactory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_DIR = os.path.join(WORK_DIR, "LlamaFactory")
if os.path.isdir(os.path.join(LLAMA_DIR, "src")):
    sys.path.insert(0, os.path.join(LLAMA_DIR, "src"))

from llamafactory.hparams import (
    DataArguments, FinetuningArguments, GeneratingArguments,
    ModelArguments, TrainingArguments,
)
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.sft.workflow import run_sft

from routing.pkm_module import inject_pkm_into_qwen


@dataclass
class PKMArguments:
    """Args for PKM injection."""

    pkm_layer_indices: str = field(
        default="14",
        metadata={"help": "Comma-separated list of layer indices to add PKM. e.g. '14' or '12,18'."},
    )
    pkm_position: str = field(
        default="parallel",
        metadata={"help": "PKM placement: 'parallel' (PKM(x) + MLP(x)) or 'serial' (PKM(MLP(x)) + MLP(x))."},
    )
    pkm_n_keys: int = field(default=128)
    pkm_heads: int = field(default=4)
    pkm_knn: int = field(default=32)
    pkm_gate_init: float = field(default=0.0)
    pkm_freeze_gate: bool = field(default=False)
    pkm_gate_type: str = field(default="scalar", metadata={"help": "scalar | linear | mlp"})
    pkm_weight_decay: float = field(default=-1.0, metadata={"help": "if >=0, override weight decay for PKM params"})
    pkm_lr: float = field(
        default=1e-3,
        metadata={"help": "Separate learning rate for PKM params (typically larger than base lr)."},
    )


def main():
    parser = HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments,
        FinetuningArguments, GeneratingArguments, PKMArguments,
    ))

    (model_args, data_args, training_args, finetuning_args,
     generating_args, pkm_args) = parser.parse_args_into_dataclasses()

    # Parse layer indices
    layer_indices = [int(x.strip()) for x in pkm_args.pkm_layer_indices.split(",") if x.strip()]
    print(f"[pkm] layer_indices={layer_indices}, position={pkm_args.pkm_position}, "
          f"n_keys={pkm_args.pkm_n_keys}, gate_init={pkm_args.pkm_gate_init}, "
          f"pkm_lr={pkm_args.pkm_lr}, base_lr={training_args.learning_rate}")

    # ---- Monkey-patch load_model to inject PKM after loading ----
    from llamafactory.model import loader as _loader_module
    _original_load_model = _loader_module.load_model

    def _load_model_with_pkm(tokenizer, model_args, finetuning_args, is_trainable=False, **kwargs):
        model = _original_load_model(tokenizer, model_args, finetuning_args, is_trainable, **kwargs)
        if is_trainable:
            inject_pkm_into_qwen(
                model,
                layer_indices=layer_indices,
                position=pkm_args.pkm_position,
                n_keys=pkm_args.pkm_n_keys,
                heads=pkm_args.pkm_heads,
                knn=pkm_args.pkm_knn,
                gate_init=pkm_args.pkm_gate_init, gate_type=pkm_args.pkm_gate_type,
            )
            if pkm_args.pkm_freeze_gate:
                for n, p in model.named_parameters():
                    if n.endswith(".gate") and "_pkm_modules" in n:
                        p.requires_grad = False
                        with __import__("torch").no_grad():
                            p.fill_(pkm_args.pkm_gate_init)
                        print(f"[pkm] Froze {n} = {p.item():.3f}")
        return model

    _loader_module.load_model = _load_model_with_pkm
    import llamafactory.train.sft.workflow as _workflow
    _workflow.load_model = _load_model_with_pkm

    # ---- Monkey-patch trainer.create_optimizer to add separate PKM param group ----
    _original_create_optimizer = CustomSeq2SeqTrainer.create_optimizer

    def _create_optimizer_with_pkm(self):
        if self.optimizer is not None:
            return self.optimizer

        # Build param groups: PKM params get pkm_lr, rest gets base lr
        opt_model = self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)

        pkm_param_names = set()
        if hasattr(opt_model, "_pkm_modules"):
            for name, _ in opt_model._pkm_modules.named_parameters():
                pkm_param_names.add(f"_pkm_modules.{name}")

        pkm_decay = []
        pkm_no_decay = []
        base_decay = []
        base_no_decay = []
        for name, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            in_pkm = name.startswith("_pkm_modules.") or any(name.endswith(s) for s in pkm_param_names)
            in_decay = name in decay_parameters
            if in_pkm:
                if in_decay:
                    pkm_decay.append(p)
                else:
                    pkm_no_decay.append(p)
            else:
                if in_decay:
                    base_decay.append(p)
                else:
                    base_no_decay.append(p)

        from transformers.utils import logging as _hf_logging
        logger = _hf_logging.get_logger("[pkm-optim]")
        print(f"[pkm-optim] base params: {sum(p.numel() for p in base_decay+base_no_decay):,}")
        print(f"[pkm-optim] pkm params: {sum(p.numel() for p in pkm_decay+pkm_no_decay):,}")

        # Get optimizer class & default kwargs
        optim_cls, optim_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
        wd = self.args.weight_decay

        param_groups = [
            {"params": base_decay, "weight_decay": wd, "lr": self.args.learning_rate},
            {"params": base_no_decay, "weight_decay": 0.0, "lr": self.args.learning_rate},
        ]
        if pkm_decay or pkm_no_decay:
            pkm_wd = pkm_args.pkm_weight_decay if pkm_args.pkm_weight_decay >= 0 else wd
            print(f"[pkm-optim] pkm_lr={pkm_args.pkm_lr}, pkm_wd={pkm_wd}, base_wd={wd}")
            param_groups += [
                {"params": pkm_decay, "weight_decay": pkm_wd, "lr": pkm_args.pkm_lr},
                {"params": pkm_no_decay, "weight_decay": 0.0, "lr": pkm_args.pkm_lr},
            ]

        # Filter out empty groups
        param_groups = [g for g in param_groups if g["params"]]

        self.optimizer = optim_cls(param_groups, **{k: v for k, v in optim_kwargs.items() if k != "lr"})
        return self.optimizer

    CustomSeq2SeqTrainer.create_optimizer = _create_optimizer_with_pkm

    # ---- Run standard LlamaFactory SFT ----
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args)


if __name__ == "__main__":
    main()
