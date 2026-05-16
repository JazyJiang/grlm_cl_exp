"""
PKM module for GRLM - wraps Tiger's HashingMemory with learnable gate (init=0).
Used to inject PKM as a parallel residual branch alongside Qwen3 MLP.

Design:
- input → HashingMemory(input) → output
- Final contribution = gate * output, where gate is learnable scalar init=0
- At init, gate=0 means PKM contributes nothing (preserves pretrained behavior)
- During training, gate is learned alongside PKM's keys/values
"""

import math
import torch
import torch.nn as nn

# Use Tiger's HashingMemory (copied to routing/pkm/)
from .pkm.memory import HashingMemory
from omegaconf import OmegaConf


def _build_pkm_cfg(input_dim: int, output_dim: int, n_keys: int = 128, heads: int = 4,
                   knn: int = 32, k_dim: int = 256, v_dim: int = -1):
    """Build minimal config dict accepted by HashingMemory."""
    return OmegaConf.create({
        "pkm": {
            "t5_seq2seq": {  # HashingMemory expects nested config with strategy key
                "pk_is_enabled": True,
                "pk_encoder_layers": "",
                "pk_decoder_layers": "0",  # placeholder
                "pk_mem_n_keys": n_keys,
                "pk_mem_heads": heads,
                "pk_mem_knn": knn,
                "pk_mem_share_values": False,
                "pk_mem_k_dim": k_dim,
                "pk_mem_v_dim": v_dim,
                "pk_swilu_projection": True,
                "pk_value_fixed_lr": 0.001,
                "pk_mem_gated": False,  # we add OUR own gate, not the inner pk gate
                "pk_peer_variant": False,
                "pk_topk": 8,
                "pk_mem_dim": None,
            }
        }
    })


class PKMResidual(nn.Module):
    """PKM module that adds gated contribution to its input.

    Args:
        hidden_size: Input/output dim.
        n_keys: Number of keys per side (key space = n_keys^2).
        heads: Number of query heads.
        knn: Top-k keys per side.
        gate_init: Initial value for the learnable gate scalar (0.0 = pkm disabled at init).
        strategy: "t5_seq2seq" (config namespace inside HashingMemory).
    """

    def __init__(self, hidden_size: int, n_keys: int = 128, heads: int = 4,
                 knn: int = 32, gate_init: float = 0.0, gate_type: str = "scalar"):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_type = gate_type

        # Build PKM module
        # k_dim must satisfy: heads * k_dim should be ≥ hidden_size for projection
        # For hidden_size=1024, heads=4 → k_dim=256 → query_proj outputs 1024
        k_dim = max(256, hidden_size // heads)
        # Each PKM instance gets its own values (mem_share_values=False)
        # so multi-layer configs don't share weights
        self.pkm = HashingMemory(
            input_dim=hidden_size, output_dim=hidden_size,
            mem_n_keys=int(n_keys), mem_heads=int(heads), mem_knn=int(knn),
            mem_k_dim=int(k_dim), mem_v_dim=-1,
            mem_share_values=False, mem_gated=False,
            swilu_projection=True, peer_variant=False,
            value_fixed_lr=None,  # use trainer's default lr
        )

        # Gate: scalar (legacy) or input-adaptive (linear/mlp)
        if gate_type == "scalar":
            # Learnable gate scalar (single param shared across tokens)
            self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
            self.gate_proj = None
        elif gate_type == "linear":
            # Per-token sigmoid gate: g_i = sigmoid(W * h_i + b)
            # bias_init makes sigmoid(b) ≈ gate_init at start
            import math as _m
            bias_init = 0.0 if gate_init <= 0 or gate_init >= 1 else _m.log(gate_init / (1 - gate_init))
            if gate_init >= 1.0: bias_init = 5.0  # sigmoid(5) ≈ 0.993
            self.gate = None
            self.gate_proj = nn.Linear(hidden_size, 1, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, bias_init)
        elif gate_type == "mlp":
            # 2-layer MLP gate
            import math as _m
            bias_init = 0.0 if gate_init <= 0 or gate_init >= 1 else _m.log(gate_init / (1 - gate_init))
            if gate_init >= 1.0: bias_init = 5.0
            self.gate = None
            self.gate_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1, bias=True),
            )
            nn.init.zeros_(self.gate_proj[2].weight)
            nn.init.constant_(self.gate_proj[2].bias, bias_init)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gated PKM contribution.
        Args:
            x: [B, T, D] hidden states.
        Returns:
            [B, T, D] gated PKM output (to be added as residual).
        """
        # HashingMemory.forward takes [N, D] or [B, T, D]
        pkm_out = self.pkm(x)
        if self.gate is not None:
            return self.gate * pkm_out
        # Input-adaptive gate
        gate_logits = self.gate_proj(x)            # [B, T, 1]
        gate = torch.sigmoid(gate_logits)
        return gate * pkm_out


def inject_pkm_into_qwen(model: nn.Module, layer_indices: list,
                          position: str = "parallel",
                          n_keys: int = 128, heads: int = 4, knn: int = 32,
                          gate_init: float = 0.0, gate_type: str = "scalar") -> nn.ModuleDict:
    """Inject PKM modules into specified Qwen3 layers.

    Args:
        model: Qwen3ForCausalLM instance.
        layer_indices: List of layer indices to add PKM to.
        position: "parallel" (PKM(x) + MLP(x)) or "serial" (PKM(MLP(x)) + MLP(x)).
        n_keys, heads, knn: PKM hyperparams.
        gate_init: Init value for gate (0 = disabled at init).

    Returns:
        ModuleDict of injected PKM modules (also attached to model._pkm_modules).

    The injection works by:
        1. Creating a PKMResidual for each target layer.
        2. Monkey-patching the layer's `forward` to add PKM contribution.
        3. Storing PKM modules in model._pkm_modules so they're saved/loaded.
    """
    hidden_size = model.config.hidden_size
    pkm_dict = nn.ModuleDict()

    for layer_idx in layer_indices:
        if layer_idx >= len(model.model.layers):
            print(f"[pkm] WARN: layer {layer_idx} >= num_layers {len(model.model.layers)}, skip")
            continue
        layer = model.model.layers[layer_idx]
        pkm = PKMResidual(hidden_size=hidden_size, n_keys=n_keys, heads=heads,
                          knn=knn, gate_init=gate_init, gate_type=gate_type)
        pkm = pkm.to(next(layer.parameters()).device).to(next(layer.parameters()).dtype)
        pkm_dict[f"layer_{layer_idx}"] = pkm

        # Monkey-patch layer forward
        _patch_layer_forward(layer, pkm, position=position)
        print(f"[pkm] Injected PKM at layer {layer_idx} (position={position}, gate_type={gate_type}, "
              f"n_keys={n_keys}, gate_init={gate_init})")

    model._pkm_modules = pkm_dict
    return pkm_dict


def _patch_layer_forward(layer: nn.Module, pkm: PKMResidual, position: str):
    """Monkey-patch layer.forward to inject PKM contribution.

    Qwen3DecoderLayer.forward signature (transformers ≥4.57):
        (hidden_states, attention_mask=None, position_ids=None, past_key_values=None, ...)
        returns (hidden_states,) or (hidden_states, present_key_value, ...)

    We add PKM contribution to hidden_states based on `position`.
    """
    original_forward = layer.forward

    def patched_forward(*args, **kwargs):
        # Capture input hidden_states
        if len(args) > 0:
            input_hidden = args[0]
        else:
            input_hidden = kwargs.get("hidden_states")

        # Run original layer
        outputs = original_forward(*args, **kwargs)

        # outputs may be a tensor or a tuple (hidden, *rest)
        if isinstance(outputs, tuple):
            output_hidden = outputs[0]
            rest = outputs[1:]
        else:
            output_hidden = outputs
            rest = ()

        # Compute PKM contribution
        if position == "parallel":
            # PKM applied to layer INPUT, added to layer output
            pkm_input = input_hidden
        elif position == "serial":
            # PKM applied to layer OUTPUT (post-MLP), added to layer output
            pkm_input = output_hidden
        else:
            raise ValueError(f"Unknown position: {position}")

        pkm_contrib = pkm(pkm_input)
        new_hidden = output_hidden + pkm_contrib

        if rest:
            return (new_hidden,) + rest
        return new_hidden

    layer.forward = patched_forward
