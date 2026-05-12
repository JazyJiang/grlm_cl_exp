"""Auxiliary prediction head for intermediate-layer supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxPredictionHead(nn.Module):
    """Predicts target tokens from an intermediate layer's hidden state."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss from intermediate hidden states.

        Args:
            hidden_states: [B, T, D] from the full-attention layer.
            labels: [B, T] target token ids (-100 for ignored positions).

        Returns:
            Scalar loss.
        """
        shift_logits = self.proj(hidden_states)[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )


def enable_aux_head(model: nn.Module, full_layer_idx: int) -> AuxPredictionHead:
    """Attach an aux prediction head after the specified decoder layer.

    The head's projection shares weights with lm_head (no extra parameters).
    After each forward pass, model._aux_hidden holds the captured hidden states.

    Args:
        model: A Qwen3ForCausalLM (or similar) model.
        full_layer_idx: Index of the full-attention layer to hook.

    Returns:
        The AuxPredictionHead module.
    """
    head = AuxPredictionHead(model.config.hidden_size, model.config.vocab_size)
    head.proj.weight = model.lm_head.weight
    model.add_module("_aux_head", head)
    model._aux_hidden = None

    target_layer = model.model.layers[full_layer_idx]

    def _capture_hook(module, args, output):
        model._aux_hidden = output

    target_layer.register_forward_hook(_capture_hook)
    return head
