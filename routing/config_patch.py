"""Patch Qwen3 config to use per-layer sliding window attention."""

from transformers import PretrainedConfig


def patch_qwen3_config_for_routing(
    config: PretrainedConfig,
    sliding_window_size: int = 512,
    full_layer: int | None = None,
) -> None:
    """Modify Qwen3 config so that one layer uses full attention and all others use sliding window.

    Args:
        config: Qwen3Config instance (modified in-place).
        sliding_window_size: Number of tokens in the sliding window.
        full_layer: Layer index that keeps full attention.
                    Default: middle layer (num_hidden_layers // 2).
    """
    n_layers = config.num_hidden_layers
    if full_layer is None:
        full_layer = n_layers // 2

    config.layer_types = [
        "full_attention" if i == full_layer else "sliding_attention"
        for i in range(n_layers)
    ]
    config.use_sliding_window = True
    config.sliding_window = sliding_window_size
