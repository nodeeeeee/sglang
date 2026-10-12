"""Compact zero-padded FC2 rows for the TRT-LLM MXFP4 valid-dimension ABI."""

from functools import lru_cache
from inspect import signature


@lru_cache(maxsize=1)
def supports_valid_hidden_size() -> bool:
    from flashinfer import trtllm_fp4_block_scale_moe

    return "valid_hidden_size" in signature(trtllm_fp4_block_scale_moe).parameters


def compact_output_weights(weight, scale, bias, valid_hidden_size):
    """Remove complete padded row groups after the existing weight shuffle.

    The caller supplies the checkpoint's hidden size and guarantees zero padding.
    Weights and bias are shuffled within 128-row groups. Scale storage consists
    of consecutive 128-row groups, despite its apparent [expert, row, K/32] shape.
    """
    width = (valid_hidden_size + 127) // 128 * 128
    if width >= weight.shape[1]:
        return weight, scale, bias, None

    weight = weight[:, :width, :].contiguous()
    experts, _, scale_columns = scale.shape
    scale = (
        scale.reshape(experts, -1)[:, : width * scale_columns]
        .contiguous()
        .reshape(experts, width, scale_columns)
    )
    if bias is not None:
        bias = bias[:, :width].contiguous()
    return weight, scale, bias, valid_hidden_size
