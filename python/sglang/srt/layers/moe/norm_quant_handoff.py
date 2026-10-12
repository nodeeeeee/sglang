"""Consume-once norm quantization output, bounded by one GPT-OSS FFN call."""
_entry = None


def clear():
    global _entry
    _entry = None


def publish(x, quant, scale):
    global _entry
    _entry = (x, quant, scale)


def take(x, padded_width):
    global _entry
    entry = _entry
    _entry = None
    if entry is None:
        return None
    produced, quant, scale = entry
    if (produced.data_ptr() != x.data_ptr() or produced.shape != x.shape
        or produced.dtype != x.dtype or produced.stride() != x.stride()
        or quant.shape != (x.shape[0], padded_width)):
        return None
    return quant, scale
