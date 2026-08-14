"""
Shared plumbing for KVCIS-3 (three-tier probe: fp16 / int8 / discard).

Self-contained copies of the cache/quantization/eval utilities from the parent
`kvcis` repo, so this folder runs standalone. Semantics are identical to the
originals (kvcis/code/step4_compression_eval.py, methods.py, decode_dynamics.py,
cache_telemetry.py); only the packaging differs.
"""

from typing import List, Tuple

# import torch BEFORE numpy (numpy-first OpenMP conflict segfaults on Windows)
import torch
from transformers.cache_utils import DynamicCache
from datasets import load_dataset


# --------------------------------------------------------------------------- #
#  DynamicCache <-> list plumbing (transformers 4.x and 5.x layouts)            #
# --------------------------------------------------------------------------- #

def cache_to_list(cache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert a cache object to a list of (K, V) tuples (references)."""
    if cache is None:
        return []
    if isinstance(cache, (list, tuple)):
        return [(item[0], item[1]) for item in cache]
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    if hasattr(cache, "to_legacy_cache"):
        return [(k, v) for k, v in cache.to_legacy_cache()]
    # transformers >= 5.0: DynamicCache stores per-layer objects in .layers
    if hasattr(cache, "layers") and cache.layers is not None:
        result = [(l.keys, l.values) for l in cache.layers
                  if getattr(l, "keys", None) is not None]
        if result:
            return result
    raise ValueError(f"Cannot convert cache of type {type(cache)}")


def list_to_cache(kv_list) -> DynamicCache:
    cache = DynamicCache()
    for i, (k, v) in enumerate(kv_list):
        cache.update(k, v, layer_idx=i)
    return cache


# --------------------------------------------------------------------------- #
#  Quantization                                                                 #
# --------------------------------------------------------------------------- #

def fake_quant_positions_(kv_list, positions, device, bits: int = 8):
    """Round-trip int8 fake-quant of the given token positions, IN PLACE, with
    a per-(position, layer, tensor) min-max grid (vectorized)."""
    if len(positions) == 0:
        return
    levels = float(2 ** bits - 1)
    idx = torch.as_tensor(positions, dtype=torch.long, device=device)
    for k, v in kv_list:
        for t in (k, v):
            sub = t[:, :, idx, :].float()                  # [1, H, P, D]
            tmin = sub.amin(dim=(1, 3), keepdim=True)
            rng = (sub.amax(dim=(1, 3), keepdim=True) - tmin).clamp_min(1e-8)
            q = ((sub - tmin) / rng * levels).round()
            t[:, :, idx, :] = (q / levels * rng + tmin).to(t.dtype)


# --------------------------------------------------------------------------- #
#  Byte accounting (same storage model as kvcis/code/cache_telemetry.py)        #
# --------------------------------------------------------------------------- #

class BytesModel:
    """Whole-cache bytes for ONE token: fp16, int8 payload, int8 quant metadata
    (a fp32 (scale, min) pair per (token, layer, tensor) = 8 B)."""

    INT8_META_BYTES = 8

    def __init__(self, config):
        self.n_layers = config.num_hidden_layers
        n_kv = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads)
        elems = n_kv * head_dim
        self.fp16_per_tok = self.n_layers * 2 * elems * 2
        self.int8_per_tok = self.n_layers * 2 * elems * 1
        self.meta_per_tok = self.n_layers * 2 * self.INT8_META_BYTES

    def mixed_bytes(self, n_fp16: int, n_int8: int, n_discard: int = 0) -> int:
        return (n_fp16 * self.fp16_per_tok
                + n_int8 * (self.int8_per_tok + self.meta_per_tok))


# --------------------------------------------------------------------------- #
#  Evaluation                                                                   #
# --------------------------------------------------------------------------- #

def eval_loss_at_positions(model, target_ids, cache, prompt_len):
    """Sum CE loss of the continuation, scored at the targets' TRUE absolute
    positions. Required after discarding tokens: kept keys retain the RoPE
    phase of their ORIGINAL positions, so target queries must be rotated at
    prompt_len..prompt_len+T-1 regardless of the (shrunken) cache length.
    (Verified in the parent repo: keep-all reproduces baseline loss exactly.)"""
    T = target_ids.shape[1]
    pos = torch.arange(prompt_len, prompt_len + T,
                       device=target_ids.device).unsqueeze(0)
    out = model(target_ids, past_key_values=cache, position_ids=pos,
                return_dict=True)
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = target_ids[:, 1:].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                   shift_labels.view(-1))
    return loss.item(), shift_labels.numel()


def load_eval_texts(n_texts: int = 100, max_length: int = 512):
    """WikiText eval texts (same construction as the parent repo)."""
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                               split="test")
    except Exception:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                               split="test")
    texts, current = [], ""
    for item in dataset:
        text = item.get("text", "")
        if not text.strip():
            continue
        current += " " + text
        if len(current) > max_length * 4:
            texts.append(current.strip())
            current = ""
            if len(texts) >= n_texts:
                break
    return texts[:n_texts]
