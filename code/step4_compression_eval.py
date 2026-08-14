"""
KVCIS-3 Step 4: Two-probe compression evaluation (deferred eviction).

Probe A (original KVCIS, recreated) runs first and works exactly as in the
parent repo: importance >= --high-threshold -> fp16, everything else -> int8.
EVERY token is stored at insertion -- nothing is dropped up front.

Probe B then marks int8-tier tokens whose attention it predicts will go quiet
after the grace window. Once the cache has grown past that window, flagged
tokens are physically removed. Hard rules, enforced regardless of probe B:

  * fp16-tier tokens are NEVER evicted (high-attention tokens always remain)
  * the first --force-sink tokens are never evicted (attention sinks)
  * the most recent --grace-recent tokens are never evicted (too young --
    a follow-up prompt may still need them, e.g. "over" in "The cow jumped
    over ..." right after insertion)
  * only evict when P(evictable) >= --evict-margin

Correctness of removal: kept keys retain the RoPE phase of their ORIGINAL
positions, so continuations are scored at the targets' TRUE absolute
position_ids (mechanism verified in the parent repo: keep-all == baseline).

Memory note (honest accounting): deferred eviction stores everything first, so
its PEAK bytes equal plain KVCIS; the table reports the STEADY-STATE bytes
after eviction has caught up. Both are in the JSON.

Methods:
  Baseline       full bf16 cache
  Uniform-INT8   every token int8 (quantization reference)
  KVCIS          probe A only -- the original scheme, nothing evicted
  KVCIS+Evict    probe A tiers + probe B deferred eviction of int8 tokens

Example:
  python step4_compression_eval.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --extraction-layer 10 --probe-path ../data/probe2 --n-texts 20
"""

import argparse
import json
import math
from pathlib import Path

# import torch BEFORE numpy (numpy-first OpenMP conflict segfaults on Windows)
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from kv_utils import (BytesModel, cache_to_list, list_to_cache,
                      fake_quant_positions_, eval_loss_at_positions,
                      load_eval_texts)


class TwoProbeCompressor:
    """Probe A (importance regression) + probe B (evictability logit)."""

    def __init__(self, model, probe_path, extraction_layer=10,
                 high_threshold=0.9, evict_margin=0.5,
                 force_sink=4, grace_recent=16):
        self.model = model
        self.extraction_layer = extraction_layer
        self.high_threshold = high_threshold
        self.evict_margin = evict_margin
        self.force_sink = force_sink
        self.grace_recent = grace_recent
        self.device = next(model.parameters()).device

        probe_dir = Path(probe_path)
        self.wa = torch.from_numpy(
            np.load(probe_dir / "regression" / "weights.npy")).float().to(self.device)
        self.ba = float(np.load(probe_dir / "regression" / "bias.npy")[0])
        self.wb = torch.from_numpy(
            np.load(probe_dir / "evict" / "weights.npy")).float().to(self.device)
        self.bb = float(np.load(probe_dir / "evict" / "bias.npy")[0])

        self._activations = None
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._activations = hidden.detach()

    def setup_hook(self):
        layer = self.model.model.layers[self.extraction_layer]
        self._hook_handle = layer.register_forward_hook(self._hook_fn)

    def remove_hook(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def score(self, activations):
        """Probe A importance in [0,1] and probe B P(evictable), both [seq]."""
        act = activations[0].float()                       # [seq, hidden]
        importance = (act @ self.wa + self.ba).clamp(0, 1)
        p_evict = torch.sigmoid(act @ self.wb + self.bb)
        return importance.cpu(), p_evict.cpu()

    def decide(self, importance, p_evict):
        """Return (is_fp16 mask, evict mask) with all hard rules applied."""
        seq_len = importance.shape[0]
        is_fp16 = importance >= self.high_threshold        # original KVCIS tiering
        evict = (p_evict >= self.evict_margin) & ~is_fp16  # NEVER evict fp16
        ns = min(self.force_sink, seq_len)
        evict[:ns] = False                                 # sinks stay
        ng = min(self.grace_recent, seq_len)
        if ng:
            evict[seq_len - ng:] = False                   # grace window: too young
        return is_fp16, evict

    def build_cache(self, past_key_values, is_fp16, evict):
        """int8-quantize the non-fp16 tier, then drop the evicted positions."""
        kv_list = [(k.clone(), v.clone()) for k, v in cache_to_list(past_key_values)]
        int8_kept = (~is_fp16 & ~evict).nonzero(as_tuple=False).squeeze(-1)
        fake_quant_positions_(kv_list, int8_kept.tolist(), self.device)
        if evict.any():
            keep = (~evict).nonzero(as_tuple=False).squeeze(-1).to(kv_list[0][0].device)
            kv_list = [(k.index_select(2, keep).contiguous(),
                        v.index_select(2, keep).contiguous())
                       for k, v in kv_list]
        return list_to_cache(kv_list)


def main():
    parser = argparse.ArgumentParser(description="KVCIS-3 Step 4: two-probe eval")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--probe-path", type=str, required=True,
                        help="dir containing regression/ (probe A) and evict/ (probe B)")
    parser.add_argument("--extraction-layer", type=int, default=10)
    parser.add_argument("--n-texts", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-ratio", type=float, default=0.3)
    parser.add_argument("--high-threshold", type=float, default=0.9,
                        help="Probe A fp16 threshold (original KVCIS default)")
    parser.add_argument("--evict-margin", type=float, default=0.5,
                        help="Only evict when P(evictable) >= this")
    parser.add_argument("--force-sink", type=int, default=4)
    parser.add_argument("--grace-recent", type=int, default=16,
                        help="The most recent N tokens are never evicted")
    parser.add_argument("--output-dir", type=str, default="../results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager")
    model.eval()

    comp = TwoProbeCompressor(model, args.probe_path, args.extraction_layer,
                              high_threshold=args.high_threshold,
                              evict_margin=args.evict_margin,
                              force_sink=args.force_sink,
                              grace_recent=args.grace_recent)
    bm = BytesModel(model.config)

    methods = ["Baseline", "Uniform-INT8", "KVCIS", "KVCIS+Evict"]
    agg = {m: {"bytes": 0, "loss": 0.0, "tok": 0} for m in methods}
    peak_evict_bytes = 0
    counts = {"fp16": 0, "int8_kept": 0, "evicted": 0, "total": 0}

    texts = load_eval_texts(args.n_texts, args.max_length)
    print(f"Loaded {len(texts)} texts")

    for text in tqdm(texts, desc="Evaluating"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=args.max_length).to("cuda")
        seq_len = inputs.input_ids.shape[1]
        if seq_len < max(20, args.max_length * 0.5):
            continue
        prompt_len = int(seq_len * (1 - args.eval_ratio))
        prompt_ids = inputs.input_ids[:, :prompt_len]
        target_ids = inputs.input_ids[:, prompt_len:]
        if target_ids.shape[1] < 5:
            continue

        with torch.no_grad():
            comp.setup_hook()
            out = model(prompt_ids, use_cache=True, return_dict=True)
            comp.remove_hook()
            importance, p_evict = comp.score(comp._activations)
            is_fp16, evict = comp.decide(importance, p_evict)
            prefill_kv = [(k.detach().clone(), v.detach().clone())
                          for k, v in cache_to_list(out.past_key_values)]

            n_hi = int(is_fp16.sum())
            n_ev = int(evict.sum())
            n_lo = prompt_len - n_hi - n_ev
            counts["fp16"] += n_hi
            counts["int8_kept"] += n_lo
            counts["evicted"] += n_ev
            counts["total"] += prompt_len

            # Baseline
            cache = list_to_cache([(k, v) for k, v in prefill_kv])
            loss, ntok = eval_loss_at_positions(model, target_ids, cache, prompt_len)
            agg["Baseline"]["bytes"] += prompt_len * bm.fp16_per_tok
            agg["Baseline"]["loss"] += loss; agg["Baseline"]["tok"] += ntok

            # Uniform-INT8
            kvq = [(k.clone(), v.clone()) for k, v in prefill_kv]
            fake_quant_positions_(kvq, list(range(prompt_len)), comp.device)
            loss, ntok = eval_loss_at_positions(model, target_ids,
                                                list_to_cache(kvq), prompt_len)
            agg["Uniform-INT8"]["bytes"] += bm.mixed_bytes(0, prompt_len)
            agg["Uniform-INT8"]["loss"] += loss; agg["Uniform-INT8"]["tok"] += ntok

            # KVCIS: probe A only, nothing evicted (the original scheme)
            no_evict = torch.zeros_like(evict)
            cache = comp.build_cache(list_to_cache(prefill_kv), is_fp16, no_evict)
            loss, ntok = eval_loss_at_positions(model, target_ids, cache, prompt_len)
            agg["KVCIS"]["bytes"] += bm.mixed_bytes(n_hi, prompt_len - n_hi)
            agg["KVCIS"]["loss"] += loss; agg["KVCIS"]["tok"] += ntok

            # KVCIS+Evict: probe A tiers + probe B deferred eviction
            cache = comp.build_cache(list_to_cache(prefill_kv), is_fp16, evict)
            loss, ntok = eval_loss_at_positions(model, target_ids, cache, prompt_len)
            agg["KVCIS+Evict"]["bytes"] += bm.mixed_bytes(n_hi, n_lo)   # steady state
            peak_evict_bytes += bm.mixed_bytes(n_hi, prompt_len - n_hi)  # at insertion
            agg["KVCIS+Evict"]["loss"] += loss; agg["KVCIS+Evict"]["tok"] += ntok

    if agg["Baseline"]["tok"] == 0:
        raise SystemExit("No valid eval texts -- lower --max-length or raise --n-texts")

    tot = max(1, counts["total"])
    print(f"\nToken fates over {tot} prompt tokens: "
          f"fp16 {100 * counts['fp16'] / tot:.1f}% | "
          f"int8 kept {100 * counts['int8_kept'] / tot:.1f}% | "
          f"evicted later {100 * counts['evicted'] / tot:.1f}%  "
          f"(thr {args.high_threshold}, margin {args.evict_margin}, "
          f"sink {args.force_sink}, grace {args.grace_recent})")

    base_bytes = agg["Baseline"]["bytes"]
    base_ppl = math.exp(agg["Baseline"]["loss"] / agg["Baseline"]["tok"])
    results = []
    print(f"\n{'Method':<14}{'KV MB':>9}{'Mem Red.':>10}{'PPL':>11}{'PPL D':>10}")
    print("-" * 54)
    for m in methods:
        mb = agg[m]["bytes"] / 1e6
        red = (1 - agg[m]["bytes"] / base_bytes) * 100
        ppl = math.exp(agg[m]["loss"] / agg[m]["tok"])
        d = (ppl / base_ppl - 1) * 100
        print(f"{m:<14}{mb:>9.1f}{red:>9.1f}%{ppl:>11.4f}{d:>+9.2f}%")
        results.append({"method": m, "kv_bytes": agg[m]["bytes"],
                        "memory_reduction_pct": red, "perplexity": ppl,
                        "ppl_delta_pct": d, "total_tokens": agg[m]["tok"]})
    print(f"\nNOTE: KVCIS+Evict bytes are steady-state (post-eviction); its peak "
          f"at insertion is {peak_evict_bytes / 1e6:.1f} MB (= the KVCIS row).")

    out = {
        "model": args.model, "extraction_layer": args.extraction_layer,
        "n_texts": args.n_texts, "max_length": args.max_length,
        "high_threshold": args.high_threshold, "evict_margin": args.evict_margin,
        "force_sink": args.force_sink, "grace_recent": args.grace_recent,
        "token_fates": dict(counts),
        "kvcis_evict_peak_bytes": peak_evict_bytes,
        "results": results,
    }
    out_path = output_dir / "kvcis2p_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\nStep 4 complete - two-probe deferred eviction evaluated")


if __name__ == "__main__":
    main()
