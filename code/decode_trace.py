"""
KVCIS-3 decode trace: token-by-token cache growth and eviction during REAL
generation, extended to GENERATED tokens.

For each example prompt we run one greedy generation. Every token -- prompt and
generated alike -- is scored the moment it enters the cache:

  probe A (importance)  -> tier: fp16 (>= --high-threshold) or int8
  probe B (evictability) -> flagged if P(evictable) >= --evict-margin

Eviction fires during decode with the deferred semantics: a flagged int8-tier
token is removed at the first step where its AGE (steps since insertion)
exceeds --grace-recent. Hard rules as in step4: fp16 never evicted, the first
--force-sink absolute positions never evicted.

TRACE MODE IS VIRTUAL: generation itself runs on the full uncompressed cache,
and we track what each policy's cache WOULD hold at every step, byte-accurately.
This keeps one shared token sequence per example so the methods are comparable
token-for-token (actually evicting during decode would change the generated
text and break the side-by-side). Quality impact of these policies is measured
separately by step4.

Output: ../results/decode_trace.json
  examples[] -> {name, prompt, generated, tokens[] (str, origin, tier, p_evict,
                 evict_step or null), series {method -> [MB per step]},
                 events[] (step, positions evicted)}

Example:
  python decode_trace.py --probe-path ../data/probe2 --gen-steps 60
"""

import argparse
import json
from pathlib import Path

# import torch BEFORE numpy (numpy-first OpenMP conflict segfaults on Windows)
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_utils import BytesModel

EXAMPLES = [
    ("cow", "The cow jumped over the moon. Where did the cow jump?"),
    ("eiffel", "The Eiffel Tower was completed in 1889 and stands 330 metres tall. "
               "How tall is the Eiffel Tower?"),
    ("code", "# Python\ndef binary_search(arr, target):"),
]


def stable_sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, 0, None))),
                    np.exp(np.clip(z, None, 0)) / (1.0 + np.exp(np.clip(z, None, 0))))


def main():
    parser = argparse.ArgumentParser(description="KVCIS-3 decode-time trace")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--probe-path", type=str, default="../data/probe2")
    parser.add_argument("--extraction-layer", type=int, default=10)
    parser.add_argument("--gen-steps", type=int, default=60)
    parser.add_argument("--high-threshold", type=float, default=0.9)
    parser.add_argument("--evict-margin", type=float, default=0.7)
    parser.add_argument("--force-sink", type=int, default=4)
    parser.add_argument("--grace-recent", type=int, default=16,
                        help="A token may be evicted once its age exceeds this")
    parser.add_argument("--output", type=str, default="../results/decode_trace.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager")
    model.eval()

    probe_dir = Path(args.probe_path)
    wa = np.load(probe_dir / "regression" / "weights.npy")
    ba = float(np.load(probe_dir / "regression" / "bias.npy")[0])
    wb = np.load(probe_dir / "evict" / "weights.npy")
    bb = float(np.load(probe_dir / "evict" / "bias.npy")[0])
    bm = BytesModel(model.config)
    fp16_b, int8_b = bm.fp16_per_tok, bm.int8_per_tok + bm.meta_per_tok

    captured = {}

    def hook_fn(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured["acts"] = hidden[0].detach().float().cpu().numpy()  # [q_len, H]

    layer = model.model.layers[args.extraction_layer]

    all_examples = []
    for name, prompt in EXAMPLES:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        L0 = inputs.input_ids.shape[1]

        # tokens[i]: dict per token in cache order; inserted_at = generation step
        # (0 for all prompt tokens, t for the t-th generated token)
        tokens = []

        def add_tokens(acts, ids, origin, step):
            imp = np.clip(acts @ wa + ba, 0, 1)
            p_ev = stable_sigmoid(acts @ wb + bb)
            for i in range(acts.shape[0]):
                pos = len(tokens)
                tier = "fp16" if imp[i] >= args.high_threshold else "int8"
                flagged = (tier == "int8" and p_ev[i] >= args.evict_margin
                           and pos >= args.force_sink)
                tokens.append({
                    "str": tokenizer.decode([ids[i]]),
                    "origin": origin, "inserted_at": step,
                    "importance": round(float(imp[i]), 4),
                    "p_evict": round(float(p_ev[i]), 4),
                    "tier": tier, "flagged": bool(flagged),
                    "evict_step": None,
                })

        # ---- prefill ----
        handle = layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            out = model(inputs.input_ids, use_cache=True, return_dict=True)
        add_tokens(captured["acts"], inputs.input_ids[0].tolist(), "prompt", 0)
        cache = out.past_key_values
        next_id = out.logits[0, -1].argmax().view(1, 1)

        series = {"Baseline": [], "Uniform-INT8": [], "KVCIS": [], "KVCIS+Evict": []}
        events = []

        def record_step(step):
            # fire evictions: flagged, int8, not yet evicted, aged past grace
            fired = []
            for pos, tk in enumerate(tokens):
                if (tk["flagged"] and tk["evict_step"] is None
                        and step - tk["inserted_at"] > args.grace_recent):
                    tk["evict_step"] = step
                    fired.append(pos)
            if fired:
                events.append({"step": step, "evicted_positions": fired})
            L = len(tokens)
            n_hi = sum(1 for t in tokens if t["tier"] == "fp16")
            n_ev = sum(1 for t in tokens if t["evict_step"] is not None)
            series["Baseline"].append(L * fp16_b / 1e6)
            series["Uniform-INT8"].append(L * int8_b / 1e6)
            series["KVCIS"].append((n_hi * fp16_b + (L - n_hi) * int8_b) / 1e6)
            series["KVCIS+Evict"].append(
                (n_hi * fp16_b + (L - n_hi - n_ev) * int8_b) / 1e6)

        record_step(0)

        # ---- decode (generation on the FULL cache; policies tracked virtually) ----
        gen_ids = []
        with torch.no_grad():
            for t in range(1, args.gen_steps + 1):
                out = model(next_id, past_key_values=cache, use_cache=True,
                            return_dict=True)
                cache = out.past_key_values
                add_tokens(captured["acts"], [next_id.item()], "generated", t)
                gen_ids.append(next_id.item())
                record_step(t)
                next_id = out.logits[0, -1].argmax().view(1, 1)
                if gen_ids[-1] == tokenizer.eos_token_id:
                    break
        handle.remove()

        n_ev = sum(1 for t in tokens if t["evict_step"] is not None)
        n_hi = sum(1 for t in tokens if t["tier"] == "fp16")
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"[{name}] {L0} prompt + {len(gen_ids)} generated tokens | "
              f"fp16 {n_hi} | evicted {n_ev} | "
              f"final: base {series['Baseline'][-1]:.3f} MB -> "
              f"evict {series['KVCIS+Evict'][-1]:.3f} MB")
        all_examples.append({
            "name": name, "prompt": prompt, "generated": gen_text,
            "prompt_len": L0, "tokens": tokens, "series": series,
            "events": events,
        })

    out = {
        "trace_mode": "virtual (one shared generation per example; policies "
                      "tracked byte-accurately without altering the text)",
        "model": args.model,
        "config": {k: getattr(args, k) for k in
                   ("extraction_layer", "gen_steps", "high_threshold",
                    "evict_margin", "force_sink", "grace_recent")},
        "bytes_per_token": {"fp16": fp16_b, "int8_incl_meta": int8_b},
        "examples": all_examples,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nTrace saved to {out_path}")


if __name__ == "__main__":
    main()
