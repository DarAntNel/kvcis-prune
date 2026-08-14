"""
KVCIS-3 Step 2: Collect training data for the TWO-PROBE setup.

Design (two probes, deferred eviction):

  Probe A -- the ORIGINAL KVCIS regression probe, recreated unchanged: predicts
             cumulative attention importance from a token's layer-N activation.
             At runtime it assigns storage tiers exactly like original KVCIS
             (importance >= threshold -> fp16, else int8). Nothing is dropped
             at insertion; every token is stored.

  Probe B -- a NEW binary "evictability" probe trained alongside: predicts
             whether a token will stop being consulted LATER in generation.
             Its action is deferred: tokens younger than a grace window are
             never touched, and fp16-tier tokens are NEVER evicted regardless
             of what probe B says. Only aged, int8-tier, flagged tokens are
             dropped as the cache grows.

Label construction. At each generation step t the attention row sums to 1, so
a token's fair share is 1/L_t; we use the share ratio r_t(j) = a_t(j) * L_t
(1.0 = fair share), comparable across steps. For each prompt token:

  importance(j)  cumulative attention received, max-normalized per sequence
                 (probe A target -- identical to the original repo's step2)
  evictable(j)   1 if max_{t >= W} r_t(j) < --evict-max-ratio: after its first
                 W generation steps the token never again reaches even that
                 fraction of a fair share. It may have been consulted EARLY
                 (local syntax duty) -- that is fine and expected; what matters
                 is that it goes quiet afterwards. This is strictly more
                 permissive than the old "never consulted at any step" rule,
                 and is exactly the "the (position 5) adds nothing later" case.

"Quiet after W" is only as trustworthy as the horizon it was measured over:
prompts whose generation ends before W + 3 steps are skipped, and a longer
--generation-steps gives better labels.

Outputs (in --output-dir):
  activations.npy     [n_tokens, hidden]  layer-N activations (both probes' input)
  importance.npy      [n_tokens]          probe A target (original KVCIS)
  evictable.npy       [n_tokens]          probe B target (0 keep / 1 evictable)
  ratio_mean.npy      [n_tokens]          mean share ratio, whole horizon
  ratio_max.npy       [n_tokens]          max share ratio, whole horizon
  ratio_max_late.npy  [n_tokens]          max share ratio at steps >= W
  metadata.json, config.json

Example:
  python step2_collect_data.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --extraction-layer 10 --n-prompts 200 --generation-steps 30 \
      --evict-window 10 --output-dir ../data
"""

import argparse
import json
from pathlib import Path

# import torch BEFORE numpy (numpy-first OpenMP conflict segfaults on Windows)
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset


def load_diverse_prompts(n_prompts: int = 500):
    """Same prompt mix as the parent repo's step2 (wikitext + alpaca + code)."""
    prompts = []
    try:
        print("Loading general text (wikitext-103)...")
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                               split="train", streaming=True)
        count, buf = 0, ""
        for item in dataset:
            text = item.get("text", "")
            if not text.strip():
                continue
            buf += " " + text
            if len(buf) > 400:
                prompts.append(buf.strip()[:500])
                buf = ""
                count += 1
                if count >= n_prompts // 2:
                    break
    except Exception as e:
        print(f"Could not load wikitext general text: {e}")
    try:
        print("Loading Alpaca...")
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        for item in list(dataset)[:n_prompts // 4]:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            if instruction:
                prompt = f"Instruction: {instruction}"
                if input_text:
                    prompt += f"\nInput: {input_text}"
                prompts.append(prompt)
    except Exception as e:
        print(f"Could not load Alpaca: {e}")
    code_prompts = [
        "def fibonacci(n):",
        "class DatabaseConnection:",
        "import pandas as pd\n\ndef load_data(path):",
        "async def fetch_url(url):",
        "# Binary search implementation\ndef binary_search(arr, target):",
    ]
    prompts.extend(code_prompts * (n_prompts // 20))
    if len(prompts) < n_prompts:
        simple = [
            "The weather today is",
            "In the year 2024,",
            "The most important thing about",
            "Scientists have discovered that",
            "The history of artificial intelligence",
        ]
        while len(prompts) < n_prompts:
            prompts.extend(simple)
    return prompts[:n_prompts]


class TwoProbeCollector:
    def __init__(self, model, tokenizer, extraction_layer, generation_steps,
                 evict_window, evict_max_ratio):
        self.model = model
        self.tokenizer = tokenizer
        self.extraction_layer = extraction_layer
        self.generation_steps = generation_steps
        self.evict_window = evict_window
        self.evict_max_ratio = evict_max_ratio
        self.device = next(model.parameters()).device
        self._activations = None
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._activations = hidden[0].detach().float().cpu().numpy()

    def setup_hook(self):
        layer = self.model.model.layers[self.extraction_layer]
        self._hook_handle = layer.register_forward_hook(self._hook_fn)

    def remove_hook(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def collect_single_prompt(self, prompt, prompt_idx, max_prompt_tokens=96):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=max_prompt_tokens).to(self.device)
        input_ids = inputs.input_ids
        prompt_length = input_ids.shape[1]
        if prompt_length < 5:
            return None

        # Activations at the extraction layer: both probes' INPUT, available at
        # insertion time (before any attention from future tokens exists).
        self.setup_hook()
        with torch.no_grad():
            self.model(input_ids, use_cache=False)
        self.remove_hook()
        activations = self._activations  # [prompt_length, hidden]

        # Generate; record each step's attention row over the prompt tokens.
        ratios = []                      # per step: a_t(j) * L_t  [prompt_length]
        attention_received = np.zeros(prompt_length)
        current_ids = input_ids.clone()
        for _ in range(self.generation_steps):
            with torch.no_grad():
                outputs = self.model(current_ids, output_attentions=True,
                                     return_dict=True)
            attn = outputs.attentions[self.extraction_layer]
            last_attn = attn[0, :, -1, :prompt_length].mean(dim=0).float().cpu().numpy()
            L_t = current_ids.shape[1]
            ratios.append(last_attn * L_t)
            attention_received += last_attn
            next_token = outputs.logits[0, -1, :].argmax()
            current_ids = torch.cat(
                [current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        # "quiet after W" needs a few observed steps past the window
        if len(ratios) < self.evict_window + 3:
            return None

        R = np.stack(ratios)                       # [T, prompt_length]
        r_mean = R.mean(axis=0)
        r_max = R.max(axis=0)
        r_max_late = R[self.evict_window:].max(axis=0)

        # Probe A target: identical to the original repo's step2.
        importance = attention_received / max(attention_received.max(), 1e-12)
        # Probe B target: consulted early is fine; quiet AFTER the window.
        evictable = (r_max_late < self.evict_max_ratio).astype(np.int64)

        token_ids = input_ids[0].tolist()
        meta = [{"token_id": token_ids[p],
                 "token_str": self.tokenizer.decode([token_ids[p]]),
                 "position": p, "prompt_idx": prompt_idx,
                 "importance": float(importance[p]),
                 "evictable": int(evictable[p]),
                 "r_mean": float(r_mean[p]), "r_max": float(r_max[p]),
                 "r_max_late": float(r_max_late[p])}
                for p in range(prompt_length)]
        return activations, importance, evictable, r_mean, r_max, r_max_late, meta


def main():
    parser = argparse.ArgumentParser(description="KVCIS-3 Step 2: two-probe training data")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--extraction-layer", type=int, default=10)
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--max-prompt-tokens", type=int, default=96)
    parser.add_argument("--generation-steps", type=int, default=30,
                        help="Label horizon; must exceed --evict-window by >= 3")
    parser.add_argument("--evict-window", type=int, default=10,
                        help="Grace window W: evictability is judged on steps >= W only")
    parser.add_argument("--evict-max-ratio", type=float, default=0.5,
                        help="Evictable if the max share ratio at steps >= W stays below this")
    parser.add_argument("--output-dir", type=str, default="../data")
    args = parser.parse_args()

    if args.generation_steps < args.evict_window + 3:
        raise SystemExit("--generation-steps must be at least --evict-window + 3")

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

    prompts = load_diverse_prompts(args.n_prompts)
    print(f"Loaded {len(prompts)} prompts")

    collector = TwoProbeCollector(model, tokenizer, args.extraction_layer,
                                  args.generation_steps,
                                  args.evict_window, args.evict_max_ratio)

    acts, imps, evs, rmeans, rmaxs, rlates, metadata = [], [], [], [], [], [], []
    n_skipped = 0
    for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        try:
            out = collector.collect_single_prompt(prompt, i, args.max_prompt_tokens)
        except Exception as e:
            print(f"\nError on prompt {i}: {e}")
            continue
        if out is None:
            n_skipped += 1
            continue
        a, im, ev, rm, rx, rl, md = out
        acts.append(a); imps.append(im); evs.append(ev)
        rmeans.append(rm); rmaxs.append(rx); rlates.append(rl)
        metadata.extend(md)

    activations = np.concatenate(acts, axis=0)
    importance = np.concatenate(imps)
    evictable = np.concatenate(evs)
    r_mean = np.concatenate(rmeans)
    r_max = np.concatenate(rmaxs)
    r_max_late = np.concatenate(rlates)
    print(f"\nCollected {len(importance)} token samples "
          f"({n_skipped} prompts skipped: horizon ended before window + 3)")

    np.save(output_dir / "activations.npy", activations)
    np.save(output_dir / "importance.npy", importance)
    np.save(output_dir / "evictable.npy", evictable)
    np.save(output_dir / "ratio_mean.npy", r_mean)
    np.save(output_dir / "ratio_max.npy", r_max)
    np.save(output_dir / "ratio_max_late.npy", r_max_late)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    n_ev = int(evictable.sum())
    config = {
        "model": args.model, "extraction_layer": args.extraction_layer,
        "n_prompts": args.n_prompts, "max_prompt_tokens": args.max_prompt_tokens,
        "generation_steps": args.generation_steps,
        "evict_window": args.evict_window,
        "evict_max_ratio": args.evict_max_ratio,
        "total_tokens": int(len(importance)),
        "n_evictable": n_ev,
        "evictable_fraction": n_ev / max(1, len(evictable)),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nProbe A target (importance): "
          f"min {importance.min():.4f}  mean {importance.mean():.4f}  "
          f"max {importance.max():.4f}")
    bos = importance[[m["position"] == 0 for m in metadata]]
    if len(bos):
        print(f"  position-0 mean {bos.mean():.4f} (attention sink; should be ~1.0)")
    print(f"Probe B target (evictable): {n_ev}/{len(evictable)} "
          f"({100 * n_ev / max(1, len(evictable)):.1f}%) quiet after step "
          f"{args.evict_window}")
    if n_ev < max(20, 0.01 * len(evictable)) or n_ev > 0.99 * len(evictable):
        print("  WARNING: evictable class is extreme -- adjust --evict-max-ratio "
              "or --evict-window")
    print(f"\nData saved to {output_dir}")
    print("\nStep 2 complete - two-probe training data collected")


if __name__ == "__main__":
    main()
