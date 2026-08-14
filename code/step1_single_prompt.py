"""
KVCIS-3 Step 1: Single Prompt Test (two-probe setup).

Validates, on one prompt, everything step2/step3/step4 depend on:

1. Run a prompt through the model
2. Extract activations at the extraction layer (both probes' input,
   available at insertion time)
3. Record the PER-STEP attention trajectory each prompt token receives
   during generation, as share ratios r_t(j) = a_t(j) * L_t
4. Show the trajectory-derived evictability label (quiet after the grace
   window W) next to each token
5. If trained probes exist (--probe-path): probe A's importance -> fp16/int8
   tier (original KVCIS behavior) and probe B's P(evictable) -> deferred
   eviction decision, with the hard rules applied (fp16 never evicted,
   sinks protected, recent tokens still in grace)

This is a sanity check before full data collection, not an eval.

Example:
  python step1_single_prompt.py --model Qwen/Qwen2.5-1.5B-Instruct --extraction-layer 10
"""

import argparse
from pathlib import Path

# import torch BEFORE numpy (numpy-first OpenMP conflict segfaults on Windows)
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_utils import BytesModel


def main():
    parser = argparse.ArgumentParser(description="KVCIS-3 Step 1: Single Prompt Test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--extraction-layer", type=int, default=10)
    parser.add_argument("--prompt", type=str,
                        default="The cow jumped over the moon. Where did the cow jump?")
    parser.add_argument("--generation-steps", type=int, default=16,
                        help="Label horizon; must exceed --evict-window by >= 3")
    parser.add_argument("--evict-window", type=int, default=6,
                        help="Grace window W (scaled down for a short demo prompt; "
                             "step2 defaults to 10)")
    parser.add_argument("--evict-max-ratio", type=float, default=0.5)
    parser.add_argument("--high-threshold", type=float, default=0.9,
                        help="Probe A fp16 threshold (original KVCIS)")
    parser.add_argument("--evict-margin", type=float, default=0.5)
    parser.add_argument("--force-sink", type=int, default=4)
    parser.add_argument("--grace-recent", type=int, default=4,
                        help="Scaled down for a short demo prompt; step4 defaults to 16")
    parser.add_argument("--probe-path", type=str, default="../data/probe2",
                        help="Dir with regression/ (probe A) and evict/ (probe B)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",  # need eager for attention extraction
    )
    model.eval()

    # --- activation extraction at the probes' layer --------------------------- #
    activations = {}

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        activations["layer"] = hidden[0].detach().float().cpu().numpy()

    layer = model.model.layers[args.extraction_layer]
    handle = layer.register_forward_hook(hook_fn)

    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    prompt_length = inputs.input_ids.shape[1]

    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {prompt_length}")
    print(f"Extraction layer: {args.extraction_layer}")
    print(f"Horizon {args.generation_steps} steps | evict window {args.evict_window} "
          f"| evict-max-ratio {args.evict_max_ratio}")

    with torch.no_grad():
        outputs = model(inputs.input_ids, output_attentions=True, return_dict=True)
    handle.remove()

    acts = activations["layer"]  # [prompt_length, hidden]
    print(f"\nActivations shape: {acts.shape}")
    print(f"  Expected: [{prompt_length}, {model.config.hidden_size}]")
    attn = outputs.attentions[args.extraction_layer]
    print(f"Attention shape: {tuple(attn.shape)}")
    print(f"  Expected: [1, num_heads, {prompt_length}, {prompt_length}]")

    # --- per-step share-ratio trajectory --------------------------------------- #
    print("\n--- Generation Test (recording share-ratio trajectory) ---")
    ratios = []
    current_ids = inputs.input_ids.clone()
    for step in range(args.generation_steps):
        with torch.no_grad():
            out = model(current_ids, output_attentions=True, return_dict=True)
        a = out.attentions[args.extraction_layer]
        last_attn = a[0, :, -1, :prompt_length].mean(dim=0).float().cpu().numpy()
        r_t = last_attn * current_ids.shape[1]
        ratios.append(r_t)
        next_token = out.logits[0, -1, :].argmax()
        current_ids = torch.cat(
            [current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        print(f"  Step {step + 1:>2}: '{tokenizer.decode(next_token)}' | "
              f"share ratio over prompt: min={r_t.min():.2f}, max={r_t.max():.2f}")
        if next_token.item() == tokenizer.eos_token_id:
            print("  (EOS)")
            break

    print(f"\nFull output: {tokenizer.decode(current_ids[0], skip_special_tokens=True)}")

    if len(ratios) < args.evict_window + 3:
        raise SystemExit("Horizon too short: need >= evict-window + 3 recorded steps")

    R = np.stack(ratios)                    # [T, prompt_length]
    r_mean = R.mean(axis=0)
    r_max = R.max(axis=0)
    r_max_late = R[args.evict_window:].max(axis=0)
    evict_label = (r_max_late < args.evict_max_ratio).astype(int)

    # --- probes (if trained) ---------------------------------------------------- #
    probe_dir = Path(args.probe_path)
    imp = p_ev = None
    pa, pb = probe_dir / "regression", probe_dir / "evict"
    if (pa / "weights.npy").exists() and (pb / "weights.npy").exists():
        wa = np.load(pa / "weights.npy"); ba = np.load(pa / "bias.npy")[0]
        wb = np.load(pb / "weights.npy"); bb = np.load(pb / "bias.npy")[0]
        if wa.shape[0] == acts.shape[1]:
            imp = np.clip(acts @ wa + ba, 0, 1)
            z = acts @ wb + bb
            p_ev = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, 0, None))),
                            np.exp(np.clip(z, None, 0)) / (1.0 + np.exp(np.clip(z, None, 0))))
            print(f"\nLoaded probe A (regression) and probe B (evict) from {probe_dir}")
        else:
            print(f"\nProbe dim {wa.shape[0]} != model dim {acts.shape[1]} -- "
                  f"skipping probe predictions (retrain step3)")
    else:
        print(f"\nNo trained probes at {probe_dir} -- showing trajectory labels only "
              f"(run step2 + step3 to train them)")

    # --- decisions with the hard rules ------------------------------------------ #
    if imp is not None:
        is_fp16 = imp >= args.high_threshold
        evict = (p_ev >= args.evict_margin) & ~is_fp16      # NEVER evict fp16
        evict[:min(args.force_sink, prompt_length)] = False  # sinks stay
        ng = min(args.grace_recent, prompt_length)
        if ng:
            evict[prompt_length - ng:] = False               # still in grace
    else:
        is_fp16 = evict = None

    # --- per-token table ---------------------------------------------------------- #
    print("\n--- Token fates ---")
    header = f"  {'pos':>3} {'token':<12}{'r_mean':>7}{'r_late':>7}{'traj':>6}"
    if imp is not None:
        header += f"{'imp(A)':>8}{'P_ev(B)':>9}  fate"
    print(header)
    for p in range(prompt_length):
        tok = tokenizer.decode([inputs.input_ids[0, p].item()]).replace("\n", "\\n")
        row = (f"  {p:>3} {tok:<12}{r_mean[p]:>7.2f}{r_max_late[p]:>7.2f}"
               f"{('evict' if evict_label[p] else 'keep'):>6}")
        if imp is not None:
            if is_fp16[p]:
                fate = "fp16 (never evicted)"
            elif evict[p]:
                fate = "int8 -> EVICTED after grace"
            elif p_ev[p] >= args.evict_margin:
                fate = "int8 (protected: sink/grace)"
            else:
                fate = "int8 kept"
            row += f"{imp[p]:>8.2f}{p_ev[p]:>9.2f}  {fate}"
        print(row)

    n_traj_ev = int(evict_label.sum())
    print(f"\nTrajectory says quiet-after-W: {n_traj_ev}/{prompt_length} tokens")
    if imp is not None:
        agree = float((( p_ev >= args.evict_margin).astype(int) == evict_label).mean())
        print(f"Probe B agreement with trajectory (pre-rules): {agree:.1%}")
        n_hi = int(is_fp16.sum()); n_ev = int(evict.sum())
        n_lo = prompt_length - n_hi - n_ev
        bm = BytesModel(model.config)
        full = prompt_length * bm.fp16_per_tok
        peak = bm.mixed_bytes(n_hi, prompt_length - n_hi)   # everything stored first
        steady = bm.mixed_bytes(n_hi, n_lo)                 # after eviction catches up
        print(f"\nStorage on this prompt: {n_hi} fp16 + {n_lo} int8 kept "
              f"+ {n_ev} evicted after grace")
        print(f"  all-fp16 {full / 1e3:.1f} KB -> peak {peak / 1e3:.1f} KB "
              f"(KVCIS) -> steady {steady / 1e3:.1f} KB "
              f"({(1 - steady / full) * 100:.1f}% reduction)")

    print("\nStep 1 complete - activation extraction, trajectory labeling, and "
          "two-probe decisions working")


if __name__ == "__main__":
    main()
