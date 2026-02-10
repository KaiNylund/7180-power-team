import json
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
from nnsight import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import numpy as np
import torch

BASE = Path(__file__).parent


## NOTE: to run properly, use the virtual env from the "Representation geometry" demo
# at https://github.com/davidbau/puns

# ── Load environment ──────────────────────────────────────────────────────────
env_path = BASE / ".env.local"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

# Create a file .env.local with these two keys
os.environ["NNSIGHT_API_KEY"] = os.getenv("NDIF_API_KEY", "")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = BASE / "activation_results"
DATASET_FNAME = "contrasting_power_over_prompts"
dataset_short = "contrasting_power_over"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT = "llama31_8b_instruct"

batch_size = 10

# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_chat_prompt(tokenizer, user_text):
    """Apply the instruct chat template to match the behavioral experiment."""
    messages = [
        #{"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},# + " ___"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).replace("___", "")


# ── Token position finding ────────────────────────────────────────────────────

def find_pred_c_position(tokenizer, formatted_prompt, user_text):
    """
    Last token of the user's text — where model predicts C's completion.

    The chat template appends an assistant header after the user text,
    so we find the last content token by comparing the full formatted
    prompt with a version that has slightly shorter user text.
    """
    # Build a version with one less word to find where user content ends
    shorter = format_chat_prompt(tokenizer, user_text.rsplit(None, 1)[0])
    full_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    short_tokens = tokenizer.encode(shorter, add_special_tokens=False)

    # Find divergence point
    shared = 0
    for a, b in zip(full_tokens, short_tokens):
        if a != b:
            break
        shared += 1

    # The last token of the full user text is somewhere after the shared prefix.
    # We want the last token before the eot/assistant header.
    # Since the full prompt has more content tokens, count forward from divergence.
    # Actually, just count how many extra tokens the full prompt has before
    # the template suffix kicks in.
    #
    # Simpler: use the full prompt tokens and find the last non-special content token.
    # The generation prompt ends with the assistant header.  Count backwards to find
    # the last user content token.

    # Find where the post-user-content section starts by looking at the suffix
    suffix_text = tokenizer.apply_chat_template(
         [{"role": "user", "content": "X"}],
        tokenize=False, add_generation_prompt=True
    )
    suffix_after_content = suffix_text.split("X", 1)[1]
    suffix_tokens = tokenizer.encode(suffix_after_content, add_special_tokens=False)
    n_suffix = len(suffix_tokens)

    #print("-" * 30)
    #print(tokenizer.decode(full_tokens[len(full_tokens) - n_suffix - 1]))
    #print("-" * 30)
    return len(full_tokens) - n_suffix - 1


def collect_batch(model, layers_module, prompts_and_positions,
                  layer_indices=None, batch_size=batch_size, remote=False,
                  save_dir=None, file_prefix="", batch_offset=0,
                  target_token_ids_per_prompt=None, top_k=20):
    """
    Unified collection pass: activations and/or detailed predictions from
    a single forward pass per batch via NDIF.

    Collects:
    - Layer activations (if layer_indices is provided and non-empty)
    - Detailed predictions: top-k tokens + target token log-probs
      (if target_token_ids_per_prompt is provided)

    Server-side optimization: all proxy operations (indexing, stacking,
    log_softmax, topk) execute on the server; only compact results are
    transferred.

    Parameters:
        layer_indices: list of int — layers to collect (None/[] = skip activations)
        target_token_ids_per_prompt: list of lists — target tokens per prompt
            (None = skip detailed predictions, just collect top-1)
        top_k: int — number of top tokens to collect (default 20)
        save_dir: Path — if provided, saves incrementally
        file_prefix: str — filename prefix
        batch_offset: int — starting batch number for filenames (for resume)

    Returns:
        layer_data: dict {layer_idx: np.array} or {} if no activations
        detailed_preds: list of dicts with topk_ids, topk_logprobs,
            target_logprobs — or None if target_token_ids not provided
    """
    n = len(prompts_and_positions)
    n_batches = (n + batch_size - 1) // batch_size
    n_layers = len(layer_indices)

    # Accumulate results across batches
    layer_results = {l: [] for l in layer_indices}
    all_detailed = []

    #print(prompts_and_positions)


    # Layer outputs are tuples for most models (hidden_states, ...), access [0]
    # We assume tuple format - this works for Llama and similar architectures
    print(f"    Layer output: assuming tuple format (hidden_states, ...)", flush=True)

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = prompts_and_positions[batch_start:batch_end]
        batch_n = len(batch)

        # Compute per-prompt token lengths and left-padding offsets
        token_lengths = [
            len(model.tokenizer.encode(prompt, add_special_tokens=False))
            for prompt, pos in batch
        ]
        max_len = max(token_lengths)
        pad_offsets = [max_len - tl for tl in token_lengths]

        # Get target tokens for this batch if collecting detailed preds
        batch_targets = None
        if target_token_ids_per_prompt is not None:
            batch_targets = target_token_ids_per_prompt[batch_start:batch_end]

        # ── Server-side computation ──────────────────────────────────
        invoked_prompts = []
        with model.trace(remote=remote) as tracer:
            prompt_results = []
            for idx, ((prompt, pos), pad_offset) in enumerate(
                zip(batch, pad_offsets)
            ):
                prompt_token_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
                invoked_prompts.append(prompt_token_ids)
                adjusted_pos = pos + pad_offset
                with tracer.invoke(prompt_token_ids):
                    result = defaultdict(list)
                    # Stack all layer activations: (n_layers, hidden_dim)
                    layer_vecs = []
                    for layer_idx in layer_indices:
                        out = layers_module[layer_idx].output
                        # Layer output is tuple (hidden_states, ...), access [0]
                        hidden = out[0]
                        #print(layer_vecs)
                        #print(hidden.cpu().size(), adjusted_pos)
                        layer_vecs.append(hidden[adjusted_pos, :].cpu())
                    result["activations"] = torch.stack(layer_vecs)
                    # Predictions from the same forward pass
                    logits = model.lm_head.output[0, adjusted_pos, :].cpu()

                    # Full log-softmax for top-k and target probs
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    topk = log_probs.topk(top_k)
                    result["topk_vals"] = topk.values
                    result["topk_ids"] = topk.indices

                    # Target token log-probs
                    if target_token_ids_per_prompt is not None:
                        tids = batch_targets[idx]
                        target_lps = []
                        for tid in tids:
                            target_lps.append(log_probs[tid])
                        result["target_lps"] = torch.stack(target_lps) if target_lps \
                            else topk.values[:0]

                    prompt_results.append(result)
            saved_batch = prompt_results.save()

        # ── Client-side unpacking ────────────────────────────────────
        batch_vectors = {l: [] for l in layer_indices}

        for i in range(batch_n):
            acts = saved_batch[i]["activations"]
            if hasattr(acts, 'dtype') and acts.dtype == torch.bfloat16:
                acts = acts.half()
            acts = acts.cpu().detach().numpy()

            for j, layer_idx in enumerate(layer_indices):
                layer_results[layer_idx].append(acts[j])
                batch_vectors[layer_idx].append(acts[j])

            out_dict = {
                "topk_ids": saved_batch[i]["topk_ids"].cpu().detach().numpy().tolist(),
                "topk_logprobs": saved_batch[i]["topk_vals"].cpu().detach().numpy().tolist()
            }
            if "target_lps" in saved_batch[i]:
                out_dict["target_logprobs"] = saved_batch[i]["target_lps"].cpu().detach().numpy().tolist()

            all_detailed.append(out_dict)

        # ── Incremental save ─────────────────────────────────────────
        if save_dir is not None:
            batch_num = batch_offset + batch_start // batch_size

            for layer_idx in layer_indices:
                filename = f"{file_prefix}_layer{layer_idx:02d}_batch{batch_num:02d}.npy"
                np.save(save_dir / filename, np.stack(batch_vectors[layer_idx]))

            # Save detailed predictions for this batch as JSON
            pred_filename = f"{file_prefix}_preds_batch{batch_num:02d}.json"
            batch_detailed = all_detailed[-batch_n:]
            with open(save_dir / pred_filename, "w") as f:
                json.dump(batch_detailed, f)

        batch_display = batch_offset + batch_start // batch_size + 1
        total_batches = n_batches + batch_offset
        pad_range = f"pad=[{min(pad_offsets)}-{max(pad_offsets)}]"
        parts = []
        parts.append(f"{n_layers} layers")
        parts.append(f"top-{top_k} preds")
        print(f"    batch {batch_display}/{total_batches}: "
              f"prompts {batch_start}-{batch_end-1} "
              f"({batch_n} prompts, {', '.join(parts)}) "
              f"{pad_range}", flush=True)

    # Stack each layer's results
    layer_data = {l: np.stack(vecs) for l, vecs in layer_results.items()}
    return layer_data, all_detailed

if __name__ == "__main__":
    # Load data
    with open(f"./{DATASET_FNAME}.txt", "r") as prompts_file:
        prompts = prompts_file.read().split("\t")[:20]

    print(prompts[0])
    dataset_df = pd.read_csv("./contrasting_power_over.tsv", sep="\t").iloc[:20]

    # Initialize model
    print(f"\nInitializing model: {MODEL_NAME}", flush=True)

    # Do this if using cuda or no GPU
    model = LanguageModel(MODEL_NAME, device_map="auto")

    # Do this if using apple silicon? Not completely working yet
    #hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    #hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    #hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")
    #model = LanguageModel(hf_model, config=hf_config)
    #model.config = hf_config
    #hf_tokenizer.pad_token = hf_tokenizer.eos_token
    #model.tokenizer = hf_tokenizer

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}", flush=True)

    # Select layers (if collecting activations)
    #elif args.layers:
    #    layer_indices = sorted(args.layers)
    #    print(f"  Collecting {len(layer_indices)} layers", flush=True)
    #else:
    layer_indices = list(range(n_layers))
    print(f"  Collecting all {len(layer_indices)} layers", flush=True)

    def word_to_token(word):
        toks = model.tokenizer.encode(" " + word, add_special_tokens=False)
        return toks[0] if toks else None

    # Compute token positions
    target_token_ids_per_prompt = []
    prompts_and_positions = []
    for prompt in prompts:
        formatted_prompt = format_chat_prompt(model.tokenizer, prompt)
        pos = find_pred_c_position(model.tokenizer, formatted_prompt, prompt)
        prompts_and_positions.append((formatted_prompt, pos))
        target_token_ids_per_prompt.append([word_to_token("YES"), word_to_token("NO"), word_to_token("yes"), word_to_token("no")])


    # For debugging position where activations are collected
    #test_prompt, test_pos = prompts_and_positions[0]
    #test_tokens = model.tokenizer.encode(test_prompt, add_special_tokens=False)
    #print(f"\nDEBUG: Last 20 tokens of prompt:")
    #print(model.tokenizer.decode(test_tokens[-20:]))
    #print(f"\nToken at position {test_pos}: '{model.tokenizer.decode([test_tokens[test_pos]])}'")
    #print(f"Token at position {test_pos+1}: '{model.tokenizer.decode([test_tokens[test_pos+1]])}'")

    positions = [pos for _, pos in prompts_and_positions]
    print(f"  Position range: min={min(positions)}, max={max(positions)}, "
            f"mean={sum(positions)/len(positions):.0f}", flush=True)

    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layers_module = model.model.layers

    # Save metadata once
    metadata = []
    for i, (prompt, pos) in enumerate(prompts_and_positions):
        metadata.append({
            "index": i,
            "prompt": prompt,
            "token_position": pos,
        })

    # Build file prefix - include dataset name if not default
    file_prefix = f"{MODEL_SHORT}_{dataset_short}"

    meta_filename = f"{file_prefix}_results.json"
    meta_out = {
        "model": MODEL_NAME,
        "model_short": MODEL_SHORT,
        "dataset": dataset_short,
        "n_prompts": len(metadata),
        "n_layers_total": n_layers,
        "hidden_dim": hidden_dim,
        "file_shape": [len(metadata), hidden_dim],
        "file_axes": ["prompt", "hidden_dim"],
        "naming": f"{file_prefix}_layer{{NN}}.npy",
        "samples": metadata,
    }

    # ── Determine what to collect ─────────────────────────────────────────
    n_prompts = len(prompts_and_positions)
    n_batches = (n_prompts + batch_size - 1) // batch_size

    # Check which layers still need collection (if collecting activations)
    needed_layers = []
    for layer_idx in layer_indices:
        merged = OUTPUT_DIR / f"{file_prefix}_layer{layer_idx:02d}.npy"
        if merged.exists():
            existing = np.load(merged)
            if existing.shape[0] >= n_prompts:
                continue
        needed_layers.append(layer_idx)

    if not needed_layers:
        print(f"\nAll {len(layer_indices)} layers already collected.")

    # ── Check for partial batch files (resume support) ──────────────────
    existing_batches = 0
    remaining = prompts_and_positions
    remaining_targets = target_token_ids_per_prompt

    if needed_layers:
        sample_layer = needed_layers[0]
        for b in range(n_batches):
            bf = OUTPUT_DIR / f"{file_prefix}_layer{sample_layer:02d}_batch{b:02d}.npy"
            if bf.exists():
                existing_batches = b + 1
            else:
                break

        if existing_batches > 0:
            skip_prompts = existing_batches * batch_size
            print(f"\n  Found {existing_batches} batch files, "
                    f"resuming from prompt {skip_prompts}", flush=True)
            remaining = prompts_and_positions[skip_prompts:]
            if remaining_targets:
                remaining_targets = target_token_ids_per_prompt[skip_prompts:]

    # ── Show collection plan ────────────────────────────────────────────
    parts = []
    if needed_layers:
        parts.append(f"{len(needed_layers)} layers")

    print(f"\n--- Collecting: {', '.join(parts)} x "
            f"{len(remaining)} prompts ---", flush=True)

    if needed_layers:
        print(f"  Estimated activation download: "
                f"{len(needed_layers) * len(remaining) * hidden_dim * 2 / 1e6:.0f} MB",
                flush=True)

    # ── Run unified collection ──────────────────────────────────────────
    layer_data, detailed_raw = collect_batch(
        model, layers_module, remaining,
        layer_indices=needed_layers,
        batch_size=batch_size, remote=False,
        save_dir=OUTPUT_DIR, file_prefix=file_prefix,
        batch_offset=existing_batches,
        target_token_ids_per_prompt=remaining_targets,
        top_k=20,
    )

    # ── Post-process activations ────────────────────────────────────────
    if needed_layers:
        print(f"\n  Merging batch files...", flush=True)
        for layer_idx in needed_layers:
            parts = []
            for b in range(n_batches):
                bf = OUTPUT_DIR / f"{file_prefix}_layer{layer_idx:02d}_batch{b:02d}.npy"
                if bf.exists():
                    parts.append(np.load(bf))
            if parts:
                merged = np.concatenate(parts, axis=0)
                np.save(OUTPUT_DIR / f"{file_prefix}_layer{layer_idx:02d}.npy", merged)
            for b in range(n_batches):
                bf = OUTPUT_DIR / f"{file_prefix}_layer{layer_idx:02d}_batch{b:02d}.npy"
                if bf.exists():
                    bf.unlink()

        sample = np.load(OUTPUT_DIR / f"{file_prefix}_layer{needed_layers[0]:02d}.npy")
        print(f"  Saved {len(needed_layers)} layer files to {OUTPUT_DIR}/")
        print(f"  File shape: {sample.shape}  dtype={sample.dtype} "
                f"({sample.nbytes / 1e6:.1f} MB each)")

    # ── Post-process activations ────────────────────────────────────────
    # Merge batch files if resuming
    all_detailed_raw = []
    for b in range(n_batches):
        pf = OUTPUT_DIR / f"{file_prefix}_preds_batch{b:02d}.json"
        if pf.exists():
            with open(pf) as f:
                all_detailed_raw.extend(json.load(f))
            pf.unlink()

    # Assemble into structured JSON
    detailed = []
    for i, raw in enumerate(all_detailed_raw):
        top_tokens = []
        for tid, lp in zip(raw["topk_ids"], raw["topk_logprobs"]):
            word = model.tokenizer.decode([tid]).strip()
            top_tokens.append({
                "token_id": tid, "word": word,
                "logprob": round(lp, 4),
                "prob": round(float(np.exp(lp)), 6),
            })
        meta_out["samples"][i]["top_tokens"] = top_tokens

        # Get top-1 prediction info for metadata
        top1_id = raw["topk_ids"][0]
        top1_word = model.tokenizer.decode([top1_id]).strip().lower()
        meta_out["samples"][i]["predicted_token_id"] = top1_id
        meta_out["samples"][i]["predicted_word"] = top1_word
        if "target_logprobs" in raw:
            meta_out["samples"][i]["predicted_word"] = raw["target_logprobs"]

    with open(OUTPUT_DIR / meta_filename, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\nSaved metadata: {meta_filename}", flush=True)