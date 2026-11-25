#!/usr/bin/env python3
"""Tokenize SGLang golden prompts to binary int32 files and verify amperegen outputs against SGLang."""
import argparse
import json
import os
import struct
from transformers import AutoTokenizer

DEFAULT_MODEL_DIR = os.environ.get("AMPEREGEN_MODEL_DIR", "./models/Qwen3.8-27B-FP8")


def make_doc(n_paras: int, tag: str) -> str:
    topics = [
        "PCIe Gen4 DMA transfers and pinned host staging buffers",
        "HBM2e channel interleaving and L2 cache line alignment on GA100",
        "Tensor Core mma.sync.aligned.m16n8k16 register fragment mapping",
        "Gated DeltaNet rank-1 state updates with exponential decay",
        "Paged KV cache block allocation and FP8 E4M3 quantization range",
        "DFlash2 bidirectional sliding-window attention and dynamic grouped conv",
    ]
    return f"Doc {tag}:\n" + "".join(
        f"Note {i+1}: Combining {topics[i % len(topics)]} with {topics[(i * 3 + 1) % len(topics)]} "
        f"yields delta {(i % 19) * 0.13 + 1.2:.2f}ms at cfg {1000 + i * 7}.\n"
        for i in range(n_paras)
    )


def prep(model_dir: str, ref_dir: str, out_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(ref_dir, "sglang_golden.json")) as f:
        g = json.load(f)
    for k in ["short_code", "short_math", "short_tech"]:
        ids = tok.encode(g[k]["prompt"], add_special_tokens=False)
        with open(f"{out_dir}/{k}.bin", "wb") as bf:
            bf.write(struct.pack(f"<{len(ids)}i", *ids))
        out_ids = tok.encode(g[k]["text"], add_special_tokens=False)
        print(f"{k:12s}: prompt={len(ids):6d} | sglang_out={g[k]['completion_tokens']:4d} | sglang_tps={g[k]['decode_tps']:.2f}")

    with open(os.path.join(ref_dir, "sglang_chat_golden.json")) as f:
        cg = json.load(f)
    for name, n_paras in [("chat_short", 0), ("chat_8k", 240), ("chat_32k", 960)]:
        doc = make_doc(n_paras, name) if n_paras else ""
        msg = (
            doc
            + "\nWrite a thorough 450-word technical guide on designing an Ampere sm_80 LLM inference engine "
            "with FP8 block weights, Gated DeltaNet, and speculative decoding. Use numbered sections."
        )
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok.encode(rendered, add_special_tokens=False)
        with open(f"{out_dir}/{name}.bin", "wb") as bf:
            bf.write(struct.pack(f"<{len(ids)}i", *ids))
        print(
            f"{name:12s}: prompt={len(ids):6d} (sglang={cg[name]['prompt_tokens']:6d}) | "
            f"sglang_TTFT={cg[name]['ttft_s']:.3f}s | sglang_tps={cg[name]['decode_tps']:.2f}"
        )


def compare(model_dir: str, ref_dir: str, out_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    with open(os.path.join(ref_dir, "sglang_golden.json")) as f:
        g = json.load(f)
    with open(os.path.join(ref_dir, "sglang_chat_golden.json")) as f:
        cg = json.load(f)
    all_ref = {**g, **cg}

    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".out.bin"):
            continue
        name = fn[:-8]
        base_name = name.split("__")[0]
        raw = open(f"{out_dir}/{fn}", "rb").read()
        ids = list(struct.unpack(f"<{len(raw)//4}i", raw))
        text = tok.decode(ids, skip_special_tokens=True)
        ref_text = all_ref.get(base_name, {}).get("text", "")
        # Compute exact prefix match length in characters and token overlap
        ref_ids = tok.encode(ref_text, add_special_tokens=False) if ref_text else []
        match_prefix = 0
        for a, b in zip(ids, ref_ids):
            if a == b:
                match_prefix += 1
            else:
                break
        print(f"=== {name} (gen={len(ids)} toks, prefix_exact_vs_sglang={match_prefix}/{min(len(ids), len(ref_ids))}) ===")
        print("  AMPEREGEN:", repr(text[:220]))
        if ref_text:
            print("  SGLANG   :", repr(ref_text[:220]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--ref-dir", default="ref")
    ap.add_argument("--dir", default="ref/prompts")
    args = ap.parse_args()
    if args.prep:
        prep(args.model_dir, args.ref_dir, args.dir)
    if args.compare:
        compare(args.model_dir, args.ref_dir, args.dir)


if __name__ == "__main__":
    main()
