#!/usr/bin/env python3
"""Benchmark live OpenAI-compatible /v1/chat/completions at short, 8K, and 32K context."""
import argparse
import json
import time
import urllib.request


def call_chat(url: str, model: str, messages, max_tokens=512, temperature=0.0):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices", [])
            if ch:
                d = ch[0].get("delta", {})
                c = d.get("content") or d.get("reasoning_content") or ""
                if c and t_first is None:
                    t_first = time.perf_counter()
                if c:
                    text.append(c)
    t1 = time.perf_counter()
    ttft = (t_first - t0) if t_first else (t1 - t0)
    decode_t = (t1 - t_first) if t_first else 0.0
    comp = usage["completion_tokens"] if usage else len(text)
    prompt = usage["prompt_tokens"] if usage else 0
    tps = (comp - 1) / decode_t if decode_t > 0 and comp > 1 else 0.0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": comp,
        "ttft_s": ttft,
        "decode_s": decode_t,
        "decode_tps": tps,
        "text": "".join(text),
    }


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="Qwen3.8-27B-FP8")
    ap.add_argument("--out", default="ref/sglang_chat_golden.json")
    args = ap.parse_args()

    res = {}
    for name, n_paras in [("chat_short", 0), ("chat_8k", 240), ("chat_32k", 960)]:
        doc = make_doc(n_paras, name) if n_paras else ""
        msg = (
            doc
            + "\nWrite a thorough 450-word technical guide on designing an Ampere sm_80 LLM inference engine "
            "with FP8 block weights, Gated DeltaNet, and speculative decoding. Use numbered sections."
        )
        r = call_chat(args.url, args.model, [{"role": "user", "content": msg}], max_tokens=512, temperature=0.0)
        res[name] = r
        print(
            f"{name:12s} | prompt={r['prompt_tokens']:6d} | out={r['completion_tokens']:4d} | "
            f"TTFT={r['ttft_s']:6.3f}s | decode={r['decode_s']:6.3f}s | decode_tps={r['decode_tps']:7.2f} tok/s",
            flush=True,
        )

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
