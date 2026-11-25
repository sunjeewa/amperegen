#!/usr/bin/env python3
"""Benchmark live SGLang server and save golden greedy reference outputs."""
import argparse
import concurrent.futures
import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "Qwen3.8-27B-FP8"


def call_stream(prompt: str, max_tokens: int = 256, temperature: float = 0.0, logprobs=None):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    lp_chunks = []
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
                c = ch[0].get("text", "")
                if c and t_first is None:
                    t_first = time.perf_counter()
                if c:
                    text.append(c)
                if logprobs is not None and ch[0].get("logprobs"):
                    lp_chunks.append(ch[0]["logprobs"])
    t1 = time.perf_counter()
    ttft = (t_first - t0) if t_first else (t1 - t0)
    decode_t = (t1 - t_first) if t_first else 0.0
    comp_toks = usage["completion_tokens"] if usage else len(text)
    prompt_toks = usage["prompt_tokens"] if usage else 0
    decode_tps = (comp_toks - 1) / decode_t if decode_t > 0 and comp_toks > 1 else 0.0
    return {
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
        "ttft_s": ttft,
        "decode_s": decode_t,
        "total_s": t1 - t0,
        "decode_tps": decode_tps,
        "text": "".join(text),
        "logprobs": lp_chunks[:25],
    }


def main():
    global URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ref/sglang_golden.json")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()
    URL = args.url

    # Warmup
    _ = call_stream("Count from 1 to 5:", max_tokens=24, temperature=0.0)

    # Non-repetitive synthetic long context generator (different numbered sentences to prevent trivial prefix-copy inflation)
    def make_context(n_paras: int, tag: str) -> str:
        topics = [
            "PCIe Gen4 DMA transfers and pinned host staging buffers",
            "HBM2e channel interleaving and L2 cache line alignment on GA100",
            "Tensor Core mma.sync.aligned.m16n8k16 register fragment mapping",
            "Gated DeltaNet rank-1 state updates with exponential decay",
            "Paged KV cache block allocation and FP8 E4M3 quantization range",
            "DFlash2 bidirectional sliding-window attention and dynamic grouped conv",
            "CUDA Graph static capture with asynchronous host-to-device input updates",
            "Continuous batching scheduler with chunked prefill and decode verify",
        ]
        parts = [f"Document ID {tag} — Systems Engineering Notes.\n"]
        for i in range(n_paras):
            t1 = topics[i % len(topics)]
            t2 = topics[(i * 3 + 1) % len(topics)]
            parts.append(
                f"Section {i+1}: When combining {t1} with {t2}, register allocation "
                f"(target {64 + (i % 128)} regs/thread) and shared memory padding "
                f"(stride {16 + (i % 32)} bytes) determine occupancy on 70 SMs under a 210W envelope. "
                f"Measured latency delta for configuration #{1000 + i * 7} was {(i % 19) * 0.13 + 1.2:.2f} ms.\n"
            )
        return "".join(parts)

    prompts = {
        "short_code": (
            "Write a clean C++20 implementation of an SPSC lock-free ring buffer with power-of-two capacity:\n```cpp\n",
            256,
            5,
        ),
        "short_math": (
            "Solve step by step: Find all positive integers n such that n^2 + 19n + 92 is a perfect square.\nSolution:\n",
            256,
            5,
        ),
        "short_tech": (
            "Explain in detail how Ampere sm_80 mma.sync.aligned.m16n8k16 maps threads to matrix fragments for BF16 inputs and FP32 accumulators:\n",
            512,
            5,
        ),
        "ctx_1k": (
            make_context(22, "CTX1K")
            + "\nBased on the engineering notes above, write a 200-word synthesis of the top 5 GPU optimization principles:\n",
            256,
            None,
        ),
        "ctx_8k": (
            make_context(180, "CTX8K")
            + "\nWrite a detailed engineering design for minimizing HBM traffic during 27B FP8 speculative verification:\n",
            512,
            None,
        ),
        "ctx_32k": (
            make_context(700, "CTX32K")
            + "\nWrite a detailed analysis of how Gated DeltaNet state management interacts with speculative decoding and KV cache eviction:\n",
            512,
            None,
        ),
    }

    results = {}
    for name, (p, max_tok, lp) in prompts.items():
        r = call_stream(p, max_tokens=max_tok, temperature=0.0, logprobs=lp)
        results[name] = {"prompt": p, **r}
        print(
            f"{name:12s} | prompt={r['prompt_tokens']:6d} | out={r['completion_tokens']:4d} | "
            f"TTFT={r['ttft_s']:6.3f}s | decode={r['decode_s']:6.3f}s | decode_tps={r['decode_tps']:7.2f} tok/s",
            flush=True,
        )

    # Also test 4 concurrent 1K->256 requests to measure batched throughput
    conc_prompts = [
        make_context(22, f"CONC4_{k}") + f"\nList 10 concrete CUDA optimization checklist items for workload #{k}:\n"
        for k in range(4)
    ]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(call_stream, cp, 256, 0.0, None) for cp in conc_prompts]
        conc_res = [f.result() for f in futs]
    t1 = time.perf_counter()
    tot_out = sum(x["completion_tokens"] for x in conc_res)
    agg_tps = tot_out / (t1 - t0)
    mean_ttft = sum(x["ttft_s"] for x in conc_res) / len(conc_res)
    results["batch4_1k"] = {
        "wall_s": t1 - t0,
        "total_out_tokens": tot_out,
        "agg_tps": agg_tps,
        "mean_ttft_s": mean_ttft,
        "per_req_tps": [x["decode_tps"] for x in conc_res],
    }
    print(
        f"batch4_1k    | total_out={tot_out:4d} | wall={t1-t0:6.3f}s | "
        f"mean_TTFT={mean_ttft:.3f}s | agg_tps={agg_tps:7.2f} tok/s",
        flush=True,
    )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved golden results to {args.out}")


if __name__ == "__main__":
    main()
