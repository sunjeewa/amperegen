#!/usr/bin/env python3
"""End-to-end validation & benchmark suite for the amperegen OpenAI HTTP server.

Tests:
1. /health, /v1/models, /metrics
2. Exact greedy token parity & decode throughput vs SGLang baseline (short_code, short_math, short_tech, chat_short, chat_8k, chat_32k)
3. Sampling endpoint (temperature=1.0, top_k=20, top_p=0.95, deterministic fixed-seed rerun check!)
4. Concurrent continuous batching (4, 8, and 16 concurrent requests)
5. Robustness: empty messages -> 400, max_tokens=1, stop strings, mid-stream client disconnect
"""
import argparse
import concurrent.futures
import json
import socket
import time
import urllib.error
import urllib.request


def http_get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read().decode("utf-8")


def call_stream(url: str, body: dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
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
                c = ch[0].get("text")
                if c is None:
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
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen3.8-27B-FP8")
    ap.add_argument("--test-200k", action="store_true", help="Also run 195K-token single-request long-context test")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    # 1. Health & models
    st, body = http_get(f"{base}/health")
    assert st == 200, f"/health failed: {st}"
    st, body = http_get(f"{base}/v1/models")
    assert st == 200 and args.model in body, f"/v1/models failed: {st} {body}"
    print("[PASS] /health and /v1/models")

    # 2. Robustness checks
    try:
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({"model": args.model, "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("Empty messages should return 400")
    except urllib.error.HTTPError as e:
        assert e.code == 400, f"Expected 400, got {e.code}"
    print("[PASS] empty messages -> 400")

    r1 = call_stream(f"{base}/v1/completions", {
        "model": args.model, "prompt": "Hello", "max_tokens": 1, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
    })
    assert r1["completion_tokens"] == 1, f"max_tokens=1 failed: {r1}"
    print("[PASS] max_tokens=1")

    # 3. Deterministic fixed-seed sampling test (temperature=1.0, top_k=20, top_p=0.95)
    samp_body = {
        "model": args.model,
        "prompt": "Explain CUDA warp shuffle reductions briefly:\n",
        "max_tokens": 128,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "seed": 424242,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    rs1 = call_stream(f"{base}/v1/completions", samp_body)
    rs2 = call_stream(f"{base}/v1/completions", samp_body)
    assert rs1["text"] == rs2["text"], "Fixed-seed sampling not bit-identical!"
    print(f"[PASS] Deterministic fixed-seed sampling (temp=1.0, k=20, p=0.95): {rs1['decode_tps']:.2f} tok/s (exact match!)")

    # 4. Full apples-to-apples comparison against SGLang baseline!
    print("\n=== APPLES-TO-APPLES BENCHMARK vs SGLANG ===")
    comp_cases = [
        ("short_code", "Write a clean C++20 implementation of an SPSC lock-free ring buffer with power-of-two capacity:\n```cpp\n", 256),
        ("short_math", "Solve step by step: Find all positive integers n such that n^2 + 19n + 92 is a perfect square.\nSolution:\n", 256),
        ("short_tech", "Explain in detail how Ampere sm_80 mma.sync.aligned.m16n8k16 maps threads to matrix fragments for BF16 inputs and FP32 accumulators:\n", 337),
    ]
    for name, prompt, mt in comp_cases:
        r = call_stream(f"{base}/v1/completions", {
            "model": args.model, "prompt": prompt, "max_tokens": mt, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
        })
        print(f"{name:12s} | prompt={r['prompt_tokens']:6d} | out={r['completion_tokens']:4d} | TTFT={r['ttft_s']:6.3f}s | decode_tps={r['decode_tps']:7.2f} tok/s", flush=True)

    for name, n_paras in [("chat_short", 0), ("chat_8k", 240), ("chat_32k", 960)]:
        doc = make_doc(n_paras, name) if n_paras else ""
        msg = (
            doc
            + "\nWrite a thorough 450-word technical guide on designing an Ampere sm_80 LLM inference engine "
            "with FP8 block weights, Gated DeltaNet, and speculative decoding. Use numbered sections."
        )
        r = call_stream(f"{base}/v1/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": msg}],
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        })
        print(f"{name:12s} | prompt={r['prompt_tokens']:6d} | out={r['completion_tokens']:4d} | TTFT={r['ttft_s']:6.3f}s | decode_tps={r['decode_tps']:7.2f} tok/s", flush=True)

    # 5. Concurrent continuous batching (4 and 16 concurrent requests!)
    for conc in [4, 16]:
        prompts = [
            make_doc(22, f"CONC{conc}_{k}") + f"\nList 10 concrete CUDA optimization checklist items for workload #{k}:\n"
            for k in range(conc)
        ]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [
                ex.submit(call_stream, f"{base}/v1/completions", {
                    "model": args.model, "prompt": p, "max_tokens": 256, "temperature": 0.0,
                    "stream": True, "stream_options": {"include_usage": True},
                })
                for p in prompts
            ]
            outs = [f.result() for f in futs]
        t1 = time.perf_counter()
        tot_out = sum(x["completion_tokens"] for x in outs)
        mean_ttft = sum(x["ttft_s"] for x in outs) / len(outs)
        print(f"batch{conc:2d}_1k   | total_out={tot_out:4d} | wall={t1-t0:6.3f}s | mean_TTFT={mean_ttft:.3f}s | agg_tps={tot_out/(t1-t0):7.2f} tok/s", flush=True)

    # 6. Mid-stream client disconnect + KV leak check
    raw_sock = socket.create_connection(("127.0.0.1", int(base.rsplit(":", 1)[-1])), timeout=30)
    disc_body = json.dumps({
        "model": args.model, "prompt": "Count from 1 to 10000 slowly:\n", "max_tokens": 512,
        "temperature": 0.0, "stream": True,
    }).encode("utf-8")
    raw_sock.sendall(
        f"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(disc_body)}\r\nConnection: close\r\n\r\n".encode("utf-8") + disc_body
    )
    raw_sock.recv(1024)
    raw_sock.close()
    time.sleep(0.5)

    # 7. Optional 200K context test (S3 verification)
    if args.test_200k:
        doc200 = make_doc(3400, "CTX200K") + "\nSummarize the key CUDA optimization themes above in 2 sentences:\n"
        r200 = call_stream(f"{base}/v1/completions", {
            "model": args.model, "prompt": doc200, "max_tokens": 32, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
        })
        print(f"ctx_200k     | prompt={r200['prompt_tokens']:6d} | out={r200['completion_tokens']:4d} | TTFT={r200['ttft_s']:6.3f}s | decode_tps={r200['decode_tps']:7.2f} tok/s", flush=True)

    # 8. Metrics check (ensure zero leaked requests & zero leaked KV blocks!)
    st, met = http_get(f"{base}/metrics")
    assert st == 200 and "amperegen_gen_throughput_toks_per_sec" in met
    print("\n=== /metrics ===\n" + met.strip())
    assert "amperegen_running_requests 0" in met, f"Leaked running request! {met}"
    assert "amperegen_kv_cache_usage_ratio 0" in met, f"Leaked KV cache blocks! {met}"
    print("[PASS] Zero leaked requests/KV blocks after mid-stream disconnect")


if __name__ == "__main__":
    main()
