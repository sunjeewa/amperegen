#!/usr/bin/env python3
"""Stage-1 pure-PyTorch reference for Qwen3.8-27B-FP8 (GDN layer 0, Full-Attn layer 3, DFlash2).

Generates deterministic golden test vectors (binary .bin files) to verify every amperegen CUDA kernel
against exact PyTorch math (CPU/FP32 or BF16) before running full-model inference.
"""
import argparse
import os
import struct
import torch
import torch.nn.functional as F
from safetensors import safe_open

DEFAULT_MODEL_DIR = os.environ.get("AMPEREGEN_MODEL_DIR", "./models/Qwen3.8-27B-FP8")


def dequant_fp8_block128(w_fp8: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    # w_fp8: [N, K] float8_e4m3fn, scale_inv: [N/128, K/128] bf16
    N, K = w_fp8.shape
    w = w_fp8.to(torch.float32)
    s = scale_inv.to(torch.float32).repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w * s[:N, :K]).to(torch.bfloat16)


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * r * (1.0 + weight.float())).to(torch.bfloat16)


def plain_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * r * weight.float()).to(torch.bfloat16)


def rmsnorm_gated_swish(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = xf * r * weight.float()
    gate = F.silu(z.float())
    return (normed * gate).to(torch.bfloat16)


def apply_neox_partial_rope(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int = 64, theta: float = 1e7):
    # x: [T, H, D]
    xf = x.float()
    x_rot = xf[..., :rotary_dim]
    x_pass = xf[..., rotary_dim:]
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]  # [T, half]
    cos = freqs.cos()[:, None, :]
    sin = freqs.sin()[:, None, :]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    out_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([out_rot, x_pass], dim=-1).to(x.dtype)


def ref_gdn_layer(layer_path: str, layer_idx: int, x_in: torch.Tensor):
    """Exact pure-PyTorch reference for Qwen3.8 GDN layer (prefill seq of length T, then 1-step decode)."""
    pref = f"model.language_model.layers.{layer_idx}."
    with safe_open(layer_path, framework="pt", device="cpu") as f:
        def get(name):
            return f.get_tensor(pref + name)
        def get_fp8(name):
            return dequant_fp8_block128(get(name + ".weight"), get(name + ".weight_scale_inv"))

        in_norm_w = get("input_layernorm.weight")
        post_norm_w = get("post_attention_layernorm.weight")
        w_qkv = get_fp8("linear_attn.in_proj_qkv")  # [10240, 5120]
        w_z = get_fp8("linear_attn.in_proj_z")      # [6144, 5120]
        w_a = get("linear_attn.in_proj_a.weight")   # [48, 5120]
        w_b = get("linear_attn.in_proj_b.weight")   # [48, 5120]
        conv_w = get("linear_attn.conv1d.weight").squeeze(1)  # [10240, 4]
        A_log = get("linear_attn.A_log").float()    # [48]
        dt_bias = get("linear_attn.dt_bias").float()# [48]
        gdn_norm_w = get("linear_attn.norm.weight") # [128]
        w_out = get_fp8("linear_attn.out_proj")     # [5120, 6144]
        w_gate = get_fp8("mlp.gate_proj")           # [17408, 5120]
        w_up = get_fp8("mlp.up_proj")               # [17408, 5120]
        w_down = get_fp8("mlp.down_proj")           # [5120, 17408]

    T = x_in.shape[0]
    residual = x_in.clone()
    h = gemma_rmsnorm(residual, in_norm_w)

    mixed_qkv = F.linear(h, w_qkv)  # [T, 10240]
    z = F.linear(h, w_z).view(T, 48, 128)
    a = F.linear(h, w_a).float()    # [T, 48]
    b = F.linear(h, w_b).float()    # [T, 48]

    # Causal depthwise conv1d k=4 + SiLU
    # Pad left by 3 zeros
    x_conv = F.pad(mixed_qkv.float().T.unsqueeze(0), (3, 0))  # [1, 10240, T+3]
    y_conv = F.conv1d(x_conv, conv_w.float().unsqueeze(1), groups=10240).squeeze(0).T  # [T, 10240]
    conv_state = mixed_qkv[-3:].T.contiguous() if T >= 3 else F.pad(mixed_qkv.T, (3 - T, 0))
    y_conv = F.silu(y_conv).to(torch.bfloat16)

    q = y_conv[:, :2048].view(T, 16, 128).float()
    k = y_conv[:, 2048:4096].view(T, 16, 128).float()
    v = y_conv[:, 4096:].view(T, 48, 128).float()

    # Gated DeltaNet exact recurrence (matching SGLang fused_recurrent.py!)
    scale = 128.0 ** -0.5
    S = torch.zeros(48, 128, 128, dtype=torch.float32)  # [HV, V, K]
    outs = torch.empty(T, 48, 128, dtype=torch.bfloat16)
    for t in range(T):
        g = -torch.exp(A_log) * F.softplus(a[t] + dt_bias)  # [48]
        beta = torch.sigmoid(b[t])                          # [48]
        qt = q[t] / torch.sqrt((q[t] * q[t]).sum(dim=-1, keepdim=True) + 1e-6) * scale  # [16, 128]
        kt = k[t] / torch.sqrt((k[t] * k[t]).sum(dim=-1, keepdim=True) + 1e-6)          # [16, 128]
        # hk = hv // 3
        qt_exp = qt.repeat_interleave(3, dim=0)  # [48, 128]
        kt_exp = kt.repeat_interleave(3, dim=0)  # [48, 128]
        vt = v[t]  # [48, 128]
        S = S * torch.exp(g)[:, None, None]
        kv_mem = torch.einsum("hvk,hk->hv", S, kt_exp)
        delta = (vt - kv_mem) * beta[:, None]
        S = S + delta[:, :, None] * kt_exp[:, None, :]
        ot = torch.einsum("hvk,hk->hv", S, qt_exp)
        outs[t] = ot.to(torch.bfloat16)

    gated = rmsnorm_gated_swish(outs, z, gdn_norm_w).view(T, 6144)
    attn_out = F.linear(gated, w_out)

    residual = (residual.float() + attn_out.float()).to(torch.bfloat16)
    h2 = gemma_rmsnorm(residual, post_norm_w)
    mlp_mid = (F.silu(F.linear(h2, w_gate).float()) * F.linear(h2, w_up).float()).to(torch.bfloat16)
    mlp_out = F.linear(mlp_mid, w_down)
    out = (residual.float() + mlp_out.float()).to(torch.bfloat16)
    return out, S.to(torch.bfloat16), conv_state.to(torch.bfloat16)


def ref_full_attn_layer(layer_path: str, layer_idx: int, x_in: torch.Tensor):
    """Exact pure-PyTorch reference for Qwen3.8 full-attention layer."""
    pref = f"model.language_model.layers.{layer_idx}."
    with safe_open(layer_path, framework="pt", device="cpu") as f:
        def get(name):
            return f.get_tensor(pref + name)
        def get_fp8(name):
            return dequant_fp8_block128(get(name + ".weight"), get(name + ".weight_scale_inv"))

        in_norm_w = get("input_layernorm.weight")
        post_norm_w = get("post_attention_layernorm.weight")
        w_q = get_fp8("self_attn.q_proj")   # [12288, 5120]
        w_k = get_fp8("self_attn.k_proj")   # [1024, 5120]
        w_v = get_fp8("self_attn.v_proj")   # [1024, 5120]
        w_o = get_fp8("self_attn.o_proj")   # [5120, 6144]
        q_norm_w = get("self_attn.q_norm.weight")
        k_norm_w = get("self_attn.k_norm.weight")
        w_gate = get_fp8("mlp.gate_proj")
        w_up = get_fp8("mlp.up_proj")
        w_down = get_fp8("mlp.down_proj")

    T = x_in.shape[0]
    positions = torch.arange(T, dtype=torch.int64)
    residual = x_in.clone()
    h = gemma_rmsnorm(residual, in_norm_w)

    q_gate = F.linear(h, w_q).view(T, 24, 512)
    q, gate = torch.chunk(q_gate, 2, dim=-1)  # each [T, 24, 256]
    k = F.linear(h, w_k).view(T, 4, 256)
    v = F.linear(h, w_v).view(T, 4, 256)

    q = gemma_rmsnorm(q, q_norm_w)
    k = gemma_rmsnorm(k, k_norm_w)
    q = apply_neox_partial_rope(q, positions, rotary_dim=64, theta=1e7)
    k = apply_neox_partial_rope(k, positions, rotary_dim=64, theta=1e7)

    # Simulate FP8 E4M3 KV cache (scale = 1.0, matching SGLang!)
    k_fp8 = k.to(torch.float8_e4m3fn).to(torch.float32)
    v_fp8 = v.to(torch.float8_e4m3fn).to(torch.float32)

    # Causal GQA 24:4 attention
    k_rep = k_fp8.repeat_interleave(6, dim=1)  # [T, 24, 256]
    v_rep = v_fp8.repeat_interleave(6, dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), k_rep) * (256.0 ** -0.5)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask[None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    attn = torch.einsum("hts,shd->thd", probs, v_rep)  # [T, 24, 256]
    attn = (attn * torch.sigmoid(gate.float())).to(torch.bfloat16).reshape(T, 6144)

    attn_out = F.linear(attn, w_o)
    residual = (residual.float() + attn_out.float()).to(torch.bfloat16)
    h2 = gemma_rmsnorm(residual, post_norm_w)
    mlp_mid = (F.silu(F.linear(h2, w_gate).float()) * F.linear(h2, w_up).float()).to(torch.bfloat16)
    mlp_out = F.linear(mlp_mid, w_down)
    out = (residual.float() + mlp_out.float()).to(torch.bfloat16)
    return out


def save_bf16(path: str, t: torch.Tensor):
    raw = t.contiguous().to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", default="ref/stage1")
    ap.add_argument("--seq-len", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(12345)
    # Grab real embeddings for deterministic token ids!
    base_ids = [151644, 872, 198, 3838, 374, 220, 17, 489, 220, 17, 30, 151645, 198, 151644, 77091, 198]
    tok_ids = [base_ids[i % len(base_ids)] + (i // len(base_ids)) for i in range(args.seq_len)]
    with safe_open(f"{args.model_dir}/outside.safetensors", framework="pt", device="cpu") as f:
        emb = f.get_tensor("model.language_model.embed_tokens.weight")
        x_in = emb[tok_ids].clone().to(torch.bfloat16)

    save_bf16(f"{args.out_dir}/x_in.bf16", x_in)
    with open(f"{args.out_dir}/tok_ids.bin", "wb") as f:
        f.write(struct.pack(f"<{len(tok_ids)}i", *tok_ids))

    print(f"Running pure-PyTorch reference for Layer 0 (GDN), T={args.seq_len}...", flush=True)
    l0_out, l0_S, l0_conv = ref_gdn_layer(f"{args.model_dir}/layers-0.safetensors", 0, x_in)
    save_bf16(f"{args.out_dir}/layer0_out.bf16", l0_out)
    save_bf16(f"{args.out_dir}/layer0_state.bf16", l0_S)
    save_bf16(f"{args.out_dir}/layer0_conv.bf16", l0_conv)
    print(f"  layer0_out: mean={l0_out.float().mean():.5f} std={l0_out.float().std():.5f} max={l0_out.float().abs().max():.5f}")

    print(f"Running pure-PyTorch reference for Layer 3 (Full-Attn), T={args.seq_len}...", flush=True)
    l3_out = ref_full_attn_layer(f"{args.model_dir}/layers-3.safetensors", 3, l0_out)
    save_bf16(f"{args.out_dir}/layer3_out.bf16", l3_out)
    print(f"  layer3_out: mean={l3_out.float().mean():.5f} std={l3_out.float().std():.5f} max={l3_out.float().abs().max():.5f}")
    print("Stage-1 golden vectors saved to", args.out_dir)


if __name__ == "__main__":
    main()
