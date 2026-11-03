# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""SM100 GPT-OSS BF16 QKV projection, NeoX RoPE and NHD KV-cache write.

The GEMM rounds to BF16 before RoPE, matching separate projection and rotation.
Its retired operand shared memory exchanges the two halves of each head.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import TgvGemmCuteExtKernel


class _GptOssQkvRopeKernel(TgvGemmCuteExtKernel):
    """Reuse TGV's MMA pipeline and fuse its rounded-output consumer."""

    @cute.experimental.jit
    def run_epilog(
        self,
        bar_tma_epilog,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        gBias_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        scratch: cute.Tensor,
        output: cute.Tensor,
        epilogue_args: tuple,
    ):
        pos, cs, kc, vc, loc = epilogue_args
        # Metadata is independent of GEMM, but may be produced by a prior
        # kernel. PDL wait must precede these reads. Prefetch before MMA wait.
        # Keep explicit arch.load calls: tensor indexing regresses this prefetch
        # path at 32/64 tokens on SM100.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        rCos = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute.make_layout(self.cta_n // 4),
            alignment=16,
        )
        rSin = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute.make_layout(self.cta_n // 4),
            alignment=16,
        )
        rSlot = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.rmem,
            cute.make_layout(self.cta_n // 4),
            alignment=16,
        )
        bx, by, bz = cute.arch.block_idx()
        for i in cutlass.range_constexpr(self.cta_n // 4):
            tok = by * self.cta_n + epi_tid // 32 + i * 4
            rSlot[i] = cute.arch.load(loc.iterator + tok, cutlass.Int64)
            rCos[i] = cutlass.Float32(0)
            rSin[i] = cutlass.Float32(0)
            if bx < 72:
                pp = cute.arch.load(pos.iterator + tok, cutlass.Int64)
                rCos[i] = cute.arch.load(
                    cs.iterator + pp * 64 + epi_tid % 32, cutlass.Float32
                )
                rSin[i] = cute.arch.load(
                    cs.iterator + pp * 64 + epi_tid % 32 + 32, cutlass.Float32
                )
        exchange = cute.make_tensor(
            cute.recast_ptr(scratch.iterator, swizzle_=cute.make_swizzle(0, 4, 3)),
            cute.make_layout((64, self.cta_n), stride=(1, 64)),
        )
        # The MMA completion wait in epilog_warp also retires operand reads,
        # so scratch can now hold rounded BF16 values without a new allocation.
        self.epilog_warp(
            bar_tma_epilog,
            bar_mma_epilog,
            bar_tmem_alloc,
            tmem_base_ptr,
            acc_layout,
            gD_tile,
            gBias_tile,
            epi_tid,
            c_dtype,
            d_layout,
            exchange,
        )
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        self.rope_store(output, kc, vc, epi_tid, exchange, rCos, rSin, rSlot)

    @cute.experimental.jit
    def rope_store(
        self,
        out: cute.Tensor,
        kc: cute.Tensor,
        vc: cute.Tensor,
        tid: cutlass.Int32,
        exchange: cute.Tensor,
        rCos: cute.Tensor,
        rSin: cute.Tensor,
        rSlot: cute.Tensor,
    ):
        bx, by, bz = cute.arch.block_idx()
        for i in cutlass.range_constexpr(self.cta_n * 32 // 128):
            idx = tid + i * 128
            tok = by * self.cta_n + idx // 32
            pair = idx % 32
            feat = bx * 64 + pair
            if tok < out.shape[1]:
                x = exchange[pair, idx // 32].to(cutlass.Float32)
                y = exchange[pair + 32, idx // 32].to(cutlass.Float32)
                slot = rSlot[i]
                if feat < 4608:
                    c = rCos[i]
                    sn = rSin[i]
                    # Match native RoPE: round the sine product before the FMA.
                    rounding = cute.math.RoundingMode.NEAREST_EVEN
                    ys = cute.math.mul(y, sn, rounding=rounding)
                    xs = cute.math.mul(x, sn, rounding=rounding)
                    rx = cute.math.fma(x, c, -ys, rounding=rounding).to(
                        cutlass.BFloat16
                    )
                    ry = cute.math.fma(y, c, xs, rounding=rounding).to(cutlass.BFloat16)
                    out[feat, tok, 0] = rx
                    out[feat + 32, tok, 0] = ry
                    if feat >= 4096:
                        if slot > 0 and slot < kc.shape[0]:
                            kc[slot, feat - 4096] = rx
                            kc[slot, feat - 4096 + 32] = ry
                else:
                    out[feat, tok, 0] = x.to(cutlass.BFloat16)
                    out[feat + 32, tok, 0] = y.to(cutlass.BFloat16)
                    if slot > 0 and slot < vc.shape[0]:
                        vc[slot, feat - 4608] = x.to(cutlass.BFloat16)
                        vc[slot, feat - 4608 + 32] = y.to(cutlass.BFloat16)


_COMPILE_CACHE = {}


def gpt_oss_qkv_rope_store(
    x, weight, bias, positions, cos_sin_cache, k_cache, v_cache, cache_loc, out=None
):
    """Project and rotate 16/32/64 tokens; write BF16 NHD KV, skipping slot 0.

    Requires SM100, hidden width 2880, 64 Q heads and 8 KV heads of width 64.
    Positions must index cos_sin_cache; positive cache locations must be unique.
    Cache locations must index the allocation (slot zero is reserved padding).
    """
    n = x.shape[0]
    if (
        x.shape != (n, 2880)
        or n not in (16, 32, 64)
        or weight.shape != (5120, 2880)
        or bias.shape != (5120,)
        or positions.shape != (n,)
        or cache_loc.shape != (n,)
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != 64
        or k_cache.shape != v_cache.shape
        or tuple(k_cache.shape[1:]) not in ((512,), (8, 64))
    ):
        raise ValueError("unsupported GPT-OSS QKV/RoPE shapes")
    bf16 = (x, weight, bias, k_cache, v_cache)
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in bf16):
        raise ValueError("projection and KV tensors must be contiguous BF16")
    if (
        positions.dtype != torch.int64
        or cache_loc.dtype != torch.int64
        or cos_sin_cache.dtype != torch.float32
        or any(not t.is_contiguous() for t in (positions, cache_loc, cos_sin_cache))
    ):
        raise ValueError("RoPE requires contiguous int64 indices and FP32 cos/sin")
    if (
        not x.is_cuda
        or any(
            t.device != x.device for t in (*bf16, positions, cos_sin_cache, cache_loc)
        )
        or torch.cuda.get_device_capability(x.device) != (10, 0)
    ):
        raise ValueError("GPT-OSS QKV/RoPE requires tensors on one SM100 device")
    if out is None:
        out = torch.empty((n, 5120), device=x.device, dtype=x.dtype)
    if (
        out.shape != (n, 5120)
        or out.dtype != x.dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous BF16 [tokens, 5120]")
    tensors = [
        weight.unsqueeze(-1),
        x.unsqueeze(-1),
        out.T.unsqueeze(-1),
        bias.detach().as_strided((5120, n, 1), (1, 0, 0)),
        positions,
        cos_sin_cache,
        k_cache.view(-1, 512),
        v_cache.view(-1, 512),
        cache_loc,
    ]
    args = [from_dlpack(t.detach(), assumed_align=2) for t in tensors]
    key = (
        x.device,
        n,
        tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    if key not in _COMPILE_CACHE:
        op = _GptOssQkvRopeKernel(
            acc_dtype=cutlass.Float32,
            cta_m=64,
            cta_n=n,
            cta_k=128,
            num_ab_stage={16: 12, 32: 11, 64: 9}[n],
            use_2cta=True,
            use_pdl=True,
            has_bias=True,
            out_dtype=cutlass.BFloat16,
        )
        _COMPILE_CACHE[key] = cute_ext.compile(
            op, *args[:4], make_fake_stream(), tuple(args[4:])
        )
    _COMPILE_CACHE[key](*args[:4], stream, tuple(args[4:]))
    return out
