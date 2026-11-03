"""Protect BF16 rounding, NeoX pairing and KV writes across graph replays."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0),
    "SM100 required",
)
class TestGptOssQkvRope(CustomTestCase):
    def test_projection_rounding_and_graph_cache_updates(self):
        # Removing the intermediate BF16 rounding or reusing stale prefetched
        # positions/cache locations changes observable QKV or cache contents.
        from sglang.kernels.ops.attention.gpt_oss_qkv_rope import (
            gpt_oss_qkv_rope_store,
        )
        from sglang.kernels.ops.attention.rope import (
            apply_rope_inplace_with_kvcache,
        )
        from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm

        torch.manual_seed(123)
        weight = torch.randn(5120, 2880, device="cuda", dtype=torch.bfloat16) * 0.125
        bias = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
        bias.requires_grad_(True)
        phase = torch.arange(16384, device="cuda").float()[:, None] / (
            10000 ** (torch.arange(32, device="cuda").float()[None, :] / 32)
        )
        cos_sin = torch.cat([phase.cos(), phase.sin()], dim=-1)
        with torch.no_grad():
            for n in (16, 32, 64):
                with self.subTest(tokens=n):
                    x = torch.randn(n, 2880, device="cuda", dtype=torch.bfloat16)
                    positions = torch.arange(n, device="cuda")
                    locations = torch.cat(
                        [
                            torch.tensor([0, 127], device="cuda", dtype=torch.int64),
                            torch.randperm(126, device="cuda")[: n - 2] + 1,
                        ]
                    )
                    k = torch.full(
                        (128, 8, 64), -3, device="cuda", dtype=torch.bfloat16
                    )
                    v = k.clone()
                    out = torch.empty(n, 5120, device="cuda", dtype=torch.bfloat16)

                    def fused():
                        return gpt_oss_qkv_rope_store(
                            x, weight, bias, positions, cos_sin, k, v, locations, out
                        )

                    fused()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fused()
                    for replay in (False, True):
                        if replay:
                            positions.add_(8192)
                            locations[1:].copy_(locations[1:].flip(0))
                            x.copy_(torch.randn_like(x))
                        k.fill_(-3)
                        v.fill_(-3)
                        kr, vr = k.clone(), v.clone()
                        ref = cutedsl_bf16_gemm(x, weight, bias)
                        apply_rope_inplace_with_kvcache(
                            ref[:, :4096].view(n, 64, 64),
                            ref[:, 4096:4608].view(n, 8, 64),
                            ref[:, 4608:].view(n, 8, 64),
                            kr.view(128, 512),
                            vr.view(128, 512),
                            cos_sin,
                            positions,
                            locations,
                            is_neox=True,
                            reserved_skip_index=0,
                        )
                        graph.replay() if replay else fused()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(out, ref, rtol=0, atol=0)
                        torch.testing.assert_close(k, kr, rtol=0, atol=0)
                        torch.testing.assert_close(v, vr, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
