"""trtllm_mha's paged K/V views keep the pool's slot stride.

``_reshape_paged_kv_cache`` turns a per-layer ``[slots, ...]`` buffer into the
``[pages, heads, page_size, head_dim]`` layout the trtllm kernels take. Its
nvfp4 callers also hand it block-scale rows that arrive FLAT
(``[slots, heads * head_dim // 16]``), so the row has to be shaped before the
pages are split: shaping it afterwards, or splitting with ``.view()``,
re-derives the size-1 page dimension's stride from the row at
``page_size == 1``. On a strided buffer (the unified pool's slots are a whole
entry apart, wider than one row) that stride is then wrong.

An HND pool hands over page-major ``[pages, heads, page_size, head_dim]``
buffers instead, which are already the kernel layout: they must reach the
kernel as themselves, head for head and token for token.

CPU-only.

    python -m pytest test/registered/unit/layers/attention/test_trtllm_mha_paged_kv.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _strided_rows(num_slots, row_shape, slot_stride):
    # Distinct values, so equal values mean equal addresses.
    backing = torch.arange(num_slots * slot_stride, dtype=torch.float32)
    return backing.as_strided(
        (num_slots, *row_shape), (slot_stride, *torch.empty(row_shape).stride())
    )


class TestTRTLLMHAPagedKV(CustomTestCase):
    def test_reshape_keeps_slot_stride_for_packed_and_flat_rows(self):
        H, D, pages = 4, 32, 3
        slot_stride = 3 * H * D
        layer = SimpleNamespace(tp_k_head_num=H, tp_v_head_num=H)
        # 16 is the smallest page size trtllm_mha runs at; at 1, `.view()`
        # would give the size-1 page dim the row stride.
        for page_size in (1, 16):
            backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
            backend.page_size = page_size
            # (row shape handed in, head_dim argument): nvfp4 packed K/V rows,
            # then the flat block-scale rows.
            for row_shape, head_dim in (
                ((H, D // 2), D // 2),
                ((H * D // 16,), D // 16),
            ):
                with self.subTest(page_size=page_size, row_shape=row_shape):
                    k = _strided_rows(pages * page_size, row_shape, slot_stride)
                    v = _strided_rows(pages * page_size, row_shape, slot_stride)
                    outs = backend._reshape_paged_kv_cache(k, v, layer, head_dim)
                    for flat, out in zip((k, v), outs):
                        self.assertEqual(
                            tuple(out.shape), (pages, H, page_size, head_dim)
                        )
                        self.assertEqual(out.stride(0), page_size * slot_stride)
                        self.assertEqual(out.stride(2), slot_stride)
                        for token in range(pages * page_size):
                            page, slot = divmod(token, page_size)
                            self.assertTrue(
                                torch.equal(
                                    out[page, :, slot],
                                    flat[token].reshape(H, head_dim),
                                )
                            )

    def test_reshape_hands_the_hnd_pool_buffer_through(self):
        H, D, pages, page_size = 4, 32, 3, 16
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.page_size = page_size
        layer = SimpleNamespace(tp_k_head_num=H, tp_v_head_num=H)
        # Distinct values, so equal values mean equal (page, head, token).
        k = torch.arange(pages * H * page_size * D, dtype=torch.float32).view(
            pages, H, page_size, D
        )
        v = k + 0.5
        outs = backend._reshape_paged_kv_cache(k, v, layer, D)
        for buf, out in zip((k, v), outs):
            self.assertEqual(out.shape, buf.shape)
            self.assertEqual(out.stride(), buf.stride())
            self.assertEqual(out.data_ptr(), buf.data_ptr())
            self.assertTrue(torch.equal(out, buf))

    def test_fused_store_keeps_snapshot_after_allocator_mapping_changes(self):
        """Reusing the live mapping after the WAR fence would write a new slot."""
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        mapping = torch.tensor([0, 9, 7, 5, 3])
        full_loc = torch.tensor([1, 2, 0])
        snapshot = mapping[full_loc].clone()
        backend._swa_kv_pool = SimpleNamespace(
            layers_mapping={0: (0, False), 1: (0, True)},
            full_kv_pool=SimpleNamespace(use_hnd=False),
            swa_kv_pool=SimpleNamespace(use_hnd=False),
            full_to_swa_index_mapping=mapping,
        )
        backend.forward_metadata = SimpleNamespace(swa_out_cache_loc=snapshot)
        fb = SimpleNamespace(out_cache_loc=full_loc, positions=torch.arange(3))
        full, swa = SimpleNamespace(layer_id=0), SimpleNamespace(layer_id=1)
        mapping[1:3] = torch.tensor([4, 2])
        loc = backend.get_fused_set_kv_buffer_cache_loc(swa, fb)
        self.assertEqual(loc.tolist(), [9, 7, 0])
        self.assertEqual(loc.data_ptr(), snapshot.data_ptr())
        self.assertIs(backend.get_fused_set_kv_buffer_cache_loc(full, fb), full_loc)

        # Captured decode must consume updated contents at the same address.
        snapshot.copy_(torch.tensor([6, 8, 0]))
        self.assertEqual(loc.tolist(), [6, 8, 0])
        fb.out_cache_loc = full_loc[:2]
        self.assertEqual(
            backend.get_fused_set_kv_buffer_cache_loc(swa, fb).tolist(), [6, 8]
        )

    def test_fused_swa_falls_back_across_modes_and_unsupported_layouts(self):
        """Wrapper dispatch must not opt unsupported prefill/graph paths in."""
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
        from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            enable_breakable_cuda_graph,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            enable_tc_piecewise_cuda_graph,
        )
        from sglang.srt.models import utils

        pool = SWAKVPool.__new__(SWAKVPool)
        pool.dtype = torch.bfloat16
        pool.layers_mapping = {0: (0, True)}
        pool.swa_kv_pool = SimpleNamespace(use_hnd=False)
        pool.full_kv_pool = SimpleNamespace(use_hnd=False)
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend._swa_kv_pool = pool
        backend.forward_metadata = SimpleNamespace(
            swa_out_cache_loc=torch.tensor([5, 2, 0])
        )
        hybrid = HybridAttnBackend.__new__(HybridAttnBackend)
        hybrid.decode_backend = backend
        hybrid.prefill_backend = AttentionBackend()
        tbo = TboAttnBackend.__new__(TboAttnBackend)
        tbo.primary = hybrid
        tbo.token_to_kv_pool = pool
        fb = SimpleNamespace(
            out_cache_loc=torch.tensor([1, 2, 0]),
            positions=torch.arange(3),
            forward_mode=ForwardMode.DECODE,
            dcp_kv_mask=None,
        )
        layer = SimpleNamespace(layer_id=0)
        with (
            forward_context(ForwardContext(attn_backend=tbo)),
            patch.object(utils, "_is_cuda", True),
            patch.object(utils, "_is_hip", False),
        ):
            self.assertTrue(utils.enable_fused_set_kv_buffer(fb, layer))
            self.assertFalse(utils.enable_fused_set_kv_buffer(fb))
            for graph_context in (
                enable_tc_piecewise_cuda_graph,
                enable_breakable_cuda_graph,
            ):
                with graph_context():
                    self.assertFalse(utils.enable_fused_set_kv_buffer(fb, layer))
            fb.forward_mode = ForwardMode.EXTEND
            self.assertFalse(utils.enable_fused_set_kv_buffer(fb, layer))
            fb.forward_mode = ForwardMode.DECODE
            pool.swa_kv_pool.use_hnd = True
            self.assertFalse(utils.enable_fused_set_kv_buffer(fb, layer))
            pool.swa_kv_pool.use_hnd = False
            fb.dcp_kv_mask = torch.ones(3, dtype=torch.bool)
            self.assertFalse(utils.enable_fused_set_kv_buffer(fb, layer))
            fb.dcp_kv_mask = None
            backend.forward_metadata.swa_out_cache_loc = None
            self.assertFalse(utils.enable_fused_set_kv_buffer(fb, layer))

    def test_gpt_oss_native_rope_and_prepare_decision_preserve_kv_writes(self):
        """Native RoPE cannot store KV; core must honor what prepare executed."""
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.models import gpt_oss, utils

        class Projection(torch.nn.Module):
            def forward(self, x):
                return x, None

        class CacheAttention(torch.nn.Module):
            layer_id = 0

            def __init__(self):
                super().__init__()
                self.cache = torch.zeros(2, 64)

            def forward(self, q, k, v, fb, *, sinks, save_kv_cache):
                if save_kv_cache:
                    self.cache.copy_(k)
                return self.cache.clone()

        pool = SWAKVPool.__new__(SWAKVPool)
        pool.dtype = torch.bfloat16
        pool.layers_mapping = {0: (0, True)}
        pool.swa_kv_pool = SimpleNamespace(use_hnd=False)
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.token_to_kv_pool = backend._swa_kv_pool = pool
        snapshot = torch.tensor([2, 1])
        fb = SimpleNamespace(
            out_cache_loc=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            forward_mode=ForwardMode.DECODE,
            dcp_kv_mask=None,
        )
        attention = gpt_oss.GptOssAttention.__new__(gpt_oss.GptOssAttention)
        torch.nn.Module.__init__(attention)
        attention.q_size = attention.kv_size = 64
        attention.qkv_proj = attention.o_proj = Projection()
        attention.attn = CacheAttention()
        attention.sinks = None
        attention.rotary_emb = RotaryEmbedding(64, 64, 8, 10000, True, torch.float32)
        # Exercise real CPU RoPE math; GPU cache writes are the mocked boundary.
        attention.rotary_emb._forward_method = attention.rotary_emb.forward_native
        x = torch.arange(2 * 192, dtype=torch.float32).reshape(2, 192) / 100
        with (
            forward_context(ForwardContext(attn_backend=backend)),
            patch.object(utils, "_is_cuda", True),
            patch.object(utils, "_is_hip", False),
            patch.object(gpt_oss, "_is_cuda", True),
            patch.object(gpt_oss, "_is_npu", False),
        ):
            for mode in ("native", "fallback", "snapshot_appears_after_prepare"):
                with self.subTest(mode=mode):
                    attention.attn.cache.zero_()
                    attention.rotary_emb._force_native = mode == "native"
                    attention.rotary_emb.use_fallback_kernel = mode == "fallback"
                    backend.forward_metadata = SimpleNamespace(
                        swa_out_cache_loc=(
                            None
                            if mode == "snapshot_appears_after_prepare"
                            else snapshot
                        )
                    )
                    prepared = attention.forward_prepare(fb.positions, x.clone(), fb)
                    # A fresh availability decision here would incorrectly skip
                    # the separate store when prepare had no usable snapshot.
                    backend.forward_metadata.swa_out_cache_loc = snapshot
                    expected_k = prepared[2][1].clone()
                    output = attention.forward_core(prepared)
                    self.assertTrue(torch.equal(output, expected_k))
                    self.assertTrue(torch.equal(attention.attn.cache, expected_k))


if __name__ == "__main__":
    unittest.main()
