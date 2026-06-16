# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integration test: flashmask_attention(..., sink=sink).backward() routes
through the new ``FlashMaskFunc`` sink path and produces correct ``sink.grad``.

This complements ``test_flashmask_sink_bwd.py`` (kernel-level) by exercising
the public API exposed via ``flashmask_attention`` (and therefore the
paddlefleet_ops facade as well).
"""

import math
import unittest

import numpy as np
import paddle


def _is_fa4_supported() -> bool:
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        cap = paddle.device.cuda.get_device_capability()
    except Exception:
        return False
    if cap[0] != 10:
        return False
    try:
        from flash_mask.cute.interface import flashmask_attention  # noqa: F401
    except Exception:
        try:
            from paddlefleet_ops.flash_mask.cute.interface import (  # noqa: F401
                flashmask_attention,
            )
        except Exception:
            return False
    return True


if not _is_fa4_supported():
    raise unittest.SkipTest(
        "flashmask_attention(..., sink=...) test requires Blackwell GPU (SM100) "
        "with the flash_mask.cute kernels available."
    )


try:
    from flash_mask.cute.interface import flashmask_attention
except Exception:
    from paddlefleet_ops.flash_mask.cute.interface import flashmask_attention

# Force FA4 dispatch (the FA2 fallback does not support sink and flashmask v4
# is the only backend that ships the fused dsink kernel).
paddle.set_flags({"FLAGS_flash_attn_version": 4})


def _sink_attention_fp32_reference(q, k, v, sink, dout, causal):
    q32 = q.astype("float32").detach()
    k32 = k.astype("float32").detach()
    v32 = v.astype("float32").detach()
    sink32 = sink.astype("float32").detach()
    dout32 = dout.astype("float32").detach()
    sink32.stop_gradient = False

    B, S, H, D = q32.shape
    scale = 1.0 / math.sqrt(D)
    qt = q32.transpose([0, 2, 1, 3])
    kt = k32.transpose([0, 2, 1, 3])
    vt = v32.transpose([0, 2, 1, 3])
    scores = paddle.matmul(qt, kt, transpose_y=True) * scale
    if causal:
        mask = paddle.tril(paddle.ones([S, S], dtype="bool"))
        scores = paddle.where(mask, scores, paddle.full_like(scores, -1e30))
    sink_logit = sink32.reshape([1, H, 1, 1]).expand([B, H, S, 1])
    scores_aug = paddle.concat([scores, sink_logit], axis=-1)
    probs_aug = paddle.nn.functional.softmax(scores_aug, axis=-1)
    probs = probs_aug[..., :-1]
    out = paddle.matmul(probs, vt).transpose([0, 2, 1, 3])
    (out * dout32).sum().backward()
    return sink32.grad.detach()


class TestFlashMaskAttentionSinkAPI(unittest.TestCase):
    def _run(self, b, s, h, d, causal):
        paddle.seed(7)
        np.random.seed(7)
        dt = paddle.bfloat16
        q = (paddle.randn([b, s, h, d], dtype=dt) * 0.5)
        k = (paddle.randn([b, s, h, d], dtype=dt) * 0.5)
        v = (paddle.randn([b, s, h, d], dtype=dt) * 0.5)
        sink = (paddle.randn([h], dtype=dt) * 0.5)
        dout = paddle.randn([b, s, h, d], dtype=dt) * 0.5

        q.stop_gradient = False
        k.stop_gradient = False
        v.stop_gradient = False
        sink.stop_gradient = False

        out = flashmask_attention(
            q, k, v,
            startend_row_indices=None,
            causal=causal,
            sink=sink,
        )
        (out * dout).sum().backward()
        self.assertIsNotNone(sink.grad)
        dsink_fused = sink.grad.astype("float32").numpy()

        dsink_ref = _sink_attention_fp32_reference(
            q, k, v, sink, dout, causal
        ).numpy()

        atol = max(1e-2, 5e-3 * np.abs(dsink_ref).max())
        np.testing.assert_allclose(
            dsink_fused, dsink_ref,
            rtol=2e-2, atol=atol,
            err_msg=f"sink.grad mismatch b={b} s={s} h={h} d={d} causal={causal}",
        )

    def test_causal(self):
        self._run(2, 512, 8, 128, causal=True)

    def test_noncausal(self):
        self._run(1, 256, 4, 64, causal=False)

    def test_stop_gradient_sink_returns_3tuple(self):
        # When sink.stop_gradient is True, FlashMaskFunc.backward must NOT try
        # to allocate a sink grad. The PyLayer should behave identically to the
        # pre-existing 3-tuple path.
        paddle.seed(0)
        b, s, h, d = 1, 128, 4, 64
        dt = paddle.bfloat16
        q = paddle.randn([b, s, h, d], dtype=dt) * 0.5
        k = paddle.randn([b, s, h, d], dtype=dt) * 0.5
        v = paddle.randn([b, s, h, d], dtype=dt) * 0.5
        sink = paddle.randn([h], dtype=dt) * 0.5
        dout = paddle.randn([b, s, h, d], dtype=dt) * 0.5
        q.stop_gradient = False
        k.stop_gradient = False
        v.stop_gradient = False
        sink.stop_gradient = True
        out = flashmask_attention(q, k, v, causal=True, sink=sink)
        (out * dout).sum().backward()
        self.assertIsNone(sink.grad)
        self.assertIsNotNone(q.grad)


if __name__ == "__main__":
    unittest.main()
