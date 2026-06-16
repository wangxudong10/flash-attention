# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for fused dsink computation in the FlashMask v4 (sm100) bwd.

We validate the new ``learnable_sink`` output of ``_flash_attn_bwd`` against:

1. The previous tilelang-based reference (``flashattn_bwd_dsink`` from
   ``sink_impl`` prior to the fusion). We re-implement that exact formula
   in pure paddle to avoid a tilelang dependency.

2. A plain ``float32`` eager autograd reference that runs the full sink
   attention forward in fp32 and reads ``sink.grad``.
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
        from flash_mask.cute.interface import (  # noqa: F401
            _flash_attn_bwd,
            _flash_attn_fwd,
        )
        from flash_mask.cute.flash_bwd_dsink import (  # noqa: F401
            FlashAttentionBackwardDsink,
        )
    except Exception:
        try:
            import paddlefleet_ops  # noqa: F401  (registers flash_mask alias)
            from paddlefleet_ops.flash_mask.cute.interface import (  # noqa: F401
                _flash_attn_bwd,
                _flash_attn_fwd,
            )
            from paddlefleet_ops.flash_mask.cute.flash_bwd_dsink import (  # noqa: F401
                FlashAttentionBackwardDsink,
            )
        except Exception:
            return False
    return True


if not _is_fa4_supported():
    raise unittest.SkipTest(
        "FlashMask v4 (FA4) dsink fusion test requires Blackwell GPU (SM100) "
        "with the flash_mask.cute kernels available."
    )


try:
    from flash_mask.cute.interface import _flash_attn_bwd, _flash_attn_fwd
except Exception:
    import paddlefleet_ops  # noqa: F401
    from paddlefleet_ops.flash_mask.cute.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
    )


# ----------------------- references --------------------------------------- #


def _dsink_reference_from_lse_delta(sink, lse, delta):
    """Mirror of the legacy tilelang ``flashattn_bwd_dsink`` kernel, in paddle.

    Args:
        sink: ``[H]`` (any float dtype)
        lse:  ``[B, H, S]`` float32 (natural log)
        delta:``[B, H, S]`` float32, ``= sum(O*dO, dim=-1)``
    Returns:
        ``[H]`` float32 dsink.
    """
    sink_fp32 = sink.astype("float32")
    # Legacy tilelang formula: contrib = -exp2(sink * log2_e - lse * log2_e) * delta
    #                                  = -exp(sink - lse) * delta
    sink_b = sink_fp32.reshape([1, -1, 1])
    contrib = -paddle.exp(sink_b - lse) * delta  # [B, H, S]
    return contrib.sum(axis=[0, 2])


def _sink_attention_fp32_reference(q, k, v, sink, dout, causal):
    """Full-precision eager forward + autograd backward for sink attention.

    Layout: ``[B, S, H, D]``. Returns dsink as float32 ``[H]``.
    """
    q32 = q.astype("float32").detach()
    k32 = k.astype("float32").detach()
    v32 = v.astype("float32").detach()
    sink32 = sink.astype("float32").detach()
    dout32 = dout.astype("float32").detach()
    q32.stop_gradient = False
    k32.stop_gradient = False
    v32.stop_gradient = False
    sink32.stop_gradient = False

    B, S, H, D = q32.shape
    scale = 1.0 / math.sqrt(D)
    # [B, H, S, D]
    qt = q32.transpose([0, 2, 1, 3])
    kt = k32.transpose([0, 2, 1, 3])
    vt = v32.transpose([0, 2, 1, 3])

    scores = paddle.matmul(qt, kt, transpose_y=True) * scale  # [B,H,S,S]
    if causal:
        mask = paddle.tril(
            paddle.ones([S, S], dtype="bool")
        )  # True = keep
        scores = paddle.where(
            mask, scores, paddle.full_like(scores, -1e30)
        )

    # softmax with sink: append sink logit per head as extra column.
    # softmax over (S + 1) where last logit = sink[h] (broadcast over rows).
    sink_logit = sink32.reshape([1, H, 1, 1]).expand([B, H, S, 1])
    scores_aug = paddle.concat([scores, sink_logit], axis=-1)
    probs_aug = paddle.nn.functional.softmax(scores_aug, axis=-1)
    probs = probs_aug[..., :-1]  # drop the sink column for the value mixing
    out = paddle.matmul(probs, vt)  # [B,H,S,D]
    out = out.transpose([0, 2, 1, 3])  # [B,S,H,D]

    loss = (out * dout32).sum()
    loss.backward()
    return sink32.grad.detach()


# ------------------------- core test runner ------------------------------- #


def _run_case(b, s, h, d, *, causal, sink_init="randn"):
    paddle.seed(123)
    np.random.seed(123)

    dtype = paddle.bfloat16
    q = paddle.randn([b, s, h, d], dtype=dtype) * 0.5
    k = paddle.randn([b, s, h, d], dtype=dtype) * 0.5
    v = paddle.randn([b, s, h, d], dtype=dtype) * 0.5
    if sink_init == "randn":
        sink = (paddle.randn([h], dtype=dtype) * 0.5)
    else:
        sink = paddle.zeros([h], dtype=dtype)
    dout = paddle.randn([b, s, h, d], dtype=dtype) * 0.5

    # Forward via FA4 to get O, lse
    out, lse = _flash_attn_fwd(
        q, k, v,
        causal=causal,
        return_lse=True,
        learnable_sink=sink,
        pack_gqa=False,
    )

    # Backward (fused dsink path)
    bwd_outputs = _flash_attn_bwd(
        q, k, v,
        out, dout, lse,
        None,
        causal=causal,
        deterministic=False,
        learnable_sink=sink,
    )
    assert len(bwd_outputs) == 4, (
        "_flash_attn_bwd must return 4-tuple when learnable_sink is provided"
    )
    dq, dk, dv, dsink_fused = bwd_outputs
    dsink_fused = dsink_fused.astype("float32").numpy()

    # ---- Reference 1: legacy tilelang formula, evaluated in paddle.
    delta = (out.astype("float32") * dout.astype("float32")).sum(axis=-1)
    delta = delta.transpose([0, 2, 1])  # [B, H, S]  -- matches sink_impl pre-call
    dsink_legacy = (
        _dsink_reference_from_lse_delta(sink, lse, delta).numpy()
    )
    np.testing.assert_allclose(
        dsink_fused,
        dsink_legacy,
        rtol=5e-3,
        atol=5e-3,
        err_msg=(
            f"dsink (fused) != dsink (legacy paddle formula) for "
            f"b={b}, s={s}, h={h}, d={d}, causal={causal}"
        ),
    )

    # ---- Reference 2: full fp32 eager autograd through sink-augmented softmax.
    dsink_fp32 = _sink_attention_fp32_reference(
        q, k, v, sink, dout, causal=causal
    ).numpy()
    # bf16 inputs through fp32 reductions: empirically max_rel is <= ~1e-2 once
    # the reference magnitude is non-trivial (>= O(1)). For tiny-magnitude refs
    # (S < a few hundred) the rtol bound is meaningless and atol is the gate, so
    # we keep a comfortable atol but tighten rtol to 2e-2.
    atol = max(1e-2, 5e-3 * np.abs(dsink_fp32).max())
    np.testing.assert_allclose(
        dsink_fused,
        dsink_fp32,
        rtol=2e-2,
        atol=atol,
        err_msg=(
            f"dsink (fused) != dsink (fp32 eager) for "
            f"b={b}, s={s}, h={h}, d={d}, causal={causal}"
        ),
    )


# ---------------------------- test cases ---------------------------------- #


class TestFlashMaskSinkBackward(unittest.TestCase):
    def test_small_noncausal(self):
        _run_case(1, 256, 4, 64, causal=False)

    def test_small_causal(self):
        _run_case(1, 256, 4, 64, causal=True)

    def test_medium_causal(self):
        _run_case(2, 512, 8, 128, causal=True)

    def test_zero_sink(self):
        # Sanity: with sink == 0, dsink reduces to -sum exp(-lse) * delta;
        # the kernel must still produce results consistent across paths.
        _run_case(1, 512, 4, 64, causal=False, sink_init="zero")

    def test_stop_gradient_skips_dsink(self):
        # When learnable_sink is None, _flash_attn_bwd must keep the legacy
        # 3-tuple return for backward compatibility.
        paddle.seed(0)
        b, s, h, d = 1, 128, 4, 64
        dtype = paddle.bfloat16
        q = paddle.randn([b, s, h, d], dtype=dtype) * 0.5
        k = paddle.randn([b, s, h, d], dtype=dtype) * 0.5
        v = paddle.randn([b, s, h, d], dtype=dtype) * 0.5
        dout = paddle.randn([b, s, h, d], dtype=dtype) * 0.5
        out, lse = _flash_attn_fwd(
            q, k, v, causal=True, return_lse=True, pack_gqa=False
        )
        outputs = _flash_attn_bwd(
            q, k, v, out, dout, lse, None,
            causal=True, deterministic=False,
            learnable_sink=None,
        )
        self.assertEqual(len(outputs), 3)


if __name__ == "__main__":
    unittest.main()
