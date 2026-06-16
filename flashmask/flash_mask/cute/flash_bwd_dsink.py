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

"""dsink kernel for FlashMask v4 (sm100) backward.

Computes per-head gradient of the learnable sink parameter, fused with the
existing backward `preprocess` outputs (`dpsum`, `lse_log2`).

For each (b, h, s) position the sink contribution to the loss is

    delta[b, h, s]  = sum_v output[b, h, s, v] * grad_output[b, h, s, v]
    dsink_per_token = -exp2(sink[h] * log2_e - lse[b, h, s] * log2_e) * delta

The preprocess kernel already produces:
    mdPsum[b, h, s]   = delta[b, h, s]      (Float32)
    mLSElog2[b, h, s] = lse[b, h, s] * log2_e  (Float32, padded rows = 0)

So this kernel only needs to evaluate the elementwise expression and reduce
over (b, s) into the per-head output `mdSink[h]` (Float32).
"""

import math
import operator
from typing import Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

from flash_mask.cute import utils


@dsl_user_op
def _atomic_add_fp32_asm(a, gmem_ptr, *, loc=None, ip=None) -> None:
    """Single-element fp32 atomic-add via inline PTX.

    The ``utils.atomic_add_fp32`` helper in this build calls
    ``nvvm.atomicrmw(res=...)`` which the installed cutlass-dsl does not
    accept; we replicate the simpler ``red.global.add.f32`` PTX form used
    by ``copy_utils.atomic_add_fp32x4`` for the v4 case.
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip)],
        "red.global.add.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class FlashAttentionBackwardDsink:
    """Reduce ``-exp2(sink * log2_e - lse_log2) * dpsum`` into per-head dsink.

    The kernel launches a 3-D grid ``(ceil_div(seqlen_q_padded, m_block_size),
    num_head, num_batch)`` so each CTA processes a contiguous chunk of
    sequence positions for a single (head, batch) tuple.

    The thread block reduces locally, then atomically adds its partial result
    to ``mdSink[head_idx]``. The output tensor must be zero-initialised by
    the caller.
    """

    def __init__(
        self,
        sink_dtype: Type[cutlass.Numeric],
        m_block_size: int = 128,
        num_threads: int = 128,
    ):
        assert num_threads % cute.arch.WARP_SIZE == 0, (
            "num_threads must be a multiple of warp size"
        )
        assert m_block_size > 0, "m_block_size must be positive"
        # `m_block_size` may exceed `num_threads`; each thread iterates over
        # `m_block_size / num_threads` rows.
        assert m_block_size % num_threads == 0, (
            "m_block_size must be a multiple of num_threads"
        )
        self.sink_dtype = sink_dtype
        self.m_block_size = m_block_size
        self.num_threads = num_threads
        self.rows_per_thread = m_block_size // num_threads
        self.num_warps = num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mSink: cute.Tensor,        # [num_head], sink_dtype
        mLSElog2: cute.Tensor,     # [batch, num_head, seqlen_q_padded], Float32
        mdPsum: cute.Tensor,       # [batch, num_head, seqlen_q_padded], Float32
        mdSink: cute.Tensor,       # [num_head], Float32 (atomically accumulated)
        stream: cuda.CUstream,
    ):
        if const_expr(mSink.element_type != self.sink_dtype):
            raise TypeError("sink dtype mismatch")
        if const_expr(mLSElog2.element_type not in [Float32]):
            raise TypeError("lse_log2 must be Float32")
        if const_expr(mdPsum.element_type not in [Float32]):
            raise TypeError("dpsum must be Float32")
        if const_expr(mdSink.element_type not in [Float32]):
            raise TypeError("dsink must be Float32")

        num_batch = mLSElog2.shape[0]
        num_head = mLSElog2.shape[1]
        seqlen_q_padded = mLSElog2.shape[2]
        num_m_blocks = (seqlen_q_padded + self.m_block_size - 1) // self.m_block_size

        grid_dim = (num_m_blocks, num_head, num_batch)

        self.kernel(
            mSink,
            mLSElog2,
            mdPsum,
            mdSink,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSink: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdSink: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()

        seqlen_q_padded = mLSElog2.shape[2]

        # log2(e), used to convert from natural log to log2 domain.
        LOG2_E = math.log2(math.e)

        # Load sink scalar for this head once into a register.
        sink_val_native = mSink[head_idx]
        sink_val = Float32(sink_val_native)
        sink_log2 = sink_val * LOG2_E

        # Per-thread accumulator across `rows_per_thread` rows.
        partial = Float32(0.0)
        block_offset = m_block * self.m_block_size
        for r in cutlass.range_constexpr(self.rows_per_thread):
            row = block_offset + r * self.num_threads + tidx
            if row < seqlen_q_padded:
                lse_log2 = mLSElog2[batch_idx, head_idx, row]
                # For padded rows (seqlen_q <= row < seqlen_q_padded), the
                # preprocess kernel leaves `lse_log2` at +inf*LOG2_E and writes
                # `dpsum = 0`. With `lse_log2 = +inf`, the term becomes
                # `-exp2(sink_log2 - inf) * 0 = -exp2(-inf) * 0 = 0`, so no
                # extra masking is required here.
                dpsum_val = mdPsum[batch_idx, head_idx, row]
                partial += -utils.exp2f(sink_log2 - lse_log2) * dpsum_val

        # Reduce within each warp.
        warp_sum = utils.warp_reduce(partial, operator.add)

        # Cross-warp reduction via shared memory: warp 0 collects the per-warp
        # sums and reduces them with another warp_reduce.
        smem = cutlass.utils.SmemAllocator()
        sScratch = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.num_warps,)),
            byte_alignment=4,
        )

        warp_idx = tidx // cute.arch.WARP_SIZE
        lane_idx = tidx % cute.arch.WARP_SIZE
        if lane_idx == 0:
            sScratch[warp_idx] = warp_sum

        cute.arch.barrier()

        if warp_idx == 0:
            block_sum = Float32(0.0)
            if lane_idx < self.num_warps:
                block_sum = sScratch[lane_idx]
            block_sum = utils.warp_reduce(
                block_sum, operator.add, width=self.num_warps
            )
            if lane_idx == 0:
                # Atomic add into the per-head output. Output is zero-init'd
                # outside the kernel by the caller.
                ptr = utils.elem_pointer(mdSink, (head_idx,))
                _atomic_add_fp32_asm(block_sum, ptr)
