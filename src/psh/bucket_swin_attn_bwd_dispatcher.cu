/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/30/24
 */


#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "psh/gemm.cuh"
#include "psh/bucket_swin_attn_bwd_dispatcher.cuh"

namespace f3d {

extern void
buck_swin_bwd(at::Tensor &Q, at::Tensor &K, at::Tensor &V, at::Tensor &O, at::Tensor &dO, at::Tensor &LSE,
              at::Tensor &Delta, at::Tensor &dQ, at::Tensor &dK, at::Tensor &dV,
              at::Tensor &Scope_buckets, uint32_t bucket_size) {
  assert_contiguous({Q, K, V, O, dO, Delta, dQ, dK, dV, Scope_buckets});
  assert_same_device({Q, K, V, O, dO, Delta, dQ, dK, dV, Scope_buckets});

  auto stream = c10::cuda::getCurrentCUDAStream(Q.get_device());
  auto shmem_size = tk::MAX_SHARED_MEMORY;
  auto bsize = uint32_t(Q.size(0));
  auto heads = uint32_t(Q.size(2));
  auto headd = uint16_t(Q.size(3));
  auto scopes = uint32_t(Scope_buckets.size(0));
  auto warp_per_cta = (headd < 128) ? 8u: 4u;
  const uint16_t pipe_stage = (headd < 64) ? 3 : ((headd < 128) ? 2: 1);
  auto warp_per_cta_bwd = 4u;
  switch (headd) {
    case 16:
      warp_per_cta_bwd = 8u;
      break;
    case 32:
      warp_per_cta_bwd = 8u;
      break;
    case 64:
      warp_per_cta_bwd = 8u;
      break;
    case 128:
      warp_per_cta_bwd = 4u;
      break;
  }

  const std::unordered_set<uint16_t> head_dim_set = {16, 32, 64, 128};
  auto & dev_mgr = DeviceManagerSingleton::instance();
  dev_mgr.set_device(Q.get_device());

  if (Q.size(0) != K.size(0)) {
    throw_format_error("Unmatched B dim: %lld and %lld", Q.size(0), K.size(0));
  }
  if (Q.size(1) < 16u * warp_per_cta) {
    throw_format_error("SeqL %lld is smaller than %u", Q.size(1), 16u * warp_per_cta);
  }
  if (head_dim_set.find(Q.size(3)) == head_dim_set.end()) {
    throw_format_error("Unexpected HeadD %lld", Q.size(3));
  }
  if (bucket_size < warp_per_cta * 16u) {
    throw_format_error("Bucket size %u is smaller than the expected %u", bucket_size, warp_per_cta * 16u);
  }
  if (bucket_size % (warp_per_cta * 16u) != 0) {
    throw_format_error("Bucket size %d is not divisible by %d", bucket_size, warp_per_cta * 16);
  }
  if (Scope_buckets.numel() * bucket_size != Q.size(1)) {
    throw_format_error("Total bucket tokens %d don't add up to total sequence length %d",
                       Scope_buckets.numel() * bucket_size, Q.size(1));
  }
  if (Delta.dtype() != at::ScalarType::Float) {
    throw_format_error("Delta vector must be fp32 dtype, instead of %s.",
                       c10::toString(Delta.dtype().toScalarType()));
  }
  if (LSE.dtype() != at::ScalarType::Float) {
    throw_format_error("LSE vector must be fp32 dtype, instead of %s.",
                       c10::toString(LSE.dtype().toScalarType()));
  }
  if (Scope_buckets.dtype() != at::ScalarType::UInt32) {
    throw_format_error("Scope_buckets must be UInt32 dtype, instead of %s.",
                       c10::toString(Scope_buckets.dtype().toScalarType()));
  }

  if (dev_mgr.get_sync_stream()) {
    cudaStreamSynchronize(stream);
    cudaCheckLastErr();
  }

  auto & delta_disp = BuckSwinAttnBwdDeltaDispatcher::instance();
  auto & bwd_disp = BuckSwinAttnBwdDispatcher::instance();

  auto delta_ker = delta_disp.get_kernel_instance(headd, warp_per_cta, pipe_stage);
  auto bwd_ker = bwd_disp.get_kernel_instance(headd, 8, 1);
  cudaFuncSetAttribute(delta_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();
  cudaFuncSetAttribute(bwd_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();

  dim3 delta_grid{heads, bsize};
  dim3 bwd_grid{heads, scopes, bsize};

  delta_ker<<<delta_grid, warp_per_cta * WarpSize, shmem_size, stream>>>(
    (bf16 *) O.data_ptr(), (bf16 *) dO.data_ptr(), (float *) Delta.data_ptr(),
    O.size(0), O.size(1), O.size(2),
    O.stride(0), O.stride(1), O.stride(2),
    Delta.stride(0), Delta.stride(1), Delta.stride(2)
    );
  cudaCheckLastErr();

  bwd_ker<<<bwd_grid, warp_per_cta_bwd * WarpSize, shmem_size, stream>>>(
    (bf16 *) Q.data_ptr(), (bf16 *) K.data_ptr(), (bf16 *) V.data_ptr(), (bf16 *) O.data_ptr(),
    (bf16 *) dO.data_ptr(), (float *) LSE.data_ptr(), (float *) Delta.data_ptr(),
    (bf16 *) dQ.data_ptr(), (bf16 *) dK.data_ptr(), (bf16 *) dV.data_ptr(),
    Q.size(0), bucket_size * Scope_buckets.size(1), Q.size(2),
    Q.stride(0), Q.stride(1), Q.stride(2),
    LSE.stride(0), LSE.stride(1), LSE.stride(2),
    (uint32_t *) Scope_buckets.data_ptr(),
    Scope_buckets.stride(0), bucket_size
    );
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaStreamSynchronize(stream);
    cudaCheckLastErr();
  }
}

} // end of ::f3d