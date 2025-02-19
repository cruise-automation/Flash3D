/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/23/24
 */


#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "psh/bucket_swin_attn_fwd_dispatcher.cuh"

namespace f3d {

extern void buck_swin_fwd(at::Tensor &Q, at::Tensor &K, at::Tensor &V, at::Tensor &O, at::Tensor &L,
                          at::Tensor &Scope_buckets, uint32_t bucket_size) {
  assert_contiguous({Q, K, V, O, L, Scope_buckets});
  assert_same_device({Q, K, V, O, L, Scope_buckets});
  auto stream = c10::cuda::getCurrentCUDAStream(Q.get_device());
  auto shmem_size = tk::MAX_SHARED_MEMORY;
  auto bsize = uint32_t(Q.size(0));
  auto heads = uint32_t(Q.size(2));
  auto headd = uint16_t(Q.size(3));
  auto scopes = uint32_t(Scope_buckets.size(0));
  auto warp_per_cta = (headd < 128) ? 8u : 4u;

  const std::unordered_set<uint16_t> head_dim_set = {16, 32, 64, 128};
  auto & dev_mgr = DeviceManagerSingleton::instance();
  dev_mgr.set_device(Q.get_device());

  if (Q.size(0) != K.size(0)) {
    throw_format_error("Unmatched B dim: %lld and %lld", Q.size(0), K.size(0));
  }
  if (Q.size(1) < 16u * warp_per_cta) {
    throw_format_error("SeqL %lld is smaller than %u", Q.size(1), 16u * warp_per_cta);
  }
  if (Q.size(3) != headd) {
    throw_format_error("Unexpected HeadD %lld, expected %u", Q.size(3), headd);
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
  if (L.dtype() != at::ScalarType::Float) {
    throw_format_error("LSE vector must be fp32 dtype, instead of %s.", c10::toString(L.dtype().toScalarType()));
  }
  if (Scope_buckets.dtype() != at::ScalarType::UInt32) {
    throw_format_error("Scope_buckets must be UInt32 dtype, instead of %s.",
                       c10::toString(Scope_buckets.dtype().toScalarType()));
  }

  if (dev_mgr.get_sync_stream()) {
    cudaStreamSynchronize(stream);
    cudaCheckLastErr();
  }

  auto & fwd_disp = BuckSwinAttnFwdDispatcher::instance();

  auto attn_ker = fwd_disp.get_kernel_instance(headd, warp_per_cta, 1);
  cudaFuncSetAttribute(attn_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();

  warp_per_cta = std::min(warp_per_cta, get_max_warps_per_cta(attn_ker));
  dim3 fwd_grid{heads, scopes, bsize};

  attn_ker<<<fwd_grid, warp_per_cta * WarpSize, shmem_size, stream>>>(
    (bf16 *) Q.data_ptr(), (bf16 *) K.data_ptr(), (bf16 *) V.data_ptr(),
    (bf16 *) O.data_ptr(), (float *) L.data_ptr(),
    Q.size(0), bucket_size * Scope_buckets.size(1), Q.size(2),
    Q.stride(0), Q.stride(1), Q.stride(2),
    L.stride(0), L.stride(1), L.stride(2),
    (uint32_t *) Scope_buckets.data_ptr(), Scope_buckets.stride(0), bucket_size
    );
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaStreamSynchronize(stream);
    cudaCheckLastErr();
  }
}

} // end of ::f3d