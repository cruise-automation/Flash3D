/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 *  found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 2/3/25
 */


#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "psh/gemm.cuh"

namespace f3d {

extern void gemm_sm_bf16(at::Tensor &A, at::Tensor &B, at::Tensor &O) {
  assert_contiguous({A, B, O});
  assert_same_device({A, B, O});

  if (A.size(1) != B.size(0)) {
    LOG(ERROR) << "Unmatched K dim";
    return;
  }
  if (A.size(0) != 128) {
    LOG(ERROR) << "M must be 128 instead of " << A.size(0);
    return;
  }

  auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());
  auto shmem_size = tk::MAX_SHARED_MEMORY;

  auto gemm_ker = gemm_sm_bf16_ker<8>;
  cudaFuncSetAttribute(gemm_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();

  auto & dev_mgr = DeviceManagerSingleton::instance();
  dev_mgr.set_device(A.get_device());

  gemm_ker<<<1, 8 * WarpSize, shmem_size, stream>>>(
    (bf16 *) A.data_ptr(), (bf16 *) B.data_ptr(), (bf16 *) O.data_ptr(),
    A.stride(0), B.stride(0), O.stride(0),
    A.size(0), B.size(0), A.size(1)
    );
  cudaCheckLastErr();

  cudaStreamSynchronize(stream);
  cudaCheckLastErr();
}

} // end of ::f3d