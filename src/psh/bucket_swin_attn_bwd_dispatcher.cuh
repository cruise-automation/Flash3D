/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 1/17/25
 */


#ifndef FLASH3DPSHATTN_BUCKET_SWIN_ATTN_BWD_DISPATCHER_CUH
#define FLASH3DPSHATTN_BUCKET_SWIN_ATTN_BWD_DISPATCHER_CUH

#include "common/kernel_dispatcher.cuh"
#include "psh/bucket_swin_attn_bwd.cuh"

namespace f3d {

struct BuckSwinAttnBwdDeltaKernel {
  using PtrType = void(*) (
    const bf16 * __restrict__ O_base, const bf16 * __restrict__ dO_base, float * __restrict__ delta_base,
    uint32_t B_size, uint32_t L_total, uint32_t N_head,
    uint32_t stride_batchB, uint32_t stride_seqL, uint32_t stride_headN,
    uint32_t stride_batchB_output, uint32_t stride_seqL_output, uint32_t stride_headN_output);
};

struct BuckSwinAttnBwdKernel {
  using PtrType = void(*) (
    const bf16 * __restrict__ Q_base, const bf16 * __restrict__ K_base, const bf16 * __restrict__ V_base,
    const bf16 * __restrict__ O_base, const bf16 * __restrict__ dO_base, const float * __restrict__ L_base,
    const float * __restrict__ Delta_base,
    bf16 * __restrict__ dQ_base, bf16 * __restrict__ dK_base, bf16 * __restrict__ dV_base,
    uint32_t B_size, uint32_t L_scope, uint32_t N_head,
    uint32_t stride_batchB, uint32_t stride_seqL, uint32_t stride_headN,
    uint32_t stride_batchB_colvec, uint32_t stride_seqL_colvec, uint32_t stride_headN_colvec,
    const uint32_t * __restrict__ scope_base,
    uint32_t scope_stride, uint32_t bucket_size);
};

class BuckSwinAttnBwdDeltaDispatcher :
  public AttnKernelDispatcher<BuckSwinAttnBwdDeltaDispatcher,
    BuckSwinAttnBwdDeltaKernel>{
public:
  void initialize() {
    if (this->kernel_map.size() > 0)
      return;

    this->kernel_map[{16, 16, 3}] = buck_attn_bwd_bf16_preproc<16, 16, 3>;
    this->kernel_map[{16, 16, 2}] = buck_attn_bwd_bf16_preproc<16, 16, 2>;
    this->kernel_map[{16, 16, 1}] = buck_attn_bwd_bf16_preproc<16, 16, 1>;
    this->kernel_map[{16, 8, 3}] = buck_attn_bwd_bf16_preproc<16, 8, 3>;
    this->kernel_map[{16, 8, 2}] = buck_attn_bwd_bf16_preproc<16, 8, 2>;
    this->kernel_map[{16, 8, 1}] = buck_attn_bwd_bf16_preproc<16, 8, 1>;

    this->kernel_map[{32, 16, 3}] = buck_attn_bwd_bf16_preproc<32, 16, 3>;
    this->kernel_map[{32, 16, 2}] = buck_attn_bwd_bf16_preproc<32, 16, 2>;
    this->kernel_map[{32, 16, 1}] = buck_attn_bwd_bf16_preproc<32, 16, 1>;
    this->kernel_map[{32, 8, 3}] = buck_attn_bwd_bf16_preproc<32, 8, 3>;
    this->kernel_map[{32, 8, 2}] = buck_attn_bwd_bf16_preproc<32, 8, 2>;
    this->kernel_map[{32, 8, 1}] = buck_attn_bwd_bf16_preproc<32, 8, 1>;

    this->kernel_map[{64, 16, 3}] = buck_attn_bwd_bf16_preproc<64, 16, 3>;
    this->kernel_map[{64, 16, 2}] = buck_attn_bwd_bf16_preproc<64, 16, 2>;
    this->kernel_map[{64, 16, 1}] = buck_attn_bwd_bf16_preproc<64, 16, 1>;
    this->kernel_map[{64, 8, 3}] = buck_attn_bwd_bf16_preproc<64, 8, 3>;
    this->kernel_map[{64, 8, 2}] = buck_attn_bwd_bf16_preproc<64, 8, 2>;
    this->kernel_map[{64, 8, 1}] = buck_attn_bwd_bf16_preproc<64, 8, 1>;

    this->kernel_map[{128, 16, 3}] = buck_attn_bwd_bf16_preproc<128, 16, 3>;
    this->kernel_map[{128, 16, 2}] = buck_attn_bwd_bf16_preproc<128, 16, 2>;
    this->kernel_map[{128, 16, 1}] = buck_attn_bwd_bf16_preproc<128, 16, 1>;
    this->kernel_map[{128, 8, 3}] = buck_attn_bwd_bf16_preproc<128, 8, 3>;
    this->kernel_map[{128, 8, 2}] = buck_attn_bwd_bf16_preproc<128, 8, 2>;
    this->kernel_map[{128, 8, 1}] = buck_attn_bwd_bf16_preproc<128, 8, 1>;
    this->kernel_map[{128, 4, 3}] = buck_attn_bwd_bf16_preproc<128, 4, 3>;
    this->kernel_map[{128, 4, 2}] = buck_attn_bwd_bf16_preproc<128, 4, 2>;
    this->kernel_map[{128, 4, 1}] = buck_attn_bwd_bf16_preproc<128, 4, 1>;

  }
};

class BuckSwinAttnBwdDispatcher :
  public AttnKernelDispatcher<BuckSwinAttnBwdDispatcher,
    BuckSwinAttnBwdKernel>{
public:
  void initialize() {
    if (this->kernel_map.size() > 0)
      return;

    this->kernel_map[{16, 16, 1}] = buck_attn_bwd_bf16<16, 16>;
    this->kernel_map[{16, 8, 1}]  = buck_attn_bwd_bf16<16, 8>;
    this->kernel_map[{16, 4, 1}]  = buck_attn_bwd_bf16<16, 4>;


    this->kernel_map[{32, 16, 1}] = buck_attn_bwd_bf16<32, 16>;
    this->kernel_map[{32, 8, 1}]  = buck_attn_bwd_bf16<32, 8>;
    this->kernel_map[{32, 4, 1}]  = buck_attn_bwd_bf16<32, 4>;

    this->kernel_map[{64, 16, 1}] = buck_attn_bwd_bf16<64, 16>;
    this->kernel_map[{64, 8, 1}]  = buck_attn_bwd_bf16<64, 8>;
    this->kernel_map[{64, 4, 1}]  = buck_attn_bwd_bf16<64, 4>;

    this->kernel_map[{128, 16, 1}] = buck_attn_bwd_bf16<128, 16>;
    this->kernel_map[{128, 8, 1}]  = buck_attn_bwd_bf16<128, 8>;
    this->kernel_map[{128, 4, 1}]  = buck_attn_bwd_bf16<128, 4>;
  }
};
} // end of ::f3d

#endif //FLASH3DPSHATTN_BUCKET_SWIN_ATTN_BWD_DISPATCHER_CUH
