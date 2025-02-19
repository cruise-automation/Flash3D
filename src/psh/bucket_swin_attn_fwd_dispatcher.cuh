/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 1/17/25
 */


#ifndef FLASH3DPSHATTN_BUCKET_SWIN_FWD_DISPATCHER_H
#define FLASH3DPSHATTN_BUCKET_SWIN_FWD_DISPATCHER_H

#include "common/kernel_dispatcher.cuh"
#include "psh/bucket_swin_attn_fwd.cuh"

namespace f3d {

struct BuckSwinAttnFwdKernel {
  using PtrType = void(*) (
    const bf16 * __restrict__ Q_base, const bf16 * __restrict__ K_base, const bf16* __restrict__ V_base,
    bf16* __restrict__ O_base, float * __restrict__ LSE_base,
    uint32_t B_size, uint32_t L_scope, uint32_t N_head,
    uint32_t stride_batchB, uint32_t stride_seqL, uint32_t stride_headN,
    uint32_t stride_B_LSE, uint32_t stride_L_LSE, uint32_t stride_H_LSE,
    const uint32_t * __restrict__ scope_inds,
    uint32_t scope_stride, uint32_t bucket_size
  );
};

class BuckSwinAttnFwdDispatcher :
  public AttnKernelDispatcher<BuckSwinAttnFwdDispatcher, BuckSwinAttnFwdKernel> {
public:
  void initialize() {
    if (this->kernel_map.size() > 0)
      return;
    this->kernel_map[{16, 16, 1}] = buck_attn_fwd_bf16<16, 16>;
    this->kernel_map[{16, 8, 1}]  = buck_attn_fwd_bf16<16, 8>;
    this->kernel_map[{16, 4, 1}]  = buck_attn_fwd_bf16<16, 4>;

    this->kernel_map[{32, 16, 1}] = buck_attn_fwd_bf16<32, 16>;
    this->kernel_map[{32, 8, 1}]  = buck_attn_fwd_bf16<32, 8>;
    this->kernel_map[{32, 4, 1}]  = buck_attn_fwd_bf16<32, 4>;

    this->kernel_map[{64, 16, 1}] = buck_attn_fwd_bf16<64, 16>;
    this->kernel_map[{64, 8, 1}]  = buck_attn_fwd_bf16<64, 8>;
    this->kernel_map[{64, 4, 1}]  = buck_attn_fwd_bf16<64, 4>;

    this->kernel_map[{128, 16, 1}] = buck_attn_fwd_bf16<128, 16>;
    this->kernel_map[{128, 8, 1}]  = buck_attn_fwd_bf16<128, 8>;
    this->kernel_map[{128, 4, 1}]  = buck_attn_fwd_bf16<128, 4>;
  }
};
} // end of ::f3d
#endif //FLASH3DPSHATTN_BUCKET_SWIN_FWD_DISPATCHER_H
