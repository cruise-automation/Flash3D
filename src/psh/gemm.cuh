/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 *  found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 2/3/25
 */


#ifndef FLASH3DPSHATTN_HEMM_H
#define FLASH3DPSHATTN_HEMM_H

#include <kittens.cuh>
#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "common/rand.cuh"
#include "common/load_store_async.cuh"
#include "common/elem_wise.cuh"
#include "common/device_manager.cuh"

namespace tk = kittens;

namespace f3d {

template <uint16_t TilesPerBlock=8, bool IS_DEBUG=false>
__global__ void gemm_sm_bf16_ker(
  const bf16 * __restrict__ A_base, const bf16 * __restrict__ B_base, bf16 * __restrict__ O_base,
  uint32_t stride_A_row, uint32_t stride_B_row, uint32_t stride_O_row,
  uint32_t M, uint32_t N, uint32_t K
  ) {
  uint16_t warpid = tk::warpid();
  if (warpid > TilesPerBlock) return;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) __shm);

  auto (&b_shmem)[TilesPerBlock] = al.allocate<tk::st_bf_1x1<tk::ducks::st_layout::swizzle>, TilesPerBlock>();


  tk::rt_bf_1x1<tk::ducks::rt_layout::row> a_reg;
  tk::rt_bf_1x1<tk::ducks::rt_layout::col> b_reg;
  tk::rt_fl_1x1<> o_reg;
  using a_reg_type = std::remove_reference_t<decltype(a_reg)>;
  using b_shm_type = std::remove_reference_t<decltype(b_shmem[0])>;

  uint32_t k_blocks = K / (b_shm_type::rows * TilesPerBlock);
  uint32_t Ak_iter_stride = a_reg_type::cols;
  uint32_t Ak_warp_stride = a_reg_type::rows * stride_A_row;
  uint32_t Bk_iter_stride = TilesPerBlock * b_shm_type::rows * stride_B_row;
  uint32_t Bk_warp_stride = b_shm_type::rows * stride_B_row;

  tk::zero(o_reg);

  for (auto k_iter = 0; k_iter < k_blocks; ++k_iter) {
    auto b_iter_base = B_base + k_iter * Bk_iter_stride;
    auto b_warp_base = b_iter_base + warpid * Bk_warp_stride;

    load_async_2B_elem(b_shmem[warpid], b_warp_base, stride_B_row);

    for (auto w_iter = 0; w_iter < TilesPerBlock; ++w_iter) {
      auto a_iter_base = A_base + (k_iter * TilesPerBlock + w_iter) * Ak_iter_stride;
      auto a_warp_base = a_iter_base + warpid * Ak_warp_stride;

      tk::load(a_reg, a_warp_base, int(stride_A_row));

      if (w_iter == 0) {
        _wait_sync_all();
      }

      tk::load(b_reg, b_shmem[w_iter]);
      tk::mma_AB(o_reg, a_reg, b_reg, o_reg);

      if (IS_DEBUG) {
        auto &dbg_shm = al.allocate<tk::st_bf_1x1<tk::ducks::st_layout::swizzle>>();
        if (warpid == 0) {
          tk::store(dbg_shm, o_reg);
        }
        print_stile_rec(dbg_shm, warpid, w_iter, 0, 16, 0, 16);
      }
    }
  }

  uint32_t O_warp_stride = a_reg_type::rows * stride_O_row;
  auto o_warp_base = O_base + O_warp_stride * warpid;

  tk::store(o_warp_base, o_reg, int(stride_O_row));
}


} // end of ::f3d

#endif //FLASH3DPSHATTN_HEMM_H
