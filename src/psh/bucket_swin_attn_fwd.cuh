/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 6/18/24
 */


#ifndef FLASH3DPSHATTN_BUCKET_SWIN_ATTN_H
#define FLASH3DPSHATTN_BUCKET_SWIN_ATTN_H

#include <stdexcept>
#include <glog/logging.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda/pipeline>
#include <cuda/barrier>

#include <kittens.cuh>
#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "common/rand.cuh"
#include "common/load_store.cuh"
#include "common/load_store_async.cuh"
#include "common/elem_wise.cuh"
#include "common/device_manager.cuh"
#include "common/block_minmax.cuh"
#include "psh/hash_fns.cuh"
#include "psh/two_stage_counter.cuh"
#include "psh/shmem_cumsum_counter.cuh"

namespace tk = kittens;

namespace f3d {

template <uint16_t HeadD, uint16_t WarpPerBlock=16, bool IS_DEBUG=false>
__global__ void buck_attn_fwd_bf16(
  const bf16 * __restrict__ Q_base, const bf16 * __restrict__ K_base, const bf16* __restrict__ V_base,
  bf16* __restrict__ O_base, float * __restrict__ LSE_base,
  uint32_t B_size, uint32_t L_scope, uint32_t N_head,
  uint32_t stride_batchB, uint32_t stride_seqL, uint32_t stride_headN,
  uint32_t stride_B_LSE, uint32_t stride_L_LSE, uint32_t stride_H_LSE,
  const uint32_t * __restrict__ scope_inds,
  uint32_t scope_stride, uint32_t bucket_size
  ) {
  auto warpid = tk::warpid();
  auto headid = blockIdx.x;
  auto scopeid = blockIdx.y;
  auto batchid = blockIdx.z;

  auto Q_batch_base = Q_base + batchid * stride_batchB;
  auto K_batch_base = K_base + batchid * stride_batchB;
  auto V_batch_base = V_base + batchid * stride_batchB;
  auto O_batch_base = O_base + batchid * stride_batchB;
  auto L_batch_base = LSE_base + batchid * stride_B_LSE;
  auto scope_ind_base = scope_inds + scopeid * scope_stride;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) __shm);

  auto (&k_shmem)[WarpPerBlock] = al.allocate<tk::st_bf<1, HeadD / 16u, tk::ducks::st_layout::swizzle>, WarpPerBlock>();
  auto (&v_shmem)[WarpPerBlock] = al.allocate<tk::st_bf<1, HeadD / 16u, tk::ducks::st_layout::swizzle>, WarpPerBlock>();

  // Initialize register tiles.
  tk::rt_bf<1, HeadD / 16u> q_reg, k_reg, v_reg; // v_reg need to be swapped into col_l
  tk::rt_fl_1x1<> att_block;
  tk::rt_bf_1x1<> att_block_mma;
  tk::rt_fl<1, HeadD / 16u> o_reg;
  tk::rt_fl_1x1<>::col_vec max_vec_last, max_vec; // these are column vectors for the attention block
  tk::rt_fl_1x1<>::col_vec norm_vec_last, norm_vec; // these are column vectors for the attention block
  auto (&lse_vec)[WarpPerBlock] = al.allocate<tk::sv_fl_1, WarpPerBlock>(); // accumulate LSE rows

  using q_reg_type = decltype(q_reg);
  using k_reg_type = decltype(k_reg);

  uint32_t qo_blocks = L_scope / (q_reg_type::rows * WarpPerBlock);
  uint32_t kv_blocks = L_scope / (k_reg_type::rows * WarpPerBlock);
  uint32_t blocks_per_bucket = bucket_size / (q_reg_type::rows * WarpPerBlock);
  uint32_t qo_blk_stride = WarpPerBlock * q_reg_type::rows * stride_seqL;
  uint32_t qo_warp_stride = q_reg_type::rows * stride_seqL;
  uint32_t kv_blk_stride = WarpPerBlock * k_reg_type::rows * stride_seqL;
  uint32_t kv_warp_stride = k_reg_type::rows * stride_seqL;
  // Buckets are indexed on the sequence length dimension.
  // A bucket of size 512 contains 512 tokens; a token strides at stride_seqL
  uint32_t bucket_stride = bucket_size * stride_seqL;
  uint32_t bucket_stride_LSE = bucket_size * stride_L_LSE;
  uint32_t LSE_blk_stride = WarpPerBlock * q_reg_type::rows * stride_L_LSE;
  uint32_t LSE_warp_stride = q_reg_type::rows * stride_L_LSE;


  for (auto qo_blk_iter = 0; qo_blk_iter < qo_blocks; ++ qo_blk_iter) {
    auto qo_buck_seq = qo_blk_iter / blocks_per_bucket;
    auto qo_buck_id = scope_ind_base[qo_buck_seq];
    auto qo_buck_blk_off = qo_blk_iter % blocks_per_bucket;

    auto q_bucket_base = Q_batch_base + qo_buck_id * bucket_stride;
    auto q_warp_L_base = q_bucket_base + qo_buck_blk_off * qo_blk_stride + warpid * qo_warp_stride;
    auto q_warp_LH_base = q_warp_L_base + headid * stride_headN;
    tk::load(q_reg, q_warp_LH_base, int32_t(stride_seqL));

    if (IS_DEBUG) {
      auto &dbg_shmem = al.allocate<tk::st_bf<1, HeadD / 16u, tk::ducks::st_layout::swizzle>>();
      if (0 == qo_blk_iter && 0 == warpid) {
        if (tk::laneid() == 0) {
          printf("Blk[%d,%d,%d],q_iter=%d,warpid=%d: qo_offset=%d,+warp_offset=%d,+head_offset=%d\n",
                 blockIdx.x, blockIdx.y, blockIdx.z,
                 qo_blk_iter, warpid,
                 qo_blk_iter * qo_blk_stride, qo_blk_iter * qo_blk_stride + warpid * qo_warp_stride,
                 qo_blk_iter * qo_blk_stride + warpid * qo_warp_stride + headid * stride_headN);
        }
        tk::store(dbg_shmem, q_reg);
        print_stile_rec(dbg_shmem, warpid, qo_blk_iter, 0, 16, 0, 16);
      }
      __syncthreads();
      if (0 == qo_blk_iter && 1 == warpid) {
        if (tk::laneid() == 0) {
          printf("Blk[%d,%d,%d],q_iter=%d,warpid=%d: qo_offset=%d,+warp_offset=%d,+head_offset=%d\n",
                 blockIdx.x, blockIdx.y, blockIdx.z,
                 qo_blk_iter, warpid,
                 qo_blk_iter * qo_blk_stride, qo_blk_iter * qo_blk_stride + warpid * qo_warp_stride,
                 qo_blk_iter * qo_blk_stride + warpid * qo_warp_stride + headid * stride_headN);
        }
        tk::store(dbg_shmem, q_reg);
        print_stile_rec(dbg_shmem, warpid, qo_blk_iter, 0, 16, 0, 16);
      }
    }
    // Exp function with Change-of-Base formula: 2^(x/ln(2)) = e^(x). Recall that ALUs can do 2^a by bit-shifting.
    // 1/sqrt(D) * 1/ln(2)
    mul_up(q_reg, q_reg, 1.0 / sqrt(HeadD));

    tk::neg_infty(max_vec);
    tk::zero(norm_vec);
    tk::zero(o_reg);

    for (auto kv_blk_iter = 0; kv_blk_iter < kv_blocks; ++kv_blk_iter) {
      auto kv_buck_seq = kv_blk_iter / blocks_per_bucket;
      auto kv_buck_id = scope_ind_base[kv_buck_seq];
      auto kv_buck_blk_off = kv_blk_iter % blocks_per_bucket;

      auto k_bucket_base = K_batch_base + kv_buck_id * bucket_stride;
      auto k_warp_L_base = k_bucket_base + kv_buck_blk_off * kv_blk_stride + warpid * kv_warp_stride;
      auto k_warp_LH_base = k_warp_L_base + headid * stride_headN;

      auto v_bucket_base = V_batch_base + kv_buck_id * bucket_stride;
      auto v_warp_L_base = v_bucket_base + kv_buck_blk_off * kv_blk_stride + warpid * kv_warp_stride;
      auto v_warp_LH_base = v_warp_L_base + headid * stride_headN;

      load_async_2B_elem(k_shmem[warpid], k_warp_LH_base, stride_seqL);
      load_async_2B_elem(v_shmem[warpid], v_warp_LH_base, stride_seqL);
      _wait_sync_all();
      //_wait_groups_except_and_sync_all<0>();

      #pragma unroll
      for (auto subtile = 0; subtile < WarpPerBlock; ++subtile) {
        if (IS_DEBUG) {
          if (0 == qo_blk_iter && 0 == warpid) {
            if (tk::laneid() == 0) {
              printf("Blk[%d,%d,%d]Warp[%d],kv_iter=%d,subtile=%d: kv_offset=%d,+warp_offset=%d,+head_offset=%d\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, warpid,
                     kv_blk_iter, subtile,
                     kv_blk_iter * kv_blk_stride, kv_blk_iter * kv_blk_stride + subtile * kv_warp_stride,
                     kv_blk_iter * kv_blk_stride + subtile * kv_warp_stride + headid * stride_headN);
            }
            print_stile_rec(k_shmem[subtile], subtile, kv_blk_iter, 0, 16, 0, 16);
          }
        }
        tk::load(k_reg, k_shmem[subtile]);
        tk::zero(att_block);
        tk::mma_ABt(att_block, q_reg, k_reg, att_block);
        tk::copy(norm_vec_last, norm_vec);
        tk::copy(max_vec_last, max_vec);
        tk::row_max(max_vec, att_block, max_vec);
        tk::sub_row(att_block, att_block, max_vec);
        //exp2(att_block, att_block);
        tk::exp(att_block, att_block);
        tk::sub(max_vec_last, max_vec_last, max_vec);
        //exp2(max_vec_last, max_vec_last);
        tk::exp(max_vec_last, max_vec_last);
        tk::mul(norm_vec, norm_vec, max_vec_last);
        tk::row_sum(norm_vec, att_block, norm_vec);
        tk::div_row(att_block, att_block, norm_vec);
        tk::mul(norm_vec_last, norm_vec_last, max_vec_last);
        tk::div(norm_vec_last, norm_vec_last, norm_vec);
        tk::copy(att_block_mma, att_block);
        tk::load(v_reg, v_shmem[subtile]);
        auto & v_reg_col = tk::swap_layout_inplace(v_reg);
        tk::mul_row(o_reg, o_reg, norm_vec_last);
        tk::mma_AB(o_reg, att_block_mma, v_reg_col, o_reg);
      }
      __syncthreads();
    }
    auto o_bucket_base = O_batch_base + qo_buck_id * bucket_stride;
    auto o_warp_L_base = o_bucket_base + qo_buck_blk_off * qo_blk_stride + warpid * qo_warp_stride;
    auto o_warp_LH_base = o_warp_L_base + headid * stride_headN;
    tk::store(o_warp_LH_base, o_reg, int32_t(stride_seqL));

    tk::log(norm_vec, norm_vec);
    //tk::mul(max_vec, max_vec, M_LN2);
    tk::add(max_vec, max_vec, norm_vec);
    tk::store(lse_vec[warpid], max_vec);

    auto L_bucket_base = L_batch_base + qo_buck_id * bucket_stride_LSE;
    auto L_warp_L_base = L_bucket_base + qo_buck_blk_off * LSE_blk_stride + warpid * LSE_warp_stride;
    auto L_warp_LH_base = L_warp_L_base + headid * stride_H_LSE;

    store(L_warp_LH_base, lse_vec[warpid], int32_t(stride_L_LSE));
  }
}


} // end of ::f3d

#endif //FLASH3DPSHATTN_BUCKET_SWIN_ATTN_H
