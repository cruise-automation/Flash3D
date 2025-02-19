/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 11/08/24
 */


#ifndef FLASH3DPSHATTN_BUCKET_SWIN_ATTN_BWD_CUH
#define FLASH3DPSHATTN_BUCKET_SWIN_ATTN_BWD_CUH

#include <stdexcept>
#include <glog/logging.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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

template <uint16_t HeadD, uint16_t WarpPerBlock=16, uint16_t PIPESTAGE=2, bool IS_DEBUG=false>
__global__ void buck_attn_bwd_bf16_preproc(
  const bf16 * __restrict__ O_base, const bf16 * __restrict__ dO_base, float * __restrict__ delta_base,
  uint32_t B_size, uint32_t L_total, uint32_t N_head,
  uint32_t stride_batchB, uint32_t stride_seqL, uint32_t stride_headN,
  uint32_t stride_batchB_output, uint32_t stride_seqL_output, uint32_t stride_headN_output
  ) {
  auto warpid = tk::warpid();
  auto headid = blockIdx.x;
  auto batchid = blockIdx.y;
  if (warpid >= WarpPerBlock) return;
  if (batchid >= B_size) return;
  if (headid >= N_head) return;

  auto O_batch_base = O_base + batchid * stride_batchB;
  auto dO_batch_base = dO_base + batchid * stride_batchB;
  auto D_batch_base = delta_base + batchid * stride_batchB_output;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) __shm);

  auto (&dO_shmem)[WarpPerBlock][PIPESTAGE] =
    al.allocate<tk::st_bf<1, HeadD / 16u, tk::ducks::st_layout::swizzle>, WarpPerBlock, PIPESTAGE>();
  auto (&O_shmem)[WarpPerBlock][PIPESTAGE] =
    al.allocate<tk::st_bf<1, HeadD / 16u, tk::ducks::st_layout::swizzle>, WarpPerBlock, PIPESTAGE>();
  //tk::rt_fl<1, HeadD / 16u, tk::ducks::rt_layout::row> delta_reg;

  //using D_reg_type = decltype(delta_reg)::col_vec;
  using o_shm_type = std::remove_reference_t<decltype(dO_shmem[0][0])>;
  using D_shm_type = tk::sv_fl_1;
  auto (&delta_sum_shm)[WarpPerBlock] = al.allocate<D_shm_type, WarpPerBlock>();

  uint32_t o_blocks = L_total / (o_shm_type::rows * WarpPerBlock);
  uint32_t o_blk_stride = WarpPerBlock * o_shm_type::rows * stride_seqL;
  uint32_t o_warp_stride = o_shm_type::rows * stride_seqL;
  uint32_t d_blk_stride = WarpPerBlock * D_shm_type::length * stride_seqL_output;
  uint32_t d_warp_stride = D_shm_type::length * stride_seqL_output;

  // Initial filling
  for (auto o_blk_init = 0, head_tic = 0;
    o_blk_init < o_blocks && head_tic < PIPESTAGE - 1;
    ++o_blk_init, ++head_tic) {

    auto o_blk_L_base = O_batch_base + o_blk_init * o_blk_stride;
    auto o_warp_L_base = o_blk_L_base + warpid * o_warp_stride;
    auto o_warp_LH_base = o_warp_L_base + headid * stride_headN;

    auto dO_blk_L_base = dO_batch_base + o_blk_init * o_blk_stride;
    auto dO_warp_L_base = dO_blk_L_base + warpid * o_warp_stride;
    auto dO_warp_LH_base = dO_warp_L_base + headid * stride_headN;

    load_async_2B_elem(O_shmem[warpid][head_tic], o_warp_LH_base, stride_seqL);
    load_async_2B_elem(dO_shmem[warpid][head_tic], dO_warp_LH_base, stride_seqL);
  }

  for (auto o_blk_iter = 0, tail_tic = PIPESTAGE - 1;
    o_blk_iter < o_blocks;
    ++o_blk_iter, tail_tic = (tail_tic + 1) % PIPESTAGE) {

    auto head_tic = (tail_tic + 1) % PIPESTAGE;
    auto o_blk_tail = o_blk_iter + PIPESTAGE - 1;
    if (o_blk_tail < o_blocks) {
      auto o_blk_L_tail = O_batch_base + o_blk_tail * o_blk_stride;
      auto o_warp_L_tail = o_blk_L_tail + warpid * o_warp_stride;
      auto o_warp_LH_tail = o_warp_L_tail + headid * stride_headN;
      auto do_blk_L_tail = dO_batch_base + o_blk_tail * o_blk_stride;
      auto do_warp_L_tail = do_blk_L_tail + warpid * o_warp_stride;
      auto do_warp_LH_tail = do_warp_L_tail + headid * stride_headN;

      load_async_2B_elem(O_shmem[warpid][tail_tic], o_warp_LH_tail, stride_seqL);
      load_async_2B_elem(dO_shmem[warpid][tail_tic], do_warp_LH_tail, stride_seqL);
    }
    // Deal with ending cases
    auto remain_iters = o_blocks - o_blk_iter;
    if (remain_iters > PIPESTAGE) {
      _wait_groups_except_and_sync_all<2 * (PIPESTAGE - 1)>();
    }
    if (remain_iters <= PIPESTAGE && remain_iters > 1) {
      _wait_groups_except_and_sync_all<2>();
    }
    if (remain_iters <= 1) {
      _wait_sync_all();
    }

    mul_row_sum(delta_sum_shm[warpid], O_shmem[warpid][head_tic], dO_shmem[warpid][head_tic]);

    if (warpid == 0 && IS_DEBUG) {
      //if (tk::laneid() == 0) printf("head_tic=%d, tail_tic=%d\n", head_tic, tail_tic);
      //auto & dbg_shm = al.allocate<tk::st_bf<1, HeadD / 16u, tk::ducks::st_layout::swizzle>>();
      //print_stile_rec(O_shmem[warpid][head_tic], warpid, o_blk_iter, 0, 16, 0, 16);
      if (tk::laneid() == 0) printf("dO shmem\n");
      //print_stile_rec(dO_shmem[warpid][head_tic], warpid, o_blk_iter, 0, 16, 0, 16);
      //return;
    }
    auto D_blk_L_base = D_batch_base + o_blk_iter * d_blk_stride;
    auto D_warp_L_base = D_blk_L_base + warpid * d_warp_stride;
    auto D_warp_LH_base = D_warp_L_base + headid * stride_headN_output;

    store(D_warp_LH_base, delta_sum_shm[warpid], stride_seqL_output);
    __syncthreads();
  }
}

template <uint16_t HeadD, uint16_t WarpPerBlock=16, bool IS_DEBUG=false>
__global__ void buck_attn_bwd_bf16(
  const bf16 * __restrict__ Q_base, const bf16 * __restrict__ K_base, const bf16 * __restrict__ V_base,
  const bf16 * __restrict__ O_base, const bf16 * __restrict__ dO_base, const float * __restrict__ L_base,
  const float * __restrict__ Delta_base,
  bf16 * __restrict__ dQ_base, bf16 * __restrict__ dK_base, bf16 * __restrict__ dV_base,
  uint32_t B_size, uint32_t L_scope, uint32_t N_head,
  uint32_t stride_batchB, uint32_t stride_seqL, uint32_t stride_headN,
  uint32_t stride_batchB_colvec, uint32_t stride_seqL_colvec, uint32_t stride_headN_colvec,
  const uint32_t * __restrict__ scope_base,
  uint32_t scope_stride, uint32_t bucket_size
  ) {
  // Compute thread and block indices using attention scope indexing
  uint32_t warpid  = tk::warpid();
  uint32_t headid  = blockIdx.x;
  uint32_t scopeid = blockIdx.y;
  uint32_t batchid = blockIdx.z;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) __shm);

  // Compute tile and bucket parameters.
  uint32_t q_tile_rows       = tk::rt_bf<1, HeadD/16u>::rows;
  uint32_t qo_blocks         = L_scope / (q_tile_rows * WarpPerBlock);
  uint32_t kv_blocks         = L_scope / (q_tile_rows * WarpPerBlock);
  uint32_t blocks_per_bucket = bucket_size / (q_tile_rows * WarpPerBlock);
  uint32_t qo_blk_stride     = WarpPerBlock * q_tile_rows * stride_seqL;
  uint32_t qo_warp_stride    = q_tile_rows * stride_seqL;
  uint32_t kv_blk_stride     = WarpPerBlock * q_tile_rows * stride_seqL;
  uint32_t kv_warp_stride    = q_tile_rows * stride_seqL;
  uint32_t bucket_stride     = bucket_size * stride_seqL;
  uint32_t bucket_stride_colvec = bucket_size * stride_seqL_colvec;

  // Compute base pointers for input/output tensors.
  const bf16* Q_batch       = Q_base   + batchid * stride_batchB + headid * stride_headN;
  const bf16* K_batch       = K_base   + batchid * stride_batchB + headid * stride_headN;
  const bf16* V_batch       = V_base   + batchid * stride_batchB + headid * stride_headN;
  const bf16* dO_batch      = dO_base  + batchid * stride_batchB + headid * stride_headN;
  bf16* dQ_batch            = dQ_base  + batchid * stride_batchB + headid * stride_headN;
  bf16* dK_batch            = dK_base  + batchid * stride_batchB + headid * stride_headN;
  bf16* dV_batch            = dV_base  + batchid * stride_batchB + headid * stride_headN;
  const float* L_batch      = L_base     + batchid * stride_batchB_colvec;
  const float* Delta_batch  = Delta_base + batchid * stride_batchB_colvec;
  auto scope_inds           = scope_base + scopeid * scope_stride;

  // Allocate shared memory tiles for Q, O, dO
  auto (&q_shmem)[WarpPerBlock] = al.allocate<tk::st_bf<1, HeadD/16u, tk::ducks::st_layout::swizzle>, WarpPerBlock>();
  auto (&dO_shmem)[WarpPerBlock] = al.allocate<tk::st_bf<1, HeadD/16u, tk::ducks::st_layout::swizzle>, WarpPerBlock>();

  // Declare register tiles.
  tk::rt_bf<1, HeadD/16u> q_reg;
  tk::rt_bf<1, HeadD/16u> k_reg;
  tk::rt_bf<1, HeadD/16u> v_reg;
  tk::rt_fl_1x1<> s_reg;   // to compute S = Q * K^T
  tk::rt_fl_1x1<> dp_reg;  // to compute dP = dO * V^T
  tk::rt_fl_1x1<> ds_reg;  // to compute dS = P ∘ (dP - Delta)
  tk::rt_bf_1x1<> ds_reg_bf;
  tk::rt_bf_1x1<> P_reg_half;   // to hold P = exp(S - L)
  tk::rt_bf<1, HeadD/16u, tk::ducks::rt_layout::col> dO_reg;  // dO tile
  tk::rt_fl<1, HeadD/16u> dQ_reg;  // dQ tile accumulator
  tk::rt_fl<1, HeadD/16u> dK_reg;  // dK accumulator in float
  tk::rt_fl<1, HeadD/16u> dV_reg;  // dV accumulator in float

  // Column vector registers for L and Delta.
  tk::rt_fl_1x1<>::col_vec lse_vec_reg, d_vec_reg;
  auto (&lse_vec)[WarpPerBlock] = al.allocate<tk::sv_fl_1, WarpPerBlock>(); // LSE rows
  auto (&d_vec)[WarpPerBlock] = al.allocate<tk::sv_fl_1, WarpPerBlock>(); // Delta rows

  // Outer loop: iterate over K/V blocks (j direction)
  for (uint32_t kv_blk_iter  = 0; kv_blk_iter < kv_blocks; kv_blk_iter++) {
    // Compute bucket indices for K/V.
    uint32_t kv_buck_seq     = kv_blk_iter / blocks_per_bucket;
    uint32_t kv_buck_id      = scope_inds[kv_buck_seq];
    uint32_t kv_buck_blk_off = kv_blk_iter % blocks_per_bucket;

    // Compute pointers for current K/V bucket.
    const bf16* K_bucket = K_batch + kv_buck_id * bucket_stride;
    const bf16* V_bucket = V_batch + kv_buck_id * bucket_stride;

    // Compute tile offset for K/V.
    uint32_t kv_tile_offset = kv_buck_blk_off * kv_blk_stride + warpid * kv_warp_stride;

    // Load K and V tiles directly from HMB into register tiles.
    tk::load(k_reg, K_bucket + kv_tile_offset, int32_t(stride_seqL));
    tk::load(v_reg, V_bucket + kv_tile_offset, int32_t(stride_seqL));

    // Initialize dK and dV accumulators.
    tk::zero(dK_reg);
    tk::zero(dV_reg);

    // Inner loop: iterate over Q blocks (i direction)
    for (uint32_t q_blk_iter_i = 0; q_blk_iter_i < qo_blocks; q_blk_iter_i++) {
      // Compute bucket indices for Q.
      uint32_t qo_buck_seq    = q_blk_iter_i / blocks_per_bucket;
      uint32_t qo_buck_id     = scope_inds[qo_buck_seq];
      uint32_t qo_buck_blk_off = q_blk_iter_i % blocks_per_bucket;

      // Compute pointers for current Q, O, dO, dQ tiles.
      const bf16* Q_bucket  = Q_batch  + qo_buck_id * bucket_stride;
      const bf16* dO_bucket = dO_batch + qo_buck_id * bucket_stride;
      bf16* dQ_bucket       = dQ_batch + qo_buck_id * bucket_stride;

      uint32_t tile_offset = qo_buck_blk_off * qo_blk_stride + warpid * qo_warp_stride;
      const bf16* Q_tile_ptr  = Q_bucket  + tile_offset;
      //const bf16* O_tile_ptr  = O_bucket  + tile_offset;
      const bf16* dO_tile_ptr = dO_bucket + tile_offset;
      bf16* dQ_tile_ptr       = dQ_bucket + tile_offset;

      // TODO: double-buffering shmem
      // Load Q, O, dO, dQ tiles into register tiles.
      load_async_2B_elem(q_shmem[warpid], Q_tile_ptr, int32_t(stride_seqL));
      load_async_2B_elem(dO_shmem[warpid], dO_tile_ptr, int32_t(stride_seqL));

      // Load L and Delta for current tile using colvec layout.
      uint32_t qo_blk_stride_colvec  = WarpPerBlock * q_tile_rows * stride_seqL_colvec;
      uint32_t qo_warp_stride_colvec = q_tile_rows * stride_seqL_colvec;
      const float* L_bucket_base     = L_batch + qo_buck_id * bucket_stride_colvec;
      const float* Delta_bucket_base = Delta_batch + qo_buck_id * bucket_stride_colvec;
      uint32_t col_offset = qo_buck_blk_off * qo_blk_stride_colvec +
                            warpid * qo_warp_stride_colvec +
                            headid * stride_headN_colvec;
      load(lse_vec[warpid], L_bucket_base + col_offset, int32_t(stride_seqL_colvec));
      load(d_vec[warpid], Delta_bucket_base + col_offset, int32_t(stride_seqL_colvec));

      _wait_groups_except_and_sync_all<0>();
      for (auto subtile = 0; subtile < WarpPerBlock; ++ subtile) {
        // Step 1: Compute S = Q_tile * (K_tile)^T using MMA.
        tk::load(q_reg, q_shmem[subtile]);
        tk::zero(s_reg);
        tk::mma_ABt(s_reg, q_reg, k_reg, s_reg);
        tk::mul(s_reg, s_reg, 1.0 / sqrt(HeadD));

        // Step 2: Compute P = exp(S - L) elementwise.
        tk::load(lse_vec_reg, lse_vec[warpid]);
        tk::sub_row(s_reg, s_reg, lse_vec_reg);
        //exp2(s_tile, s_tile);
        tk::exp(s_reg, s_reg);
        tk::copy(P_reg_half, s_reg);

        // Step 3: Update dV: dV_reg += (P)^T * dO_tile.
        tk::load(dO_reg, dO_shmem[subtile]);
        auto & P_reg_t = tk::swap_layout_inplace(P_reg_half);
        tk::mma_AtB(dV_reg, P_reg_t, dO_reg, dV_reg);

        // Step 4: Compute dP = dO_tile * (V_tile)^T.
        tk::zero(dp_reg);
        auto & dO_reg_trans = tk::swap_layout_inplace(dO_reg);

        tk::mma_ABt(dp_reg, dO_reg_trans, v_reg, dp_reg);

        // Step 5: Compute dS = P ∘ (dP - Delta) elementwise.
        tk::load(d_vec_reg, d_vec[warpid]);
        tk::sub_row(dp_reg, dp_reg, d_vec_reg);
        tk::mul(ds_reg, s_reg, dp_reg);
        tk::copy(ds_reg_bf, ds_reg);
        // TODO: multiply ds by a small_scale

        // Step 6: Update dQ: dQ_reg += dS * K_tile.
        tk::load(dQ_reg, dQ_tile_ptr, int32_t(stride_seqL));
        tk::mma_AB(dQ_reg, ds_reg_bf, tk::swap_layout_inplace(k_reg), dQ_reg);

        // Step 7: Update dK: dK_reg += (dS)^T * Q_tile.
        auto & ds_reg_trans = tk::swap_layout_inplace(ds_reg_bf);
        auto & q_reg_trans = tk::swap_layout_inplace(q_reg);
        tk::mma_AtB(dK_reg, ds_reg_trans, q_reg_trans, dK_reg);

        // Store updated dQ tile back to global memory.
        tk::store(dQ_tile_ptr, dQ_reg, int32_t(stride_seqL));
      }
    } // End inner loop over Q blocks

    __syncthreads();
    // Write back dK and dV accumulators for current K/V block.
    uint32_t kv_tile_offset_out = kv_buck_blk_off * kv_blk_stride + warpid * kv_warp_stride;
    bf16* dK_ptr = dK_batch + kv_buck_id * bucket_stride + kv_tile_offset_out;
    bf16* dV_ptr = dV_batch + kv_buck_id * bucket_stride + kv_tile_offset_out;
    tk::store(dK_ptr, dK_reg, int32_t(stride_seqL));
    tk::store(dV_ptr, dV_reg, int32_t(stride_seqL));
  } // End outer loop over K/V blocks
}

} // end of ::f3d

#endif //FLASH3DPSHATTN_BUCKET_SWIN_ATTN_BWD_CUH
