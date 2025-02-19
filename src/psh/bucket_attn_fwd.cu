/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 *  found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 07/15/24
 */


#include <chrono>
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


namespace tk = kittens;

namespace f3d {


template <uint16_t HeadD, uint16_t WarpPerBlock=16>
__global__ void attention_ker_bf16(
  uint32_t L_seq, const bf16 * __restrict__ Q_base, const bf16 * __restrict__ K_base, const bf16* __restrict__ V_base,
  bf16* __restrict__ O_base) {
  /*
   * Forward attention kernel for flat bucket layout without in-flight bucket indexing.
   * We assume inputs were indexed and shuffled before passing to this kernel.
   * QKV shapes are [B, L_seq, D]
   */
  using namespace kittens;

  auto warpid = tk::warpid();
  auto block_size = WARP_THREADS * WarpPerBlock;
  auto block_start = blockIdx.x * (L_seq * HeadD);
  const bf16 * Q_blk = Q_base + block_start, * K_blk = K_base + block_start, * V_blk = V_base + block_start;
  bf16 * O_blk = O_base + block_start;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) __shm);

  auto (&k_shmem)[WarpPerBlock] = al.allocate<st_bf_1x4<ducks::st_layout::swizzle>, WarpPerBlock>();
  auto (&v_shmem)[WarpPerBlock] = al.allocate<st_bf_1x4<ducks::st_layout::swizzle>, WarpPerBlock>();

  // Initialize register tiles.
  rt_bf_1x4<> q_reg, k_reg, v_reg; // v_reg need to be swapped into col_l
  rt_fl_1x1<> att_block;
  rt_bf_1x1<> att_block_mma;
  rt_fl_1x4<> o_reg;
  rt_fl_1x1<>::col_vec max_vec_last, max_vec; // these are column vectors for the attention block
  rt_fl_1x1<>::col_vec norm_vec_last, norm_vec; // these are column vectors for the attention block

  uint32_t qo_blocks = L_seq / (q_reg.rows * WarpPerBlock), kv_blocks = L_seq / (q_reg.rows * WarpPerBlock);

  for (auto q_blk_idx = 0; q_blk_idx < qo_blocks; q_blk_idx++) {
    // each warp loads its own Q tile of 16x64, and then multiplies by 1/sqrt(d)
    load(q_reg, Q_blk + (q_blk_idx * WarpPerBlock + warpid) * q_reg.num_elements, q_reg.cols);
    mul(q_reg, q_reg, __float2bfloat16(1.f / sqrt(HeadD))); // temperature adjustment
    // q_reg: [16, 64]

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_reg);

    for (auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
      // each warp loads its own chunk of k, v into shared memory
      load(v_shmem[warpid], V_blk + (kv_idx * WarpPerBlock + warpid) * q_reg.num_elements, q_reg.cols);
      load(k_shmem[warpid], K_blk + (kv_idx * WarpPerBlock + warpid) * q_reg.num_elements, q_reg.cols);
      __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

      // now each warp goes through subtiles, loads them, and then does the flash attention internal alg.
        for(auto subtile = 0; subtile < WarpPerBlock; subtile++) {
          load(k_reg, k_shmem[subtile]); // load k from shared into registers

          zero(att_block); // zero 16x16 attention tile
          mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T

          copy(norm_vec_last, norm_vec);
          copy(max_vec_last,  max_vec);

          row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
          sub_row(att_block, att_block, max_vec); // subtract max from attention -- now all <=0
          exp(att_block, att_block); // exponentiate the block in-place.

          sub(max_vec_last, max_vec_last, max_vec); // subtract new max from old max to find the new normalization.
          exp(max_vec_last, max_vec_last); // exponentiate this vector -- this is what we need to normalize by.
          mul(norm_vec, norm_vec, max_vec_last); // and the norm vec is now normalized.

          row_sum(norm_vec, att_block, norm_vec); // accumulate the new attention block onto the now-rescaled norm_vec
          div_row(att_block, att_block, norm_vec); // now the attention block is correctly normalized

          mul(norm_vec_last, norm_vec_last, max_vec_last); // normalize the previous norm vec according to the new max
          div(norm_vec_last, norm_vec_last, norm_vec); // normalize the previous norm vec according to the new norm

          copy(att_block_mma, att_block); // convert to bf16 for mma_AB

          load(v_reg, v_shmem[subtile]); // load v from shared into registers.
          rt_bf_1x4<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

          mul_row(o_reg, o_reg, norm_vec_last); // normalize o_reg in advance of mma_AB'ing onto it
          mma_AB(o_reg, att_block_mma, v_reg_col, o_reg); // mfma onto o_reg with the local attention@V matmul.
        }
        __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
    }

    store(O_blk + (q_blk_idx * WarpPerBlock + warpid) * q_reg.num_elements, o_reg, q_reg.cols);
    // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
  }
}



void attention_forward(at::Tensor &Q, at::Tensor &K, at::Tensor &V, at::Tensor &O,
                       uint32_t N, uint16_t num_head, uint16_t num_warps_opt, uint16_t num_bucket) {
  assert_contiguous({Q, K, V, O});
  assert_dtype({Q, K, V, O}, c10::ScalarType::BFloat16);

  constexpr uint16_t num_warps = 12;

  if (N % (num_warps_opt * 16) != 0) {
    char buff[256];
    sprintf(buff, "QKV length=%d indivisible by tile_len=%d", N, num_warps * 16);
    throw std::runtime_error(buff);
  }

  auto stream = c10::cuda::getCurrentCUDAStream(Q.get_device());
  auto shmem_size = tk::MAX_SHARED_MEMORY;
  auto num_blocks = num_head * num_bucket;

  if (num_warps_opt == 16) {
    constexpr uint16_t num_warps = 16;

    cudaFuncSetAttribute(
      attention_ker_bf16<64, num_warps>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shmem_size);
    cudaCheckLastErr();

    QuickTimer timer{};

    attention_ker_bf16<64, num_warps>
      <<<num_blocks, 32 * num_warps, shmem_size>>>(
        N, (bf16*)Q.data_ptr(), (bf16*)K.data_ptr(), (bf16*)V.data_ptr(), (bf16*)O.data_ptr()
        );
    cudaCheckLastErr();

    cudaCheckErr(cudaStreamSynchronize(stream));
    auto [number, str] = timer.end_and_format<std::chrono::microseconds>();
    LOG(WARNING) << "Time: " << str;
  }
  else if (num_warps_opt == 12) {
    constexpr uint16_t num_warps = 12;

    cudaFuncSetAttribute(
      attention_ker_bf16<64, num_warps>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shmem_size);
    cudaCheckLastErr();

    QuickTimer timer{};

    attention_ker_bf16<64, num_warps>
      <<<num_blocks, 32 * num_warps, shmem_size>>>(
        N, (bf16*)Q.data_ptr(), (bf16*)K.data_ptr(), (bf16*)V.data_ptr(), (bf16*)O.data_ptr()
        );
    cudaCheckLastErr();

    cudaCheckErr(cudaStreamSynchronize(stream));
    auto [number, str] = timer.end_and_format<std::chrono::microseconds>();
    LOG(WARNING) << "Time: " << str;
  }
  else if (num_warps_opt == 8) {
    constexpr uint16_t num_warps = 8;

    cudaFuncSetAttribute(
      attention_ker_bf16<64, num_warps>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shmem_size);
    cudaCheckLastErr();

    QuickTimer timer{};

    attention_ker_bf16<64, num_warps>
      <<<num_blocks, 32 * num_warps, shmem_size>>>(
        N, (bf16*)Q.data_ptr(), (bf16*)K.data_ptr(), (bf16*)V.data_ptr(), (bf16*)O.data_ptr()
        );
    cudaCheckLastErr();

    cudaCheckErr(cudaStreamSynchronize(stream));
    auto [number, str] = timer.end_and_format<std::chrono::microseconds>();
    LOG(WARNING) << "Time: " << str;
  }
  else {
    char buff[256];
    sprintf(buff, "Unsupported num_warps_opt=%d, accepting {12, 16}", num_warps_opt);
    throw std::runtime_error(buff);
  }
}

} // End of ::f3d