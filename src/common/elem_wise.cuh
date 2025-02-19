/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 *  found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/25/25
 */


#ifndef FLASH3DPSHATTN_ELEM_WISE_H
#define FLASH3DPSHATTN_ELEM_WISE_H

namespace f3d {

struct tile_exp2 {
  template<typename T> __device__ __forceinline__ static T op(const T &x) {return exp2(x);}
};
template<> __device__ __forceinline__ float tile_exp2::op<float>(const float &x) {return exp2f(x);}
template<> __device__ __forceinline__ float2 tile_exp2::op<float2>(const float2 &x) {return {exp2f(x.x), exp2f(x.y)};}
template<> __device__ __forceinline__ bf16 tile_exp2::op<bf16>(const bf16 &x) {return hexp2(x);}
template<> __device__ __forceinline__ bf16_2 tile_exp2::op<bf16_2>(const bf16_2 &x) {return h2exp2(x);}

struct tile_log2 {
  template<typename T> __device__ __forceinline__ static T op(const T &x) { return log2(x); }
};
template<> __device__ __forceinline__ float tile_log2::op<float>(const float &x) {return log2f(x); }
template<> __device__ __forceinline__ float2 tile_log2::op<float2>(const float2 &x) {return { log2f(x.x), log2f(x.y) };}
template<> __device__ __forceinline__ bf16 tile_log2::op<bf16>(const bf16 &x) {return hlog2(x); }
template<> __device__ __forceinline__ bf16_2 tile_log2::op<bf16_2>(const bf16_2 &x) {return h2log2(x); }

template<typename op, tk::ducks::rt::all T, typename U>
__device__ static inline void bin_map_upcast(T &dst, const T &src, const U &param) {
  #pragma unroll
  for(int i = 0; i < dst.height; i++) {
    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
      #pragma unroll
      for(int k = 0; k < dst.packed_per_tile; k++) {
          dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k], param);
      }
    }
  }
}

template<typename op, tk::ducks::rt::all RT, tk::ducks::st::all STL, tk::ducks::st::all STR>
__device__ static inline void bin_map_upcast(RT & dst, const STL &lhs, const STR &rhs) {

  static_assert(std::is_same_v<typename RT::layout, tk::ducks::rt_layout::row>, "Output T must be row-major");
  static_assert(RT::height == STL::height, "Reg tile and Shmem tile lhs must match height");
  static_assert(RT::width == STL::width, "Reg tile and Shmem tile lhs must match width");
  static_assert(RT::height == STR::height, "Reg tile and Shmem tile rhs must match height");
  static_assert(RT::width == STR::width, "Reg tile and Shmem tile rhs must match width");

  //using D2 = RT::dtype;
  using LHT = STL::dtype;
  using LHT2 = tk::base_types::packing<LHT>::packed_type;
  using RHT = STR::dtype;
  using RHT2 = tk::base_types::packing<RHT>::packed_type;
  int2 rowcol{0, 0};

  #pragma unroll
  for(int i = 0; i < dst.height; i++) {
    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
      #pragma unroll
      for(int k = 0; k < dst.packed_per_tile; k++) {
        auto row = i * dst.tile_size + (tk::laneid() / 4);
        auto col = j * dst.tile_size + 2 * (tk::laneid() %4);

        rowcol = {row + 0, col + 0};
        auto & lhs_pack = *(LHT2*)(&lhs[rowcol]);
        auto & rhs_pack = *(RHT2*)(&rhs[rowcol]);
        dst.tiles[i][j].data[0] = op::template op<LHT>(lhs_pack, rhs_pack);

        rowcol = {row + 8, col + 0};
        lhs_pack = *(LHT2*)(&lhs[rowcol]);
        rhs_pack = *(RHT2*)(&rhs[rowcol]);
        dst.tiles[i][j].data[1] = op::template op<LHT>(lhs_pack, rhs_pack);

        rowcol = {row + 0, col + 8};
        lhs_pack = *(LHT2*)(&lhs[rowcol]);
        rhs_pack = *(RHT2*)(&rhs[rowcol]);
        dst.tiles[i][j].data[2] = op::template op<LHT>(lhs_pack, rhs_pack);

        rowcol = {row + 8, col + 8};
        lhs_pack = *(LHT2*)(&lhs[rowcol]);
        rhs_pack = *(RHT2*)(&rhs[rowcol]);
        dst.tiles[i][j].data[3] = op::template op<LHT>(lhs_pack, rhs_pack);
      }
    }
  }
}

struct upcast_mul {
  template<typename T> __device__ __forceinline__
  static T op(const T &a, const float &b) {return float(a) * b;}
  template<typename T> __device__ __forceinline__
  static T op(const T &a, const float2 &b) {return {float(a.x) * b.x, float(a.y) * b.y};}
  template<typename T> __device__ __forceinline__
  static T op(const T &a, const T &b) {return float(a) * float(b);}
  template<typename T> __device__ __forceinline__
  static tk::base_types::packing<T>::packed_type op(
    const tk::base_types::packing<T>::packed_type &a, const tk::base_types::packing<T>::packed_type &b)
  {return {float(a.x) * float(b.x), float(a.y) * float(b.y)};}
};
template<> __device__ __forceinline__ bf16 upcast_mul::op<bf16>(const bf16 &a, const float &b)
{return __float2bfloat16(float(a) * b);}
template<> __device__ __forceinline__ bf16_2 upcast_mul::op<bf16_2>(const bf16_2 &a, const float &b)
{return __floats2bfloat162_rn(float(a.x) * b, float(a.y) * b);}
template<> __device__ __forceinline__ bf16_2 upcast_mul::op<bf16_2>(const bf16_2 &a, const float2 &b)
{return __floats2bfloat162_rn(float(a.x) * b.x, float(a.y) * b.y);}

struct upcast_mul_to_fl {
  template<typename T> __device__ __forceinline__
  static float op(const T &a, const float &b) {return float(a) * b;}
  template<typename T> __device__ __forceinline__
  static float2 op(const T &a, const float2 &b) {return {float(a.x) * b.x, float(a.y) * b.y};}
  template<typename T> __device__ __forceinline__
  static float op(const T &a, const T &b) {return float(a) * float(b);}
  template<typename T> __device__ __forceinline__
  static float2 op(const tk::base_types::packing<T>::packed_type &a, const tk::base_types::packing<T>::packed_type &b)
  {return {float(a.x) * float(b.x), float(a.y) * float(b.y)};}
};

template<tk::ducks::rt::all T>
__device__ __forceinline__ static void exp2(T &dst, const T &src)
{tk::unary_map<tile_exp2, T>(dst, src);}

template<tk::ducks::rv::all T>
__device__ __forceinline__ static void exp2(T &dst, const T &src)
{tk::unary_op<tile_exp2, T>(dst, src);}

template<tk::ducks::rt::all T>
__device__ __forceinline__ static void log2(T &dst, const T &src)
{tk::unary_map<tile_log2, T>(dst, src); }

template<tk::ducks::rv::all T>
__device__ __forceinline__ static void log2(T &dst, const T &src)
{tk::unary_op<tile_log2, T>(dst, src); }

template<tk::ducks::rt::all T, typename U>
__device__ __forceinline__ static void mul_up(T & dst, const T &lhs, const U &rhs)
{bin_map_upcast<upcast_mul, T>(dst, lhs, rhs);}

template<tk::ducks::rv::all T, typename U>
__device__ __forceinline__ static void mul_up(T & dst, const T &lhs, const U &rhs)
{bin_map_upcast<upcast_mul, T>(dst, lhs, rhs);}

template<tk::ducks::rt::all RT, tk::ducks::st::all ST>
__device__ __forceinline__ static void mul_to_fl(RT & dst, const ST &lhs, const ST &rhs)
{bin_map_upcast<upcast_mul_to_fl, RT, ST, ST>(dst, lhs, rhs);}

template<tk::ducks::sv::all SV, tk::ducks::st::all ST, bool IS_DEBUG=false>
__device__ __forceinline__ static void mul_row_sum(SV & dst, const ST &lhs, const ST &rhs) {
  auto laneid = tk::laneid();
  tk::zero(dst);
  __syncwarp();

  constexpr uint32_t elem_per_warp = WarpSize;
  constexpr uint32_t total_iters = cdiv_dev(ST::num_elements, elem_per_warp);
  static_assert(std::is_same_v<typename SV::dtype, float>, "Dst dtype must be float");

  #pragma unroll
  for (auto warp_iter = 0; warp_iter < total_iters; ++warp_iter) {
    int32_t src_elem = warp_iter * WarpSize + laneid;
    int32_t src_row = src_elem / ST::cols;
    int32_t src_col = src_elem % ST::cols;
    float prod = float(lhs[{src_row, src_col}]) * float(rhs[{src_row, src_col}]);
    float * dst_ptr = &dst[src_row];
    auto prev = atomicAdd_block(dst_ptr, prod);
    if (laneid == 0 && tk::warpid() == 0 && IS_DEBUG) {
      printf("Lane[%d],elem[%d],rc[%d,%d],dst[%d],prod=%.2f,prev=%.2f\n",
             laneid, src_elem, src_row, src_col, src_row, prod, prev);
    }
  }
  __syncwarp();
}
} // end of ::f3d

#endif //FLASH3DPSHATTN_ELEM_WISE_H
