
/*
  Inspired from: https://github.com/dblalock/bolt
*/

#include "maddness.hpp"

static_assert(__AVX2__, "AVX 2 is required! Try --march=native or -mavx2");


static void maddness_encode_fp32_t(const float *X,
				   int C,
				   int nsplits,
				   int nrows,
				   int ncols,
				   const uint32_t *splitdims,
				   const int8_t *splitvals,
				   const float *scales,
				   const float *offsets,
				   uint8_t* out) {

  /*
    Inputs:
    X* ... FP32 Matrix to be encoded. it is given as **ROW_MAJOR** order with the shape of [nrows, ncols].
    C  ... the number of codebooks.
    splitdims, splitvals, scales, offsets
    -> As it is, but the format is that:
    [ProtoType0(Bucket_0), ProtoType0(Bucket_1), ProtoType0(Bucket_2), ..., ProtoType1()]

    *out ... uint8 matrix [N, C] to store the encoded X*.

    Memo:
    X* ...
    AABBCC
    AABBCC
    AABBCC
    AABBCC
    AABBCC
    
    AA...Proto1, BB...Proto2, CC...Proto3

    => becames
    out ...
    ABC
    ABC
    ABC
    ABC
    ABC
    (Choose the best A,B,C(Encoded) from AA, BB, CC)

    Add offsets to X*, out*, for this function to change the visible araa.

    splitdims' stride(x) = 2**x
  */

  static constexpr int block_nrows = 32; // One Instruction = 32 elements
  const int64_t nblocks = ceil(nrows / (double)block_nrows);
  const int64_t total_buckets_per_tree = 2^nsplits;
  
  const float *x_ptrs[total_buckets_per_tree];
  __m256i current_vsplitval_luts[total_buckets_per_tree];
  __m256  current_vscales[total_buckets_per_tree];
  __m256  current_voffsets[total_buckets_per_tree];

  assert(nrows % block_nrows == 0); // TODO: remove it
  
  int mtree_offset = 0;
  int split_idx = 0;
  
  int STEP = ncols/C;
  
  // X.stride = [1, nrow]
  for (int cth=0;cth<C;cth++) {
    // Processing prototypes-wise: AA -> BB -> CC.
    auto out_ptr = out + (nrows * cth); // out_ptr = out[:, C:]
    mtree_offset = cth * total_buckets_per_tree;
    
    for (int bucket_n=0;bucket_n<total_buckets_per_tree;bucket_n++) {
      auto splitdim = splitdims[mtree_offset + bucket_n];
      splitdim += cth * STEP; // Add Offsets. {AABBCC}[n], n=cth*STEP + splitdim.
      
      x_ptrs[bucket_n] = X + (nrows * splitdim); // x_ptrs[bucket_b] = X[:, splitdim].

      auto splitval_ptr = splitvals + mtree_offset; // First splitval at each level.
      // Here, splitval_ptr = {Bucket(0, 0)}, {Bucket(0, 1), Bucket(1, 1)}, {Bucket(1, 1), Bucket(0, 2), ...}
      
      current_vsplitval_luts[bucket_n] = _mm256_broadcastsi128_si256(load_si128i((const __m128i *)splitval_ptr));
      current_vscales[bucket_n]  = _mm256_set1_ps(scales[mtree_offset + bucket_n]);
      current_voffsets[bucket_n] = _mm256_set1_ps(offsets[mtree_offset + bucket_n]); 
    }

    /*
      __m256 variables: current_XXX is now:

                          / length=nsplits \
      [Bucket(0, 0), Bucket(0, 1), Bucket(0, 2), Bucket(0, 3)] where Bucket(i, t) is bucket, t=tree_level, i=bucket_idx.
    */

    for (int b=0;b<nblocks;b++) {
      __m256i group_idxs = _mm256_setzero_si256();
#pragma unroll
      // tlevel=bucket_n
      for (int bucket_n=0;bucket_n<nsplits;bucket_n++) {
	auto vscales   = current_vscales[bucket_n];
	auto voffsets  = current_voffsets[bucket_n];
	auto vsplitvals_lut = current_vsplitval_luts[bucket_n];
        auto vsplitvals =  _mm256_shuffle_epi8(vsplitvals_lut, group_idxs);

	auto x_ptr = x_ptrs[bucket_n];
	x_ptrs[bucket_n] += block_nrows;
	
	// cmp
	auto x_i8 = load_4xf32_as_32xepi8_or_epu8<true, true>(x_ptr, vscales, voffsets);	
        auto masks = _mm256_cmpgt_epi8(x_i8, vsplitvals);
	auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);
	
	if (bucket_n>0) {
	  // MaddnessHash, 2*idx
	  group_idxs = _mm256_add_epi8(group_idxs, group_idxs);
	}
	
        group_idxs = _mm256_or_si256(group_idxs, masks_0_or_1);
      }
      
      group_idxs = _mm256_permutevar8x32_epi32(group_idxs, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
      _mm256_storeu_si256((__m256i *)out_ptr, group_idxs);
      out_ptr += block_nrows;
    }
    // TO ADD: Reminder
  }
}

extern "C" {
  void maddness_encode(const float *X,
		       int C,
		       int nsplits,
		       int nrows,
		       int ncols,
		       const uint32_t *splitdims,
		       const int8_t *splitvals,
		       const float *scales,
		       const float *offsets,
		       uint8_t* out) {
    maddness_encode_fp32_t(X, C, nsplits, nrows, ncols, splitdims, splitvals, scales, offsets, out);
  }

  void maddness_scan(const uint8_t* encoded_mat,
		     int N,
		     int C,
		     int M,
		     const uint8_t* luts,
		     uint8_t* out_mat) {
    // luts, out_mat ... RowMajor
    // encoded ... column major
    // for 0 ~ M, add ptr
    mithral_scan(encoded_mat, N / 32, C, M, luts, out_mat);
  }
}
