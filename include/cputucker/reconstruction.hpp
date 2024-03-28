#ifndef RECONSTRUCTION_H_
#define RECONSTRUCTION_H_

#include <omp.h>

#include "cputucker/constants.hpp"
#include "cputucker/helper.hpp"

namespace supertensor {
namespace cputucker {

template <typename TensorType, typename MatrixType, typename ErrorType, typename SchedulerType, typename TensorManagerType>
void ComputingReconstruction(TensorType *tensor, TensorType *core_tensor,
                             MatrixType ***factor_matrices, ErrorType **error_T,
                             int rank, SchedulerType *scheduler, TensorManagerType *manager) {
  using tensor_t = TensorType;
  using block_t = typename tensor_t::block_t;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

  int order = tensor->order;
  uint64_t nnz_count = tensor->nnz_count;
  uint64_t core_nnz_count = core_tensor->nnz_count;

  const uint64_t block_count = tensor->block_count;
  index_t *block_dims = tensor->block_dims;

  auto tasks = scheduler->tasks;

  for (uint64_t iter = 0; iter < tasks.size(); ++iter) {
    uint64_t block_id = tasks[iter].block_id;
    uint64_t nnz_count = tasks[iter].nnz_count;

    block_t *curr_block = (block_t *)manager->ReadBlockFromFile(block_id);
    index_t *curr_block_coord = curr_block->get_block_coord();

    index_t **core_indices = core_tensor->blocks[0]->indices;
    value_t *core_values = core_tensor->blocks[0]->values;

    #ifdef OPENMP

#pragma omp parallel 
{
  index_t *nnz_idx = cputucker::allocate<index_t>(order);
#pragma omp for schedule(static)
    for (uint64_t nnz = 0; nnz < nnz_count; ++nnz) {
      double ans = 0.0f;
      
      for (int axis = 0; axis < order; ++axis) {
        nnz_idx[axis] = curr_block->indices[axis][nnz];
      }

      for (uint64_t co_nnz = 0; co_nnz < core_tensor->nnz_count; ++co_nnz) {
        value_t temp = core_values[co_nnz];
        for (int axis = 0; axis < order; ++axis) {
          index_t part_id = curr_block_coord[axis];
          index_t pos = nnz_idx[axis] * rank + core_indices[axis][co_nnz];

          temp *= factor_matrices[axis][part_id][pos];
        }
        ans += temp;
      }
      error_T[block_id][nnz] = ans;
    }
}
#endif

#ifdef AVX
    uint64_t sz_width = std::is_same<value_t, double>::value ? 8 : 16;

    uint64_t co_nnz_avx_size = ((core_tensor->nnz_count + sz_width - 1) / sz_width) * sz_width;
    assert(co_nnz_avx_size <= core_tensor->nnz_count);
    std::cout << "core nnz avx size: " << co_nnz_avx_size << std::endl;
// Assuming data structures are properly aligned and definitions are given
// for types Loop over non-zero elements
#pragma omp parallel 
{
    index_t *nnz_idx = cputucker::allocate<index_t>(order);

#pragma omp for schedule (static)
    for (uint64_t nnz = 0; nnz < nnz_count; ++nnz) {

      for (int axis = 0; axis < order; ++axis) {
        nnz_idx[axis] = curr_block->indices[axis][nnz];
      }
      
      double ans = 0.0f;
      __m512d ans_v = _mm512_setzero_pd();  // Vectorized sum for AVX-512
      uint64_t co_nnz;

      for (co_nnz = 0; co_nnz < co_nnz_avx_size; co_nnz += sz_width) {  // Process 8 items at a time
        __m512d temp_v = _mm512_loadu_pd(&core_values[co_nnz]);  // Load 8 values

        for (int axis = 0; axis < order; ++axis) {
          index_t part_id = curr_block_coord[axis];

          __m256i nnz_idx_v = _mm256_set1_epi32(nnz_idx[axis] * rank);
          __m256i core_idx_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&core_indices[axis][co_nnz]));
          __m256i pos_v = _mm256_add_epi32(nnz_idx_v, core_idx_v);

          __m512d factor_v = _mm512_i32gather_pd(pos_v, &factor_matrices[axis][part_id][0], 8);
          temp_v = _mm512_mul_pd(temp_v, factor_v);

          // uint32_t pos[8];
          // for(int i = 0; i < 8; ++i) {
          //   pos[i] = nnz_idx[axis] * rank + core_indices[axis][co_nnz + i];
          // }
          // __m256i pos_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pos));
          // __m512d factor_v = _mm512_i32gather_pd(pos_v, factor_matrices[axis][part_id], 8);
          // temp_v = _mm512_mul_pd(temp_v, factor_v);
        }
        ans_v = _mm512_add_pd(ans_v, temp_v);  // Vectorized addition
      }
      ans = _mm512_reduce_add_pd(ans_v);  // Reduce vector to scalar

      for(co_nnz = co_nnz_avx_size; co_nnz < core_tensor->nnz_count; ++co_nnz) {
        value_t temp = core_values[co_nnz];
        for (int axis = 0; axis < order; ++axis) {
          index_t part_id = curr_block_coord[axis];
          index_t pos = nnz_idx[axis] * rank + core_indices[axis][co_nnz];
          temp *= factor_matrices[axis][part_id][pos];
        }
        ans += temp;
      }
      error_T[block_id][nnz] = ans;
    }
}

#endif
  }  // task.size()

}

template <typename TensorType, typename MatrixType, typename ErrorType, typename SchedulerType, typename TensorManager>
void Reconstruction(TensorType *tensor, TensorType *core_tensor,
                    MatrixType ***factor_matrices, double *fit,
                    ErrorType **error_T, int rank,SchedulerType *scheduler, TensorManager *manager) {
  MYPRINT("[ Reconstruction ]\n");
  using tensor_t = TensorType;
  using block_t = typename tensor_t::block_t;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

  int order = tensor->order;
  uint64_t nnz_count = tensor->nnz_count;
  uint64_t core_nnz_count = core_tensor->nnz_count;

  uint64_t block_count = tensor->block_count;
  uint64_t max_nnz_count_in_block = tensor->get_max_nnz_count_in_block();
  index_t *block_dims = tensor->block_dims;

  double recons_time = omp_get_wtime();

  ComputingReconstruction(tensor, core_tensor, factor_matrices, error_T, rank, scheduler, manager);

  double error_step_time = omp_get_wtime();
  uint64_t start_cycle_time = rdtsc();
  value_t Error = 0.0f;

  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    block_t *curr_block = (block_t *)manager->ReadBlockFromFile(block_id);
    value_t *curr_block_values = curr_block->values;
    value_t *curr_block_error = error_T[block_id];
    uint64_t curr_block_nnz_count = curr_block->nnz_count;


    ///////////////////////////////////////////////////////////////////
    // ---------------------------------------------------------------
    #ifdef OPENMP
      #pragma omp parallel for schedule(static) reduction(+ : Error)
      for (uint64_t nnz = 0; nnz < curr_block_nnz_count; ++nnz) {
        value_t err_tmp = curr_block_values[nnz] - curr_block_error[nnz];
        Error += err_tmp * err_tmp;
      }

    #endif

    #ifdef AVX
    uint64_t sz_width = std::is_same<value_t, double>::value ? 8 : 16;

    uint64_t nnz = 0;
    uint64_t nnz_avx_size =
        ((curr_block_nnz_count - sz_width + 1) / sz_width) * sz_width;
    assert(nnz_avx_size <= curr_block_nnz_count);
    std::cout << "nnz avx size: " << nnz_avx_size << std::endl;
      #pragma omp parallel reduction(+:Error)
      {
        __m512d sum_error_v =  _mm512_setzero_pd();  // a vector with eight zero
        #pragma omp for schedule(static)
        for (nnz = 0; nnz < nnz_avx_size; nnz += sz_width) {
          __m512d old_val_v = _mm512_loadu_pd(&curr_block_values[nnz]);
          __m512d new_val_v = _mm512_loadu_pd(&curr_block_error[nnz]);
          __m512d diff_v = _mm512_sub_pd(old_val_v, new_val_v);
          // #pragma omp critical
          sum_error_v = _mm512_fmadd_pd(diff_v, diff_v, sum_error_v);
        }
        // #pragma omp critical
        Error += _mm512_reduce_add_pd(sum_error_v);
      }
      // If the number of total elements does not cleanly fit into these vectors,
      // you would have to write a special case to thandle athe remainder.

      // Remainder (scalar operations)
    #pragma omp parallel for schedule(static) reduction(+ : Error)
    for (nnz = nnz_avx_size; nnz < curr_block_nnz_count; ++nnz) {
      value_t err_tmp = curr_block_values[nnz] - curr_block_error[nnz];
      Error += err_tmp * err_tmp;
    }

    #endif

      // ---------------------------------------------------------------
      ///////////////////////////////////////////////////////////////////
  }
  printf("Error Step Time : %lf\n", omp_get_wtime() - error_step_time);
  std::cout << "\t CPU cycles " << rdtsc() - start_cycle_time << std::endl;

  printf("Error:: %1.3f \t Norm:: %1.3f\n", Error, tensor->norm);

  if (tensor->norm == 0) {
    *fit = 1;
  } else {
    *fit = 1.0f - std::sqrt(Error) / tensor->norm;
  }

 
}
}  // namespace cputucker
}  // namespace supertensor

#endif /* RECONSTRUCTION_H_ */