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

#pragma omp parallel for schedule(static)
    for (uint64_t nnz = 0; nnz < nnz_count; ++nnz) {
      double ans = 0.0f;
      index_t *nnz_idx = cputucker::allocate<index_t>(3);
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
    delete curr_block;
  }  // task.size()

}

template <typename TensorType, typename MatrixType, typename ErrorType, typename SchedulerType, typename TensorManagerType>
void Reconstruction(TensorType *tensor, TensorType *core_tensor,
                    MatrixType ***factor_matrices, double *fit,
                    ErrorType **error_T, int rank, SchedulerType *scheduler, TensorManagerType *manager) {
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

  value_t Error = 0.0f;

  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    block_t *curr_block = (block_t *)manager->ReadBlockFromFile(block_id);
    value_t *curr_block_values = curr_block->values;
    value_t *curr_block_error = error_T[block_id];
    uint64_t curr_block_nnz_count = curr_block->nnz_count;
#pragma omp parallel for schedule(static) reduction(+ : Error)
    for (uint64_t nnz = 0; nnz < curr_block_nnz_count; ++nnz) {
      value_t err_tmp = curr_block_values[nnz] - curr_block_error[nnz];
      Error += err_tmp * err_tmp;
    }
    delete curr_block;
  }

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