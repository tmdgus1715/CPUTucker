#ifndef DELTA_CUH_
#define DELTA_CUH_


#include "cputucker/helper.hpp"
#include "cputucker/constants.hpp"
#include "cputucker/scheduler.hpp"

namespace supertensor {
namespace cputucker {

template <typename TensorType, typename ValueType, typename SchedulerType>
void ComputingDelta(TensorType *tensor, TensorType *core_tensor,
                    ValueType ***factor_matrices, ValueType **delta,
                    int curr_factor_id, int rank, SchedulerType *scheduler) {

  using tensor_t = TensorType;
  using block_t = typename tensor_t::block_t;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

  const int order = tensor->order;
  const uint64_t block_count = tensor->block_count;
  index_t *block_dims = tensor->block_dims;

  // Computing Blocks
  auto tasks = scheduler->tasks;
  printf("\t... Computing delta\n");

  for (uint64_t iter = 0; iter < tasks.size(); ++iter) {
    uint64_t block_id = tasks[iter].block_id;
    uint64_t nnz_count = tasks[iter].nnz_count;
    
    block_t *curr_block = tensor->blocks[block_id];
    index_t *curr_block_coord = curr_block->get_block_coord();

    index_t** core_indices = core_tensor->blocks[0]->indices;
    value_t* core_values = core_tensor->blocks[0]->values;

#pragma omp parallel for schedule(static)
    for (uint64_t nnz = 0; nnz < nnz_count; ++nnz) {
      index_t* nnz_idx = cputucker::allocate<index_t>(3);
      for (int axis = 0; axis < order; ++axis) {
        nnz_idx[axis] = curr_block->indices[axis][nnz];
      }

      for(int r = 0; r < rank; ++r){
        delta[block_id][nnz * rank + r] = 0.0f;
      }

      for(uint64_t co_nnz = 0; co_nnz < core_tensor->nnz_count; ++co_nnz) {
        index_t delta_col = core_indices[curr_factor_id][co_nnz];
        value_t beta = core_values[co_nnz];
        for(int axis = 0; axis < order; ++axis) {
          if(axis != curr_factor_id) {
            // index_t part_id = curr_block_coord[axis];
            // index_t pos = nnz_idx[axis] * rank + core_indices[axis][co_nnz];
            // beta *= factor_matrices[axis][part_id][pos];
            beta *= factor_matrices[axis][curr_block_coord[axis]][nnz_idx[axis] * rank + core_indices[axis][co_nnz]];
          }
        }
        delta[block_id][nnz * rank + delta_col] += beta;
      }
      cputucker::deallocate(nnz_idx);
    }

  }  // block_count loop
}

}  // namespace cputucker
}  // namespace supertensor

#endif /* DELTA_CUH_ */