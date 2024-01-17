#ifndef TUCKER_CUH_
#define TUCKER_CUH_

#include <omp.h>

#include "cputucker/constants.hpp"
#include "cputucker/scheduler.hpp"
#include "cputucker/helper.hpp"
#include "cputucker/update.hpp"
#include "cputucker/reconstruction.hpp"


namespace supertensor {
namespace cputucker {
template <typename TensorType, typename OptimizerType, typename SchedulerType>
void TuckerDecomposition(TensorType *tensor, OptimizerType *optimizer,  SchedulerType *scheduler) {

  PrintLine();
  MYPRINT("Tucker Decomposition\n");
  PrintLine();

  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;
  using block_t = typename tensor_t::block_t;
  using optimizer_t = OptimizerType;
  using scheduler_t = SchedulerType;

  unsigned short order = tensor->order;
  index_t *dims = tensor->dims;
  index_t *block_dims = tensor->block_dims;
  index_t *partition_dims = tensor->partition_dims;

  int rank = optimizer->rank;

  value_t **factor_matrices[cputucker::constants::kMaxOrder];

  printf("\t Initializing factor matrices, core tensor, and intermediate data(delta, B, C, and error_T)\n");
  // Allocate sub_factor matrices
  for (int axis = 0; axis < order; ++axis) {
    factor_matrices[axis] = cputucker::allocate<value_t *>(sizeof(value_t *) * partition_dims[axis]);
    index_t sub_factor_row = block_dims[axis];
    for (index_t part = 0; part < partition_dims[axis]; ++part) {
      factor_matrices[axis][part] = cputucker::allocate<value_t>(sizeof(value_t) * sub_factor_row * rank);
    }
  }

  // Initialize sub_factor matrices
  for (int axis = 0; axis < order; ++axis) {
    index_t sub_factor_row = block_dims[axis];
    for (index_t part = 0; part < partition_dims[axis]; ++part) {
      if (part + 1 == partition_dims[axis]) {
        sub_factor_row = dims[axis] - part * block_dims[axis];
      }
      for (index_t row = 0; row < sub_factor_row; ++row) {
        for (int col = 0; col < rank; ++col) {
          // Random values between 0 and 1
          factor_matrices[axis][part][row * rank + col] = cputucker::frand<double>(0, 1);
        }
      }
    }
  }

  // Core tensor
  tensor_t *core_tensor = new tensor_t(order);
  index_t *core_dims = cputucker::allocate<index_t>(order);
  index_t *core_part_dims = cputucker::allocate<index_t>(order);
  uint64_t core_nnz_count = 1;
  for (int axis = 0; axis < order; ++axis) {
    core_dims[axis] = rank;
    core_part_dims[axis] = 1;
    core_nnz_count *= rank;
  }

  core_tensor->set_dims(core_dims);
  core_tensor->set_nnz_count(core_nnz_count);
  core_tensor->MakeBlocks(1, &core_nnz_count);

  block_t *curr_core_tensor_block = core_tensor->blocks[0];

  // #pragma omp parallel for
  for (uint64_t i = 0; i < core_nnz_count; ++i) {
    curr_core_tensor_block->values[i] = cputucker::frand<double>(0, 1);
    index_t mult = 1;
    for (short axis = order; --axis >= 0;) {
      index_t idx = 0;
      if (axis == order - 1) {
        idx = i % core_dims[axis];
      } else if (axis == 0) {
        idx = i / mult;
      } else {
        idx = (i / mult) % core_dims[axis];
      }
      curr_core_tensor_block->indices[axis][i] = idx;
      mult *= core_dims[axis];
    }
    assert(mult == core_nnz_count);
  }

  const uint64_t block_count = tensor->block_count;
  const index_t max_block_dim = tensor->get_max_block_dim();
  const index_t max_partition_dim = tensor->get_max_partition_dim();

  using matrix_t = double;
  value_t **delta = cputucker::allocate<value_t *>(block_count);
  matrix_t **B = cputucker::allocate<matrix_t *>(max_partition_dim);
  matrix_t **C = cputucker::allocate<matrix_t *>(max_partition_dim);
  value_t **error_T = cputucker::allocate<value_t *>(block_count);

  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    block_t *curr_block = tensor->blocks[block_id];
    delta[block_id] = cputucker::allocate<value_t>(curr_block->nnz_count * rank);
    error_T[block_id] = cputucker::allocate<value_t>(curr_block->nnz_count);
  }
  for (index_t part = 0; part < max_partition_dim; ++part) {
    B[part] = cputucker::allocate<matrix_t>(max_block_dim * rank * rank);
    C[part] = cputucker::allocate<matrix_t>(max_block_dim * rank);
  }


  int iter = 0;
  double p_fit = -1;
  double fit = -1;

  double avg_time = omp_get_wtime();

  while (1) {
    double itertime = omp_get_wtime(), steptime;
    steptime = itertime;
    cputucker::UpdateFactorMatrices<tensor_t, matrix_t, value_t, scheduler_t>(tensor, core_tensor, factor_matrices, delta, B, C, rank, scheduler);
    printf("Factor Time : %lf\n", omp_get_wtime() - steptime);

    steptime = omp_get_wtime();
    cputucker::Reconstruction<tensor_t, value_t, value_t, scheduler_t>(tensor, core_tensor, factor_matrices, &fit, error_T, rank, scheduler);
    printf("Recon Time : %lf\n\n", omp_get_wtime() - steptime);
    steptime = omp_get_wtime();

    ++iter;

    std::cout << "iter " << iter << "\t Fit: " << fit << std::endl;
    printf("iter%d :      Fit : %lf\tElapsed Time : %lf\n\n", iter, fit,  omp_get_wtime() - itertime);
    if (iter >= cputucker::constants::kMaxIteration || (p_fit != -1 && cputucker::abs<double>(p_fit - fit) <= cputucker::constants::kLambda)) {
      break;
    }
    p_fit = fit;
  }

  MYPRINT("DONE\n");
}

}  // namespace cputucker
}  // namespace supertensor

#endif /* TUCKER_CUH_ */