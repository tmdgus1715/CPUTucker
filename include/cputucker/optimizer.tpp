#include <cassert>

#include "cputucker/constants.hpp"
#include "cputucker/helper.hpp"
#include "cputucker/optimizer.hpp"
#include "optimizer.hpp"

namespace supertensor {
namespace cputucker {

OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::Initialize(unsigned int new_rank, 
                                                    uint64_t new_mem_size,
                                                    tensor_t* new_data) {
  rank = new_rank;
  mem_size = new_mem_size;
  this->_data = new_data;
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::GetAllDataSize() {
  size_t ret_size = 0;

  ret_size += this->_get_data_size_sub_tensor();
  ret_size += this->_get_data_size_core_tensor();
  ret_size += this->_get_data_size_sub_factors();
  ret_size += this->_get_data_size_sub_delta();

  return ret_size;
}

OPTIMIZER_TEMPLATE
typename Optimizer<OPTIMIZER_TEMPLATE_ARGS>::index_t*
Optimizer<OPTIMIZER_TEMPLATE_ARGS>::FindPartitionParms() {
  MYPRINT("Find partition find_partition_parms\n");

  unsigned short order = this->_data->order;
  index_t* dims = this->_data->dims;

  block_dims = cputucker::allocate<index_t>(order);
  partition_dims = cputucker::allocate<index_t>(order);

  for (unsigned short axis = 0; axis < order; ++axis) {
    block_dims[axis] = dims[axis];
    //partition_dims[axis] = 1;
    partition_dims[axis] = 3;
  }
  this->_RefreshBlockDims();

  return partition_dims;
}


OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::ToString() {
  PrintLine();
  printf("< OPTIMIZER >\n");

  unsigned short order = this->_data->order;
  for (int axis = 0; axis < order; ++axis) {
    std::cout << "Partition dim [" << axis << "] = " << partition_dims[axis] << std::endl;
  }
  printf("The number of blocks: %lu\n", block_count);
  printf("Avg. nonzeros per a block: %lu\n", avg_nnz_count_per_block);
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_input_tensor() {
  size_t ret_size = this->_data->nnz_count *
                    (this->_data->order * sizeof(index_t) + sizeof(value_t));
  return ret_size;
}

// Calculates and returns the size of a sub-tensor
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_sub_tensor() {
  unsigned short order = this->_data->order;
  size_t ret_size =
      avg_nnz_count_per_block * (order * sizeof(index_t) + sizeof(value_t));

  return ret_size;
}

// Calculates and returns the size of a core tensor
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_core_tensor() {
  unsigned int order = this->_data->order;
  size_t core_nnz_count = std::pow(rank, order);
  size_t ret_size =
      core_nnz_count * (order * sizeof(index_t) + sizeof(value_t));
  return ret_size;
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_all_factors() {
  size_t ret_size = 0;
  for (unsigned short axis = 0; axis < this->_data->order; ++axis) {
    ret_size += this->_data->dims[axis] * rank;
  }
  return ret_size * sizeof(value_t);
}

// Calculates and returns the size of sub-factor matrices
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_sub_factors() {
  // sum of each sub-factor for the factor
  size_t ret_size = 0;
  for (unsigned short axis = 0; axis < this->_data->order; ++axis) {
    ret_size += block_dims[axis] * rank;
  }
  return ret_size * sizeof(value_t);
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_delta() {
  size_t ret_size = this->_data->nnz_count * rank * sizeof(value_t);
  return ret_size;
}
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_sub_delta() {
  unsigned short order = this->_data->order;
  size_t ret_size = this->avg_nnz_count_per_block * this->rank * sizeof(value_t);
  return ret_size;
}

/* Adjusting block dimensions using partition dimensions */
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_RefreshBlockDims() {
  // Initialize block dimensions
  int order = this->_data->order;
  index_t* dims = this->_data->dims;

  block_count = 1;
  for (unsigned short axis = 0; axis < order; ++axis) {
    block_dims[axis] =
        (dims[axis] + partition_dims[axis] - 1) / partition_dims[axis];
    index_t check_dim = (dims[axis] + block_dims[axis] - 1) / block_dims[axis];
    if (check_dim != partition_dims[axis]) {
      throw std::runtime_error(ERROR_LOG(
          "[ERROR] Block dimension is larger than the tensor dimension."));
    }
    block_count *= partition_dims[axis];
  }

  avg_nnz_count_per_block = (this->_data->nnz_count + block_count - 1) / block_count;
}
}  // namespace cputucker
}  // namespace supertensor
