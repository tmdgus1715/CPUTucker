#ifndef OPTIMIZER_HPP_
#define OPTIMIZER_HPP_

#include "common/human_readable.hpp"
#include "cputucker/constants.hpp"
#include "cputucker/tensor.hpp"

namespace supertensor {
namespace cputucker {

#define OPTIMIZER_TEMPLATE template <typename TensorType>
#define OPTIMIZER_TEMPLATE_ARGS TensorType

OPTIMIZER_TEMPLATE
class Optimizer {
  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

 public:
 public:
  Optimizer() {}
  ~Optimizer() {}
  void Initialize(unsigned short new_node_count, unsigned int new_rank,
                  uint64_t new_mem_size, tensor_t* new_data);

  size_t GetAllDataSize();
  index_t* FindPartitionParms();
  void ToString();

 private:
  void _DeterminePartitionType();  // which is the partition type for the input
                                   // tensor?

  /* Adjusting block dimensions using partition dimensions */
  void _RefreshBlockDims();
  void _AvailableNonzeroCountPerTask();

  /* Get data size for a CUDA execution sequence*/
  size_t _get_data_size_input_tensor();
  size_t _get_data_size_core_tensor();
  size_t _get_data_size_all_factors();
  size_t _get_data_size_delta();

  /* sub */
  size_t _get_data_size_sub_tensor();
  size_t _get_data_size_sub_factors();
  size_t _get_data_size_sub_delta();  // intermediate data size using available
                                      // nnz count;

 public:
  unsigned int rank;
  unsigned short node_count;
  uint64_t mem_size;

  index_t* block_dims;
  index_t* partition_dims;
  uint64_t block_count;

  uint64_t avg_nnz_count_per_block;  // (estimated) nonzeros per a task
  uint64_t avail_nnz_count_per_task;

 private:
  tensor_t* _data;
};
}  // namespace cputucker
}  // namespace supertensor

#include "cputucker/optimizer.tpp"
#endif /* OPTIMIZER_HPP_ */