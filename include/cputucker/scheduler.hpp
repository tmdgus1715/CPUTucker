#ifndef SCHEDULER_HPP_
#define SCHEDULER_HPP_

#include <vector>

#include "cputucker/tensor.hpp"

namespace supertensor {
namespace cputucker {

#define SCHEDULER_TEMPLATE \
  template <typename TensorType>
#define SCHEDULER_TEMPLATE_ARGS \
  TensorType

  SCHEDULER_TEMPLATE
  class Scheduler  {
    using tensor_t = TensorType;
    using index_t = typename tensor_t::index_t;
    using value_t = typename tensor_t::value_t;

  public:
    Scheduler() {
    }
    ~Scheduler() {
    }
    void Initialize();
    void Schedule(tensor_t *tensor);

  public:
    struct Task {
      uint64_t block_id;
      uint64_t nnz_count;
      uint64_t offset; // default: 0, if dense tensor is from 0 to iter - 1.

      Task(uint64_t new_block_id, uint64_t new_nnz_count)
          : block_id(new_block_id), nnz_count(new_nnz_count) {
      }
      void ToString() {
        printf("[%lu]-block \t %lu nnzs\n", block_id, nnz_count);
      }
    };

  public:
    std::vector<Task> tasks;
    uint64_t task_count;      // >= block_count
  };
  
}
}

#include "cputucker/scheduler.tpp"

#endif /* SCHEDULER_HPP_ */