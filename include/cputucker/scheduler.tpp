#include "cputucker/constants.hpp"
#include "cputucker/helper.hpp"
#include "cputucker/scheduler.hpp"
#include "cputucker/tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace supertensor {
namespace cputucker {

SCHEDULER_TEMPLATE
void Scheduler<SCHEDULER_TEMPLATE_ARGS>::Initialize() {
  this->task_count = 0;
}

SCHEDULER_TEMPLATE
void Scheduler<SCHEDULER_TEMPLATE_ARGS>::Schedule(tensor_t *tensor) {
  // Dimension partitioning
  std::vector<uint64_t> sort_nnz_count, sort_block_id;
  sort_nnz_count.resize(tensor->block_count);
  sort_block_id.resize(tensor->block_count);

  for (uint64_t block_id = 0; block_id < tensor->block_count; ++block_id) {
    sort_block_id[block_id] = block_id;
    sort_nnz_count[block_id] = tensor->blocks[block_id]->nnz_count;
  }

  // Sort block id by the number of nonzeros
  std::sort(sort_block_id.begin(), sort_block_id.end(),
            [&](const uint64_t a, const uint64_t &b) {
              return (sort_nnz_count[a] < sort_nnz_count[b]);
            });

  this->task_count = 0;
  for (uint64_t block_id = 0; block_id < tensor->block_count; ++block_id) {
    tasks.push_back(Task(sort_block_id[block_id], sort_nnz_count[sort_block_id[block_id]]));
    ++this->task_count;
  }

}

}  // namespace cputucker
}  // namespace supertensor