#ifndef DELTA_CUH_
#define DELTA_CUH_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "cputucker/circlequeue.hpp"
#include "cputucker/constants.hpp"
#include "cputucker/helper.hpp"
#include "cputucker/scheduler.hpp"

namespace supertensor {
namespace cputucker {

template <typename BlockType, typename ValueType>
struct Task {
    BlockType *block;
    ValueType *delta;
};

template <typename TensorType, typename ValueType, typename BlockType>
void ComputingBlockDelta(TensorType *tensor, TensorType *core_tensor, ValueType ***factor_matrices, int curr_factor_id, int rank,
                         BlockType *curr_block, ValueType *curr_delta) {
    using tensor_t = TensorType;
    using index_t = typename tensor_t::index_t;
    using value_t = typename tensor_t::value_t;

    const int order = tensor->order;
    const uint64_t block_count = tensor->block_count;
    const uint64_t nnz_count = curr_block->nnz_count;
    index_t *block_dims = tensor->block_dims;

    index_t *curr_block_coord = curr_block->get_block_coord();
    index_t **core_indices = core_tensor->blocks[0]->indices;
    value_t *core_values = core_tensor->blocks[0]->values;

#ifdef OPENMP
#pragma omp parallel
    {
        index_t *nnz_idx = cputucker::allocate<index_t>(order);
#pragma omp for schedule(static)
        for (uint64_t nnz = 0; nnz < nnz_count; ++nnz) {
            for (int axis = 0; axis < order; ++axis) {
                nnz_idx[axis] = curr_block->indices[axis][nnz];
            }

            for (int r = 0; r < rank; ++r) {
                curr_delta[nnz * rank + r] = 0.0f;
            }

            for (uint64_t co_nnz = 0; co_nnz < core_tensor->nnz_count; ++co_nnz) {
                index_t delta_col = core_indices[curr_factor_id][co_nnz];
                value_t beta = core_values[co_nnz];
                for (int axis = 0; axis < order; ++axis) {
                    if (axis != curr_factor_id) {
                        // index_t part_id = curr_block_coord[axis];
                        // index_t pos = nnz_idx[axis] * rank + core_indices[axis][co_nnz];
                        // beta *= factor_matrices[axis][part_id][pos];
                        beta *= factor_matrices[axis][curr_block_coord[axis]][nnz_idx[axis] * rank + core_indices[axis][co_nnz]];
                    }
                }
                curr_delta[nnz * rank + delta_col] += beta;
                // to-do : write delta per block to disk
            }
        }
        cputucker::deallocate(nnz_idx);
    }
#endif

#ifdef AVX

    uint64_t sz_width = std::is_same<value_t, double>::value ? 8 : 16;

    uint64_t co_nnz_avx_size = ((core_tensor->nnz_count + sz_width - 1) / sz_width) * sz_width;
    assert(co_nnz_avx_size <= core_tensor->nnz_count);
    std::cout << "core nnz avx size: " << co_nnz_avx_size << std::endl;

#pragma omp parallel
    {
        index_t *nnz_idx = cputucker::allocate<index_t>(order);
#pragma omp for schedule(static)
        for (uint64_t nnz = 0; nnz < nnz_count; ++nnz) {
            for (int axis = 0; axis < order; ++axis) {
                nnz_idx[axis] = curr_block->indices[axis][nnz];
            }

            // Initialize delta to 0.0f for each element in the block
            for (int r = 0; r < rank; ++r) {
                curr_delta[nnz * rank + r] = 0.0f;
            }

            uint64_t co_nnz;
            for (co_nnz = 0; co_nnz < co_nnz_avx_size; co_nnz += sz_width) {  // Process 8 items at a time
                __m512d beta_v = _mm512_loadu_pd(&core_values[co_nnz]);
                __m256i nnz_idx_vv[order - 1];
                __m512d factor_vv[order - 1];

                for (int axis = 0, local_axis = 0; axis < order; ++axis) {
                    if (axis != curr_factor_id) {
                        index_t part_id = curr_block_coord[axis];

                        // Directly compute pos_v for the gather operation
                        // __m256i nnz_idx_v = _mm256_set1_epi32(nnz_idx[axis] * rank);
                        nnz_idx_vv[local_axis] = _mm256_set1_epi32(nnz_idx[axis] * rank);
                        __m256i core_idx_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&core_indices[axis][co_nnz]));
                        __m256i pos_v = _mm256_add_epi32(nnz_idx_vv[local_axis], core_idx_v);

                        // __m512d factor_v = _mm512_i32gather_pd(pos_v, &factor_matrices[axis][part_id][0], 8);
                        // beta_v = _mm512_mul_pd(beta_v, factor_v);  // Element-wise multiply
                        factor_vv[local_axis] = _mm512_i32gather_pd(pos_v, &factor_matrices[axis][part_id][0], 8);
                        local_axis++;
                    }
                }

                for (int local_axis = 0; local_axis < order - 1; ++local_axis) {
                    // SIMD 연산을 사용하여 요소별로 곱셈 수행
                    beta_v = _mm512_mul_pd(beta_v, factor_vv[local_axis]);
                }

                __m256i del_row_v = _mm256_set1_epi32(nnz * rank);
                __m256i del_col_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&core_indices[curr_factor_id][co_nnz]));
                __m256i offset_v = _mm256_add_epi32(del_row_v, del_col_v);

                uint32_t del_offs[sz_width];
                _mm256_storeu_si256((__m256i *)del_offs, offset_v);
                value_t del_vals[sz_width];
                _mm512_storeu_pd(del_vals, beta_v);

                // Update delta; assuming scalar update due to varying delta_col
                for (int i = 0; i < sz_width; ++i) {
                    curr_delta[del_offs[i]] += del_vals[i];
                }
            }
            // Remaining
            for (co_nnz = co_nnz_avx_size; co_nnz < core_tensor->nnz_count; ++co_nnz) {
                value_t beta = core_values[co_nnz];
                for (int axis = 0; axis < order; ++axis) {
                    if (axis != curr_factor_id) {
                        // index_t part_id = curr_block_coord[axis];
                        // index_t pos = nnz_idx[axis] * rank + core_indices[axis][co_nnz];
                        // beta *= factor_matrices[axis][part_id][pos];
                        beta *= factor_matrices[axis][curr_block_coord[axis]][nnz_idx[axis] * rank + core_indices[axis][co_nnz]];
                    }
                }
                index_t delta_col = core_indices[curr_factor_id][co_nnz];
                curr_delta[nnz * rank + delta_col] += beta;
            }
        }
        cputucker::deallocate(nnz_idx);
    }
#endif
}

template <typename TensorType, typename ValueType, typename SchedulerType, typename TensorManagerType>
void ComputingDelta(TensorType *tensor, TensorType *core_tensor, ValueType ***factor_matrices, int curr_factor_id, int rank, SchedulerType *scheduler, TensorManagerType *manager) {
    using tensor_t = TensorType;
    using block_t = typename tensor_t::block_t;
    using index_t = typename tensor_t::index_t;
    using value_t = typename tensor_t::value_t;
    using task_t = Task<block_t, value_t>;

    auto tasks = scheduler->tasks;

    const uint64_t block_count = tensor->block_count;
    const int num_workers = 2;

    std::atomic<bool> production_stopped{false};
    std::atomic<bool> worker_stopped{false};

    std::mutex mtx;
    std::condition_variable go_produce, go_work1, go_work2, go_consume;

    CircularQueue<task_t> block_queue(block_count);
    CircularQueue<task_t> delta_queue(block_count);

    block_t *curr_block = nullptr;
    value_t *curr_delta = nullptr;
    task_t task;

    printf("\t... Computing delta\n");

    std::thread producer([&]() {
        for (uint64_t iter = 0; iter < tasks.size(); ++iter) {
            uint64_t block_id = tasks[iter].block_id;
            curr_block = (block_t *)manager->ReadBlockFromFile(block_id);
            curr_delta = cputucker::allocate<value_t>(tensor->get_max_nnz_count_in_block() * rank);
            task = {curr_block, curr_delta};

            {
                std::unique_lock<std::mutex> lock(mtx);
                go_produce.wait(lock, [&block_queue]() { return !block_queue.isFull(); });
                block_queue.enqueue(task);
                go_work1.notify_one();
            }
        }

        production_stopped = true;
        go_work1.notify_all();
    });

    std::vector<std::thread> workers;
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([&]() {
            while (true) {
                task_t task;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    go_work1.wait(lock, [&]() { return !block_queue.isEmpty() || production_stopped; });
                    if (block_queue.isEmpty() && production_stopped) {
                        break;
                    }
                    block_queue.dequeue(task);
                    go_produce.notify_one();
                }
                ComputingBlockDelta<TensorType, ValueType, block_t>(tensor, core_tensor, factor_matrices, curr_factor_id, rank, task.block, task.delta);
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    go_work2.wait(lock, [&]() { return !delta_queue.isFull(); });
                    delta_queue.enqueue(task);
                    go_consume.notify_all();
                }
            }

            worker_stopped = true;
            go_consume.notify_all();
        });
    }

    std::thread consumer([&]() {
        while (true) {
            task_t task;
            {
                std::unique_lock<std::mutex> lock(mtx);

                go_consume.wait(lock, [&]() { return !delta_queue.isEmpty() || worker_stopped; });
                if (delta_queue.isEmpty() && worker_stopped) {
                    break;
                }
                delta_queue.dequeue(task);
                go_work2.notify_one();
            }

            manager->WriteDeltaToFile(tensor, task.block, task.delta, rank);
            delete task.block;
            delete task.delta;
        }
    });

    producer.join();
    for (int i = 0; i < num_workers; ++i) {
        workers[i].join();
    }
    consumer.join();
}

}  // namespace cputucker
}  // namespace supertensor

#endif /* DELTA_CUH_ */