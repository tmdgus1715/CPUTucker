#ifndef UPDATE_CUH_
#define UPDATE_CUH_

#include <omp.h>

#include <Eigen/Dense>

#include "cputucker/constants.hpp"
//#include "cputucker/delta.hpp"
#include "cputucker/helper.hpp"
#include "cputucker/ioasyncdelta.hpp"

namespace supertensor {
namespace cputucker {

template <typename TensorType, typename MatrixType, typename BlockType, typename ValueType>
void ComputingBlockBC(TensorType *tensor, MatrixType **B, MatrixType **C, int curr_factor_id, int rank, BlockType *curr_block, ValueType *curr_delta) {
    using tensor_t = TensorType;
    using index_t = typename tensor_t::index_t;
    using value_t = typename tensor_t::value_t;

    index_t ii, jj;
    uint64_t kk;

    index_t *block_dims = tensor->block_dims;
    index_t *part_dims = tensor->partition_dims;

    const index_t row_count = block_dims[curr_factor_id];

    // std::cout << "block [" << block_id << "] is being processed" << std::endl;
    index_t *curr_block_coord = curr_block->get_block_coord();

    index_t part_id = curr_block_coord[curr_factor_id];
    assert(part_id < part_dims[curr_factor_id]);

#pragma omp parallel for schedule(dynamic)  // schedule(auto)
    for (index_t row = 0; row < row_count; ++row) {
        uint64_t nnz = (curr_block->count_nnz[curr_factor_id][row + 1]) - (curr_block->count_nnz[curr_factor_id][row]);
        index_t where_ptr = curr_block->count_nnz[curr_factor_id][row];
        for (kk = 0; kk < nnz; ++kk) {
            index_t pos_curr_entry = curr_block->where_nnz[curr_factor_id][where_ptr + kk];
            value_t curr_entry_val = curr_block->values[pos_curr_entry];

            uint64_t pos_delta = pos_curr_entry * rank;
            uint64_t pos_B = row * rank * rank;
            uint64_t pos_C = row * rank;

            for (ii = 0; ii < rank; ++ii) {
                value_t cache = curr_delta[pos_delta + ii];

#pragma code_align 32
                for (jj = 0; jj < rank; ++jj) {
                    B[part_id][pos_B++] += cache * curr_delta[pos_delta + jj];
                }
                C[part_id][pos_C++] += cache * curr_entry_val;
            }
        }
    }
}

template <typename TensorType, typename MatrixType, typename TensorManagerType>
void ComputingBC(TensorType *tensor, MatrixType **B, MatrixType **C, int curr_factor_id, int rank, TensorManagerType *manager) {
    using tensor_t = TensorType;
    using block_t = typename tensor_t::block_t;
    using index_t = typename tensor_t::index_t;
    using value_t = typename tensor_t::value_t;
    using task_t = Task<block_t, value_t>;

    index_t *block_dims = tensor->block_dims;
    index_t *part_dims = tensor->partition_dims;

    const index_t row_count = block_dims[curr_factor_id];
    const uint64_t block_count = tensor->block_count;
    const int num_workers = 1;

    std::atomic<bool> production_stopped{false};
    std::mutex mtx;
    std::condition_variable go_produce, go_work;

    CircularQueue<task_t> block_queue(block_count);

    block_t *curr_block = nullptr;
    value_t *curr_delta = nullptr;
    task_t task;

    // Initialize B and C
    int k, l;
    for (uint64_t part_id = 0; part_id < part_dims[curr_factor_id]; ++part_id) {
#pragma omp parallel for schedule(static)
        for (index_t row = 0; row < row_count; ++row) {
            uint64_t pos_B = row * rank * rank;
            uint64_t pos_C = row * rank;
            for (k = 0; k < rank; ++k) {
                for (l = 0; l < rank; ++l) {
                    B[part_id][pos_B] = 0.0f;
                    if (k == l) {
                        B[part_id][pos_B] = cputucker::constants::kLambda;
                    }
                    ++pos_B;
                }
                C[part_id][pos_C] = 0.0f;
                ++pos_C;
            }
        }  // !omp parallel
    }      // !part_dims

    std::thread producer([&]() {
        for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
            curr_block = (block_t *)manager->ReadBlockFromFile(block_id);
            curr_delta = cputucker::allocate<value_t>(tensor->get_max_nnz_count_in_block() * rank);
            manager->ReadDeltaFromFile(curr_delta, tensor, curr_block, rank);

            task = {curr_block, curr_delta};

            {
                std::unique_lock<std::mutex> lock(mtx);
                go_produce.wait(lock, [&block_queue]() { return !block_queue.isFull(); });
                block_queue.enqueue(task);
                go_work.notify_one();
            }
        }

        production_stopped = true;
        go_work.notify_all();
    });

    std::vector<std::thread> workers;
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([&]() {
            while (true) {
                task_t task;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    go_work.wait(lock, [&]() { return !block_queue.isEmpty() || production_stopped; });
                    if (block_queue.isEmpty() && production_stopped) {
                        break;
                    }
                    block_queue.dequeue(task);
                    go_produce.notify_one();
                }
                ComputingBlockBC<TensorType, MatrixType, block_t, value_t>(tensor, B, C, curr_factor_id, rank, task.block, task.delta);
                delete task.block;
                delete task.delta;
            }
        });
    }

    producer.join();
    for (int i = 0; i < num_workers; ++i) {
        workers[i].join();
    }
}

template <typename TensorType, typename MatrixType, typename ValueType, typename SchedulerType, typename TensorManagerType>
void UpdateFactorMatrices(TensorType *tensor, TensorType *core_tensor, ValueType ***factor_matrices, MatrixType **B, MatrixType **C, int rank, SchedulerType *scheduler, TensorManagerType *manager) {
    using tensor_t = TensorType;
    using block_t = typename tensor_t::block_t;
    using index_t = typename tensor_t::index_t;
    using value_t = typename tensor_t::value_t;
    using matrix_t = Eigen::MatrixXd;

    int order = tensor->order;

    index_t *block_dims = tensor->block_dims;
    index_t *part_dims = tensor->partition_dims;

    for (int curr_factor_id = 0; curr_factor_id < order; ++curr_factor_id) {
        MYPRINT("[ Update factor matrix %d ]\n", curr_factor_id);

        double delta_time = omp_get_wtime();
        ComputingDelta<TensorType, ValueType, SchedulerType, TensorManagerType>(tensor, core_tensor, factor_matrices, curr_factor_id, rank, scheduler, manager);
        printf("\t- Elapsed time for Computing Delta: %lf\n", omp_get_wtime() - delta_time);

        double bc_time = omp_get_wtime();
        ComputingBC(tensor, B, C, curr_factor_id, rank, manager);
        printf("\t- Elapsed time for Computing B and C: %lf\n", omp_get_wtime() - bc_time);

        double update_time = omp_get_wtime();
        index_t row_count = block_dims[curr_factor_id];
        index_t col_count = rank;

        for (uint64_t part_id = 0; part_id < part_dims[curr_factor_id]; ++part_id) {
#pragma omp parallel for schedule(static)
            for (index_t row = 0; row < row_count; ++row) {
                // Getting the inverse matrix of [B + lambda * I]
                uint64_t pos_B = row * col_count * col_count;
                uint64_t pos_C = row * col_count;
                matrix_t BB(col_count, col_count);
                for (index_t k = 0; k < col_count; ++k) {
                    for (index_t l = 0; l < col_count; ++l) {
                        BB(k, l) = B[part_id][pos_B + k * col_count + l];
                    }
                }

                matrix_t B_inv = BB.inverse();

                index_t offset = row * col_count;
                for (index_t k = 0; k < col_count; ++k) {
                    value_t res = 0;
                    for (index_t l = 0; l < col_count; ++l) {
                        res += C[part_id][pos_C + l] * B_inv(l, k);
                    }
                    factor_matrices[curr_factor_id][part_id][offset + k] = res;
                }
                BB.resize(0, 0);
                B_inv.resize(0, 0);

            }  // row size
        }      // part_dims

        printf("\t- row-wise update TIME : %lf\n", omp_get_wtime() - update_time);

    }  // ! curr_factor
}

}  // namespace cputucker
}  // namespace supertensor

#endif /* UPDATE_CUH_ */