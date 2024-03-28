// Purpose: Source file for Block class
#include <cassert>

#include "cputucker/block.hpp"
#include "cputucker/constants.hpp"
#include "cputucker/helper.hpp"

namespace supertensor {
namespace cputucker {
BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::Block() : Block(0, NULL, 0, NULL, 0) {}

BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::Block(uint64_t new_block_id, unsigned short new_order) : Block(new_block_id, NULL, new_order, NULL, 0) {}

BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::Block(uint64_t new_block_id, index_t *new_block_coord, unsigned short new_order, index_t *new_dims, uint64_t new_nnz_count) {
    if (new_order < 1) {
        throw std::runtime_error(ERROR_LOG("[ERROR] Block order should be larger than 1."));
    }
    order = new_order;
    dims = cputucker::allocate<index_t>(this->order);
    nnz_count = new_nnz_count;
    this->_block_id = new_block_id;
    this->_base_dims = cputucker::allocate<index_t>(this->order);
    this->_block_coord = cputucker::allocate<index_t>(this->order);

    for (int axis = 0; axis < order; ++axis) {
        dims[axis] = new_dims[axis];
        this->_block_coord[axis] = new_block_coord[axis];  // for setting base_dims
        this->_base_dims[axis] = dims[axis] * new_block_coord[axis];
    }

    this->buffer_ptr = nullptr;
    this->_is_allocated = false;
}

BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::~Block() {
    if (_is_allocated) {
        cputucker::deallocate<index_t>(dims);
        cputucker::deallocate<index_t>(this->_base_dims);
        cputucker::deallocate<index_t>(this->_block_coord);

        if (this->buffer_ptr != nullptr) {
            delete[] this->buffer_ptr;
        } else {
            for (unsigned short axis = 0; axis < order; ++axis) {
                cputucker::deallocate<index_t>(indices[axis]);
                if (!is_input_block) {
                    cputucker::deallocate<uint64_t>(count_nnz[axis]);
                    cputucker::deallocate<index_t>(where_nnz[axis]);
                }
            }
            cputucker::deallocate<value_t>(values);
        }   

        this->_is_allocated = false;
        nnz_count = 0;
    }
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::InsertNonzero(uint64_t pos, index_t *new_coord, value_t new_value) {
    assert(pos <= nnz_count);
    for (unsigned short axis = 0; axis < order; ++axis) {
        indices[axis][nnz_count - pos] = new_coord[axis] - this->_base_dims[axis];
    }
    values[nnz_count - pos] = new_value;
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::AssignIndicesToEachMode() {
    assert(indices != NULL);

    uint64_t *temp_nnz[cputucker::constants::kMaxOrder];

    // Loop through each axis and allocate memory for temporary variables to store
    // the number of non-zero elements, as well as where they are located
    for (unsigned short axis = 0; axis < order; ++axis) {
        temp_nnz[axis] = cputucker::allocate<uint64_t>((this->dims[axis] + 1));
        count_nnz[axis] = cputucker::allocate<uint64_t>((this->dims[axis] + 1));
        where_nnz[axis] = cputucker::allocate<index_t>(this->nnz_count);
    }

    // Loop through each axis and initialize temporary arrays to zero
    for (unsigned short axis = 0; axis < order; ++axis) {
        for (index_t k = 0; k < this->dims[axis]; ++k) {
            count_nnz[axis][k] = 0;
            temp_nnz[axis][k] = 0;
        }
    }

    // Loop through each axis and non-zero element,
    // and count the number of non-zero elements for each axis
    for (unsigned short axis = 0; axis < order; ++axis) {
        for (uint64_t nnz = 0; nnz < this->nnz_count; ++nnz) {
            index_t k = this->indices[axis][nnz];
            assert(k < dims[axis]);
            count_nnz[axis][k]++;
            temp_nnz[axis][k]++;
        }
    }

    index_t now = 0;
    index_t k;
    index_t j = 0;

    // Loop through each axis and calculate the starting index of each block
    for (unsigned short axis = 0; axis < order; ++axis) {
        now = 0;
        uint64_t max_count = 0;

        for (j = 0; j < dims[axis]; ++j) {
            k = count_nnz[axis][j];
            if (max_count < k) {
                max_count = k;
            }
            count_nnz[axis][j] = now;
            temp_nnz[axis][j] = now;
            now += k;
        }

        count_nnz[axis][j] = now;
        temp_nnz[axis][j] = now;
    }

    // Loop through each axis and non-zero element,
    // and store where each non-zero element is located
    for (unsigned short axis = 0; axis < order; ++axis) {
        uint64_t sum_idx = 0;
        for (uint64_t nnz = 0; nnz < nnz_count; ++nnz) {
            k = indices[axis][nnz];
            now = temp_nnz[axis][k];
            where_nnz[axis][now] = nnz;
            temp_nnz[axis][k]++;
            sum_idx += k;
        }
    }
    // Deallocates
    for (unsigned short axis = 0; axis < order; ++axis) {
        cputucker::deallocate<uint64_t>(temp_nnz[axis]);
    }
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::AllocateData() {
    assert(nnz_count != 0);
    assert(this->_is_allocated == false);

    // Allocate memory for indices in each axis
    for (unsigned short axis = 0; axis < order; ++axis) {
        indices[axis] = cputucker::allocate<index_t>(nnz_count);
    }
    // Allocate memory for values
    values = cputucker::allocate<value_t>(nnz_count);
    this->_is_allocated = true;
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::ToString() {
    printf("********** BLOCK[%lu] Information **********\n", this->_block_id);

    int axis;

    printf("Block coord: ");
    for (axis = 0; axis < order; ++axis) {
        std::cout << "[" << this->_block_coord[axis] << "]";
    }
    printf("\n");

    printf("Block order: %d\n", order);

    printf("Block dims: ");
    for (axis = 0; axis < order; ++axis) {
        std::cout << dims[axis] << " ";
        if (axis < order - 1) {
            printf(" X ");
        } else {
            printf("\n");
        }
    }

    printf("# nnzs: %lu\n", nnz_count);
    PrintLine();
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::set_dims(index_t *new_dims) {
    for (int axis = 0; axis < order; ++axis) {
        dims[axis] = new_dims[axis];
        this->_base_dims[axis] = dims[axis] * this->_block_coord[axis];
    }
}

BLOCK_TEMPLATE
bool Block<BLOCK_TEMPLATE_ARGS>::equals(const this_t &other) const {
    // 비교할 Block이 자기 자신인지 확인
    if (this == &other) return true;

    // 기본 메타데이터 비교
    if (this->order != other.order || this->nnz_count != other.nnz_count || this->_block_id != other._block_id || this->_is_allocated != other._is_allocated) {
        return false;
    }

    // dims 배열 비교
    for (unsigned short i = 0; i < this->order; ++i) {
        if (this->dims[i] != other.dims[i]) {
            return false;
        }
    }

    // values 배열 비교
    for (uint64_t i = 0; i < this->nnz_count; ++i) {
        if (this->values[i] != other.values[i]) {
            return false;
        }
    }

    // indices 배열 비교
    for (unsigned short i = 0; i < this->order; ++i) {
        for (uint64_t j = 0; j < this->nnz_count; ++j) {
            if (this->indices[i][j] != other.indices[i][j]) {
                return false;
            }
        }
    }

    // where_nnz 배열 비교
    for (unsigned short i = 0; i < this->order; ++i) {
        for (uint64_t j = 0; j < this->nnz_count; ++j) {
            if (this->where_nnz[i][j] != other.where_nnz[i][j]) {
                return false;
            }
        }
    }

    // count_nnz 배열 비교
    for (unsigned short i = 0; i < this->order; ++i) {
        for (index_t j = 0; j < this->dims[i] + 1; ++j) {
            if (this->count_nnz[i][j] != other.count_nnz[i][j]) {
                return false;
            }
        }
    }

    // 모든 검사를 통과했으면 두 Block이 동일하다고 판단
    return true;
}

}  // namespace cputucker
}  // namespace supertensor