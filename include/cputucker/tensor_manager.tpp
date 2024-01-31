#include <omp.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "common/human_readable.hpp"
#include "cputucker/helper.hpp"
#include "cputucker/tensor_manager.hpp"

namespace supertensor
{
  namespace cputucker
  {

    TENSOR_MANAGER_TEMPLATE
    TensorManager<TENSOR_MANAGER_ARGS>::TensorManager(Config *config)
    {
      this->config = config;
    }

    TENSOR_MANAGER_TEMPLATE
    TensorManager<TENSOR_MANAGER_ARGS>::TensorManager() {}

    /*
     * @brief Parse the input tensor from a file
     * @param file_name The name of the file containing the tensor
     * @return True if the tensor is parsed successfully, false otherwise
     */
    TENSOR_MANAGER_TEMPLATE
    bool TensorManager<TENSOR_MANAGER_ARGS>::ParseFromFile(const std::string &file_name, tensor_t **tensor)
    {
      std::ifstream file(file_name);
      if (!file.is_open())
      {
        std::string err_msg = "[ERROR] Cannot open file \"" + file_name + "\" for reading...";
        throw std::runtime_error(ERROR_LOG(err_msg));
        return false;
      }
      file.seekg(0, file.end);

      size_t file_size = static_cast<size_t>(file.tellg());
      assert(file_size > 0);
      std::cout << "Input Tensor Size (COO) \t: "
                << common::HumanReadable{(std::uintmax_t)file_size} << std::endl;

      file.seekg(0, file.beg);
      std::string buffer(file_size, '\0');
      file.read(&buffer[0], file_size);
      file.close();

      return this->_ReadData(buffer.c_str(), file_size, tensor);
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_ARGS>::WriteBlockToFile(tensor_t *tensor)
    {
      const std::string tensor_md_path = config->getFilePath("tensor_metadata");
      const std::string block_md_path = config->getFilePath("block_metadata");
      const std::string data_path = config->getFilePath("block_data");

      std::ofstream tensor_md_file(tensor_md_path, std::ios::trunc);
      if (!tensor_md_file.is_open())
      {
        throw std::runtime_error("[ERROR] Cannot open file \"" + tensor_md_path +
                                 "\" for writing...");
      }

      std::ofstream block_md_file(block_md_path, std::ios::trunc);
      if (!block_md_file.is_open())
      {
        throw std::runtime_error("[ERROR] Cannot open file \"" + block_md_path +
                                 "\" for writing...");
      }

      std::ofstream data_file(data_path, std::ios::trunc | std::ios::binary);
      if (!data_file.is_open())
      {
        throw std::runtime_error("[ERROR] Cannot open file \"" + data_path +
                                 "\" for writing...");
      }

      uint64_t count_nnz_size = 0;
      for (ushort i = 0; i < tensor->order; ++i)
      {
        count_nnz_size += (tensor->block_dims[i] + 1) * sizeof(uint64_t); // count_nnz에서 각 모드의 사이즈는 block_dim[i] + 1
      }

      this->fixed_data_size = (sizeof(value_t) + 2 * sizeof(index_t) * tensor->order) * tensor->get_max_nnz_count_in_block() + count_nnz_size;

      _WriteTensorMetadataToFile(tensor_md_file, tensor);
      for (ushort i = 0; i < tensor->block_count; ++i)
      {
        block_t *block = tensor->blocks[i];
        _WriteBlockMetadataToFile(block_md_file, block);
        _WriteBlockDataToFile(data_file, block);
      }

      tensor_md_file.close();
      block_md_file.close();
      data_file.close();
    }
    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_ARGS>::_WriteTensorMetadataToFile(std::ofstream &tensor_md_file, tensor_t *tensor)
    {
      // Tensor Description
      tensor_md_file << tensor->order << " ";
      tensor_md_file << tensor->nnz_count << " ";
      tensor_md_file << tensor->norm << " ";
      for (unsigned short i = 0; i < tensor->order; ++i)
      {
        tensor_md_file << tensor->dims[i] << " ";
      }

      // Block Description
      tensor_md_file << tensor->block_count << " ";
      for (unsigned short i = 0; i < tensor->order; ++i)
      {
        tensor_md_file << tensor->partition_dims[i] << " ";
      }
      for (unsigned short i = 0; i < tensor->order; ++i)
      {
        tensor_md_file << tensor->block_dims[i] << " ";
      }

      // Additional Metadata
      tensor_md_file << tensor->get_max_nnz_count_in_block() << " ";
      tensor_md_file << tensor->get_max_block_dim() << " ";
      tensor_md_file << tensor->get_max_partition_dim() << std::endl;
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_ARGS>::_WriteBlockMetadataToFile(std::ofstream &block_md_file, block_t *block)
    {
      ushort order = block->order;
      block_md_file << block->get_block_id() << " " << order << " "
                    << block->nnz_count << " ";

      for (ushort i = 0; i < order; ++i)
      {
        block_md_file << block->get_block_coord()[i] << " ";
      }
      for (ushort i = 0; i < order; ++i)
      {
        block_md_file << block->dims[i] << " ";
      }
      block_md_file << std::endl;
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_ARGS>::_WriteBlockDataToFile(std::ofstream &data_file, block_t *block)
    {
      value_t *values = block->values;
      index_t **indices = block->indices;
      index_t **where_nnz = block->where_nnz;
      uint64_t **count_nnz = block->count_nnz;
      uint64_t nnz_count = block->nnz_count;
      ushort order = block->order;

      data_file.seekp(block->get_block_id() * this->fixed_data_size);

      data_file.write((char *)values, sizeof(value_t) * nnz_count);

      for (ushort i = 0; i < order; ++i)
      {
        data_file.write((char *)where_nnz[i], sizeof(index_t) * nnz_count);
      }

      for (ushort i = 0; i < order; ++i)
      {
        data_file.write((char *)indices[i], sizeof(index_t) * nnz_count);
      }

      uint64_t curr_count_nnz_size = 0;
      for (ushort i = 0; i < order; ++i)
      {
        data_file.write((char *)count_nnz[i], sizeof(uint64_t) * (block->dims[i] + 1));
        curr_count_nnz_size += (block->dims[i] + 1) * sizeof(uint64_t);
      }

      uint64_t curr_data_size = 0;
      curr_data_size += (sizeof(value_t) + 2 * order * sizeof(index_t)) * nnz_count + curr_count_nnz_size;

      uint64_t null_size = fixed_data_size - curr_data_size;
      if (null_size > 0)
      {
        char *null_ary = new char[null_size]();
        data_file.write(null_ary, null_size);
        delete[] null_ary;
      }
    }

    TENSOR_MANAGER_TEMPLATE
    void *TensorManager<TENSOR_MANAGER_ARGS>::ReadTensorFromFile()
    {
      const std::string tensor_md_path = config->getFilePath("tensor_metadata");
      std::ifstream tensor_md_file(tensor_md_path);

      if (!tensor_md_file.is_open())
      {
        throw std::runtime_error("[ERROR] Cannot open file \"" + tensor_md_path + "\" for reading...");
      }

      unsigned short order;
      uint64_t nnz_count, block_count;
      value_t norm;
      uint64_t max_nnz_count_in_block, empty_block_count;
      index_t max_block_dim, max_partition_dim;

      //Tensor Description
      tensor_md_file >> order >> nnz_count >> norm;
      index_t *dims = new index_t[order];
      for (unsigned short i = 0; i < order; ++i)
      {
        tensor_md_file >> dims[i];
      }

      //Block Description
      tensor_md_file >> block_count;
      index_t *partition_dims = new index_t[order];
      index_t *block_dims = new index_t[order];
      for (unsigned short i = 0; i < order; ++i)
      {
        tensor_md_file >> partition_dims[i];
      }
      for (unsigned short i = 0; i < order; ++i)
      {
        tensor_md_file >> block_dims[i];
      }

      //Additional Metadata
      tensor_md_file >> max_nnz_count_in_block >> max_block_dim >> empty_block_count >> max_partition_dim;

      tensor_t *tensor = new tensor_t(order);
      tensor->set_dims(dims);
      tensor->set_partition_dims(partition_dims);
      tensor->set_nnz_count(nnz_count);
      tensor->set_nnz_count_in_block(max_nnz_count_in_block);
      tensor->norm = norm;
      tensor->block_count = block_count;

      tensor_md_file.close();
      return tensor;
    }

    // block 사용 후 delete필요
    TENSOR_MANAGER_TEMPLATE void *
    TensorManager<TENSOR_MANAGER_ARGS>::ReadBlockFromFile(uint64_t block_id)
    {
      const std::string block_md_path = config->getFilePath("block_metadata");
      const std::string data_path = config->getFilePath("block_data");

      std::ifstream block_md_file(block_md_path);
      std::string line;
      bool block_found = false;
      uint64_t found_block_id, nnz_count;
      ushort order;
      index_t dims[constants::kMaxOrder];
      index_t block_coord[constants::kMaxOrder];

      while (std::getline(block_md_file, line))
      {
        std::istringstream iss(line);
        iss >> found_block_id;

        if (found_block_id == block_id)
        {
          iss >> order >> nnz_count;
          block_found = true;

          for (ushort i = 0; i < order; ++i)
          {
            iss >> block_coord[i];
          }

          for (ushort i = 0; i < order; ++i)
          {
            iss >> dims[i];
          }
          break;
        }
      }

      if (!block_found)
      {
        throw std::runtime_error("Cannot find block");
      }

      std::ifstream data_file(data_path, std::ios::binary);
      if (!data_file)
      {
        throw std::runtime_error("Cannot open data file.");
      }

      uint64_t block_offset = fixed_data_size * block_id;
      char *buffer = new char[fixed_data_size];

      data_file.seekg(block_offset, std::ios::beg);
      data_file.read(buffer, fixed_data_size);
      char *ptr = buffer;

      value_t *values = reinterpret_cast<value_t *>(ptr);
      ptr += sizeof(value_t) * nnz_count;

      index_t *where_nnz[constants::kMaxOrder];
      for (ushort i = 0; i < order; ++i)
      {
        where_nnz[i] = reinterpret_cast<index_t *>(ptr);
        ptr += sizeof(index_t) * nnz_count;
      }

      index_t *indices[constants::kMaxOrder];
      for (ushort i = 0; i < order; ++i)
      {
        indices[i] = reinterpret_cast<index_t *>(ptr);
        ptr += sizeof(index_t) * nnz_count;
      }

      uint64_t *count_nnz[constants::kMaxOrder];
      for (ushort i = 0; i < order; ++i)
      {
        count_nnz[i] = reinterpret_cast<uint64_t *>(ptr);
        ptr += sizeof(uint64_t) * (dims[i] + 1);
      }

      block_t *new_block = new block_t(found_block_id, block_coord, order, dims, nnz_count);
      new_block->values = values;
      for (ushort i = 0; i < order; ++i)
      {
        new_block->where_nnz[i] = where_nnz[i];
        new_block->indices[i] = indices[i];
        new_block->count_nnz[i] = count_nnz[i];
      }

      new_block->buffer_ptr = buffer;
      new_block->set_is_allocated(true);

      data_file.close();

      return new_block;
    }

    /*
     * @brief Parse the input tensor from a string
     * @param buffer The string containing the tensor
     * @param buffer_length The length of the string
     * @return True if the tensor is parsed successfully, false otherwise
     */
    TENSOR_MANAGER_TEMPLATE
    bool TensorManager<TENSOR_MANAGER_ARGS>::_ReadData(const char *buffer,
                                                       const size_t buffer_length,
                                                       tensor_t **tensor)
    {
      int thread_id = 0;
      int thread_count = 0;
      int order = (*tensor)->order;

      std::vector<uint64_t> *pos;
      std::vector<index_t> *local_max_dims;
      std::vector<index_t> *local_dim_offset;
      std::vector<uint64_t> nnz_prefix_sum;

      index_t *global_max_dims;
      uint64_t global_nnz_count = 0;

      value_t *values;
      index_t *indices[order];

#pragma omp parallel private(thread_id)
      {
        thread_id = omp_get_thread_num();
        thread_count = omp_get_num_threads();

// Initialize local variables
#pragma omp single
        {
          pos = new std::vector<uint64_t>[thread_count];
          local_max_dims = new std::vector<index_t>[thread_count];
          local_dim_offset = new std::vector<index_t>[thread_count];
          nnz_prefix_sum.resize(thread_count);
        }
        pos[thread_id].push_back(0);
        local_max_dims[thread_id].resize(order);
        local_dim_offset[thread_id].resize(order);

        for (ushort axis = 0; axis < order; ++axis)
        {
          local_max_dims[thread_id][axis] = std::numeric_limits<index_t>::min();
          local_dim_offset[thread_id][axis] = std::numeric_limits<index_t>::max();
        }
        // 1. Find '\n' : the number of nonzeros
#pragma omp for reduction(+ : global_nnz_count)
        for (size_t sz = 0; sz < buffer_length; ++sz)
        {
          if (buffer[sz] == '\n')
          {
            global_nnz_count++;
            pos[thread_id].push_back(sz + 1);
          }
        }

#pragma omp barrier
        if (thread_id > 0)
        {
          pos[thread_id].front() = pos[thread_id - 1].back();
        }
#pragma omp barrier
#pragma omp single
        {
          // prefix sum
          nnz_prefix_sum[0] = 0;
          for (int tid = 1; tid < thread_count; ++tid)
          {
            nnz_prefix_sum[tid] = nnz_prefix_sum[tid - 1] + (pos[tid - 1].size() - 1);
          }
          assert(nnz_prefix_sum.back() + pos[thread_count - 1].size() - 1 == global_nnz_count);

          global_max_dims = cputucker::allocate<index_t>(order);
          for (ushort axis = 0; axis < order; ++axis)
          {
            global_max_dims[axis] = std::numeric_limits<index_t>::min();
            indices[axis] = cputucker::allocate<index_t>(global_nnz_count);
          }
          values = cputucker::allocate<value_t>(global_nnz_count);
        }

        for (uint64_t nnz = 1; nnz < pos[thread_id].size(); ++nnz)
        {
          // Calculate the starting position of the current slice in the buffer
          const int len = pos[thread_id][nnz] - pos[thread_id][nnz - 1] - 1;
          uint64_t buff_ptr = pos[thread_id][nnz - 1];
          char *buff = const_cast<char *>(&buffer[buff_ptr]);

          // Tokenize	the slice by newline characters
          char *rest = strtok_r(buff, "\n", &buff);
          char *token;

          if (rest != NULL)
          {
            // Calculate the offset of the current thread in the global index
            uint64_t offset = nnz_prefix_sum[thread_id];
            ushort axis = 0;
            /* Coordinate */
            // Loop through each coordinate in the slice
            while ((token = strtok_r(rest, " \t", &rest)) && (axis < order))
            {
              index_t idx = strtoull(token, NULL, 10);

              // Update the maximum and minimum indices for the current axis
              local_max_dims[thread_id][axis] = std::max<index_t>(local_max_dims[thread_id][axis], idx);
              local_dim_offset[thread_id][axis] = std::min<index_t>(local_dim_offset[thread_id][axis], idx);

              // Store the current index in the global indices array,
              // with 1-indexing (subtract 1 from idx)
              indices[axis][offset + nnz - 1] = idx - 1; // 1-Indexing
              ++axis;
            } // !while

            /* Value */
            // Parse the value of the current slice
            value_t val;
            if (token != NULL)
            {
              val = std::stod(token);
            }
            else
            {
              // If the slice does not have a value, generate a random one between 0 and 1
              val = cputucker::frand<value_t>(0, 1);
            }
            values[offset + nnz - 1] = val;
          }
        } // !for

// 2. extract metadata for the tensor (dims, offsets, and #nnzs)
#pragma omp critical
        {
          // Update the global max dimension for the axis
          for (ushort axis = 0; axis < order; ++axis)
          {
            global_max_dims[axis] = std::max<index_t>(global_max_dims[axis], local_max_dims[thread_id][axis]);
            if (local_dim_offset[thread_id][axis] < 1)
            {
              // outputs are based on base-0 indexing
              throw std::runtime_error(ERROR_LOG("We note that input tensors must follow base-1 indexing"));
            }
          }
        } // !omp critical
      }   //! omp

      uint64_t block_id = 0;

      (*tensor)->set_dims(global_max_dims);
      (*tensor)->set_nnz_count(global_nnz_count);

      (*tensor)->MakeBlocks(1, &global_nnz_count);
      (*tensor)->InsertData(block_id, &indices[0], values);
      (*tensor)->blocks[block_id]->ToString();

      // Deallocate
      delete[] pos;
      delete[] local_max_dims;
      delete[] local_dim_offset;
      nnz_prefix_sum.clear();
      std::vector<uint64_t>().swap(nnz_prefix_sum);

      return true;
    }

    TENSOR_MANAGER_TEMPLATE
    template <typename OptimizerType>
    void TensorManager<TENSOR_MANAGER_ARGS>::CreateTensorBlocks(tensor_t **src, tensor_t **dest, OptimizerType *optimizer)
    {
      PrintLine();
      MYPRINT("CreateTensorBlocks\n");
      PrintLine();

      printf("... 1) Creating tensor blocks\n");
      const ushort order = (*src)->order;
      const index_t *const dims = (*src)->dims;
      const uint64_t nnz_count = (*src)->nnz_count;

      const index_t *const block_dims = optimizer->block_dims;
      const index_t *const partition_dims = optimizer->partition_dims;
      const uint64_t block_count = optimizer->block_count;

      index_t **indices = (*src)->blocks[0]->indices;
      value_t *values = (*src)->blocks[0]->values;

      // 1. Count nonzeros per block
      std::vector<std::vector<uint64_t>> local_nnz_histograms(omp_get_max_threads(), std::vector<uint64_t>(block_count, 0));
      std::vector<std::vector<index_t>> local_nnz_coords(omp_get_max_threads(), std::vector<index_t>(order, 0));

      printf("... 2) Counting nonzeros per block\n");
#pragma omp parallel
      {
        const int thread_id = omp_get_thread_num();
        const int thread_count = omp_get_num_threads();

#pragma omp for
        for (uint64_t nnz = 0; nnz < nnz_count; ++nnz)
        {
          // Convert coordinates of nonzero into block id
          uint64_t block_id = 0;
          uint64_t mult = 1;
          for (ushort iter = 0; iter < order; ++iter)
          {
            ushort axis = order - iter - 1;

            assert(indices[axis][nnz] < dims[axis] && "Coordinate is out of bounds");
            index_t block_idx = indices[axis][nnz] / block_dims[axis];
            assert(block_idx < partition_dims[axis] && "Block coordinate is out of bounds");
            block_id += block_idx * mult;
            mult *= partition_dims[axis];
          }
          assert(block_id < block_count);
          ++local_nnz_histograms[thread_id][block_id];
        } // !omp for
      }   // omp parallel

      printf("... 3) Creating blocks\n");
      uint64_t check_nnz_count = 0;
      std::vector<uint64_t> global_nnz_histogram(block_count, 0);
      for (int tid = 0; tid < omp_get_max_threads(); ++tid)
      {
        for (uint64_t block_id = 0; block_id < block_count; ++block_id)
        {
          global_nnz_histogram[block_id] += local_nnz_histograms[tid][block_id];
          check_nnz_count += local_nnz_histograms[tid][block_id];
        }
      }

      assert(check_nnz_count == nnz_count);
      (*dest)->set_partition_dims(partition_dims);
      (*dest)->MakeBlocks(block_count, global_nnz_histogram.data());

      printf("... 4) Inserting data\n");
      value_t NormX = 0.0f;
      omp_lock_t lck;
      omp_init_lock(&lck);

      for (uint64_t nnz = 0; nnz < nnz_count; ++nnz)
      {
        std::vector<index_t> local_tensor_coord(order, 0);
        value_t val = values[nnz];
        uint64_t block_id = 0;
        uint64_t mult = 1;
        for (ushort iter = 0; iter < order; ++iter)
        {
          ushort axis = order - iter - 1;

          assert(indices[axis][nnz] < dims[axis] && "Coordinate is out of bounds");
          local_tensor_coord[axis] = indices[axis][nnz];
          index_t block_idx = indices[axis][nnz] / block_dims[axis];

          assert(block_idx < partition_dims[axis] && "Block coordinate is out of bounds");
          block_id += block_idx * mult;
          mult *= partition_dims[axis];
        }
        assert(block_id < block_count);
        uint64_t pos;
        pos = global_nnz_histogram[block_id];
        --global_nnz_histogram[block_id];

        (*dest)->blocks[block_id]->InsertNonzero(pos, local_tensor_coord.data(), val);
        NormX += val * val;
      }

      // Compute norm
      (*dest)->norm = std::sqrt(NormX);

      // Assign indices and print result
      (*dest)->AssignIndicesOfEachBlock();

      // Destroy locks and free memory
      global_nnz_histogram.clear();
      std::vector<uint64_t>().swap(global_nnz_histogram);
      printf("... 5) Done\n");
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_ARGS>::WriteDeltaToFile(TensorType *tensor, BlockType *block, ValueType *delta_block, int rank)
    {
      const std::string delta_path = config->getFilePath("delta");

      uint64_t block_id = block->get_block_id();
      uint64_t nnz_count = block->nnz_count;

      std::ofstream delta_file(delta_path, std::ios::binary);

      const uint64_t fixed_delta_size = sizeof(value_t) * tensor->get_max_nnz_count_in_block() * rank;
      const uint64_t delta_offset = fixed_delta_size * block_id;
      delta_file.seekp(delta_offset, std::ios::beg);

      uint64_t curr_delta_size = sizeof(value_t) * nnz_count * rank;
      delta_file.write(reinterpret_cast<const char *>(delta_block), curr_delta_size);

      uint64_t null_size = fixed_delta_size - curr_delta_size;
      if (null_size > 0)
      {
        char *null_ary = new char[null_size]();
        delta_file.write(null_ary, null_size);
        delete[] null_ary;
      }
      delta_file.close();
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_ARGS>::ReadDeltaFromFile(ValueType *delta_block, TensorType *tensor, BlockType *block, int rank)
    {
      const std::string delta_path = config->getFilePath("delta");

      uint64_t block_id = block->get_block_id();
      uint64_t nnz_count = block->nnz_count;

      std::ifstream delta_file(delta_path, std::ios::binary);

      const uint64_t fixed_delta_size = sizeof(value_t) * tensor->get_max_nnz_count_in_block() * rank;
      const uint64_t delta_offset = fixed_delta_size * block_id;

      delta_file.seekg(delta_offset, std::ios::beg);
      delta_file.read(reinterpret_cast<char *>(delta_block), fixed_delta_size);

      delta_file.close();
    }

  } // namespace cputucker
} // namespace supertensor