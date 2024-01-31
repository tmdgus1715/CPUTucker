#ifndef TENSOR_MANAGER_HPP_
#define TENSOR_MANAGER_HPP_

#include "cputucker/config.hpp"

namespace supertensor
{
  namespace cputucker
  {

#define TENSOR_MANAGER_TEMPLATE template <typename TensorType, typename BlockType, typename ValueType>
#define TENSOR_MANAGER_ARGS TensorType, BlockType, ValueType

    TENSOR_MANAGER_TEMPLATE
    class TensorManager
    {
      using tensor_t = TensorType;
      using index_t = typename tensor_t::index_t;
      using value_t = typename tensor_t::value_t;
      using block_t = typename tensor_t::block_t;

    public:
      TensorManager();
      TensorManager(Config *config);
      ~TensorManager();

      bool ParseFromFile(const std::string &file_name, tensor_t **tensor);
      template <typename OptimizerType>
      void CreateTensorBlocks(tensor_t **src, tensor_t **dest, OptimizerType *optimizer);

      //---write---
      void WriteBlockToFile(tensor_t *tensor);
      void WriteDeltaToFile(TensorType *tensor, BlockType *block, ValueType *delta_block, int rank);

      //---read---
      void *ReadTensorFromFile();
      void *ReadBlockFromFile(uint64_t block_id);
      void ReadDeltaFromFile(ValueType *delta_block, TensorType *tensor, BlockType *block, int rank);

    private:
      void _WriteTensorMetadataToFile(std::ofstream &tensor_md_file, tensor_t *tensor);
      void _WriteBlockMetadataToFile(std::ofstream &block_md_file, block_t *block);
      void _WriteBlockDataToFile(std::ofstream &data_file, block_t *block);

      bool _ReadData(const char *buffer, const size_t buffer_length, tensor_t **tensor);

    public:
      uint64_t fixed_data_size;

    private:
      Config *config;
    }; // class TensorMANAGER
  }    // namespace cputucker
} // namespace supertensor
#include "cputucker/tensor_manager.tpp"
#endif // TENSOR_READER_HPP_