
#include <iostream>

#include "common/human_readable.hpp"
#include "common/size.hpp"
#include "cputucker/cmdline_opts.hpp"
#include "cputucker/helper.hpp"
#include "cputucker/optimizer.hpp"
#include "cputucker/scheduler.hpp"
#include "cputucker/tensor.hpp"
#include "cputucker/tensor_manager.hpp"
#include "cputucker/tucker.hpp"
#include "cputucker/config.hpp"

int main(int argc, char *argv[])
{
  using namespace supertensor::cputucker;

  CommandLineOptions *options = new CommandLineOptions;
  CommandLineOptions::ReturnStatus ret = options->Parse(argc, argv);
  // TODO rank and gpu options are not being parsed correctly
  if (CommandLineOptions::OPTS_SUCCESS == ret)
  {
    // Input file
    std::cout << options->get_input_path() << std::endl;

    using index_t = uint32_t;
    using value_t = double;
    using block_t = Block<index_t, value_t>;
    using tensor_t = Tensor<block_t>;
    using tensor_manager_t = TensorManager<tensor_t, block_t, value_t>;
    using optimizer_t = Optimizer<tensor_t>;
    using scheduler_t = Scheduler<tensor_t>;

    bool is_double = std::is_same<value_t, double>::value;
    if (is_double)
    {
      printf("Values are double type.\n");
    }
    else
    {
      printf("Values are float type.\n");
    }

    // Read tensor from file
    tensor_manager_t *tensor_manager = new tensor_manager_t(new Config("config.properties"));
    tensor_t *input_tensor = new tensor_t(options->get_order());
    tensor_manager->ParseFromFile(options->get_input_path(), &input_tensor);
    // input_tensor->ToString();

    //   size_t avail_mem_size = common::GiB<size_t>(4);  // 4GiB
    size_t avail_mem_size = common::MiB<size_t>(100); // 100Mib
    // Find optimal partition parameters from optimizer
    optimizer_t *optimizer = new optimizer_t;
    optimizer->Initialize(options->get_rank(), avail_mem_size, input_tensor);
    index_t *partition_dims = optimizer->FindPartitionParms();
    optimizer->ToString();

    std::cout << optimizer->mem_size << " " << optimizer->block_count;

    // Create tensor blocks ( = sub-tensors )
    tensor_t *tensor_blocks = new tensor_t(input_tensor);
    tensor_manager->CreateTensorBlocks<optimizer_t>(&input_tensor, &tensor_blocks, optimizer);
    tensor_blocks->ToString();
    tensor_manager->WriteBlockToFile(tensor_blocks);

    // for (uint64_t bid = 0; bid < tensor_blocks->block_count; ++bid)
    // {
    //   tensor_blocks->blocks[bid]->ToString();
    // }

    // TODO block scheduling
    MYPRINT("\t... Initialize Scheduler\n");
    scheduler_t *scheduler = new scheduler_t;
    scheduler->Initialize();
    scheduler->Schedule(tensor_blocks);

    delete input_tensor;
    delete tensor_blocks;

    tensor_t *tensor = (tensor_t *)tensor_manager->ReadTensorFromFile();
    TuckerDecomposition<tensor_t, optimizer_t, scheduler_t, tensor_manager_t>(tensor, optimizer, scheduler, tensor_manager);
  }
  else
  {
    std::cout << "ERROR - problem with options." << std::endl;
  }

  return 0;
}
