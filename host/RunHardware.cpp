/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      October 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include "hlslib/SDAccel.h"
#include "Filter.h"
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char **argv) {

  float ratio;
  if (argc > 1) { 
    ratio = std::stof(argv[1]); 
  } else {
    ratio = 0.5;
  }

  std::default_random_engine rng(5);
  std::uniform_real_distribution<Data_t> dist(0, 1);

  std::cout << "Initializing memory...\n" << std::flush;
  std::vector<Data_t> reference_input(kN);
  std::vector<MemoryPack_t> input(kN / kMemoryWidth);

  for (int i = 0; i < kN / kMemoryWidth; ++i) {
    for (int j = 0; j < kMemoryWidth; ++j) {
      reference_input[i * kMemoryWidth + j] = dist(rng);
    }
    input[i].Pack(&reference_input[i * kMemoryWidth]);
  }

  std::vector<Data_t> reference_output(kN, 0);
  std::vector<MemoryPack_t> output(kN / kMemoryWidth,
                                   MemoryPack_t(static_cast<Data_t>(0)));

  std::cout << "Creating context...\n" << std::flush;
  hlslib::ocl::Context context;

  std::cout << "Creating host memory...\n" << std::flush;
  auto input_device =
      context.MakeBuffer<MemoryPack_t, hlslib::ocl::Access::read>(
          hlslib::ocl::MemoryBank::bank0, input.cbegin(), input.cend());
  auto output_device =
      context.MakeBuffer<MemoryPack_t, hlslib::ocl::Access::write>(
          hlslib::ocl::MemoryBank::bank1, output.cbegin(), output.cend());

  std::cout << "Creating program...\n" << std::flush;
  auto program = context.MakeProgram("Filter_hw_emu.xclbin");

  std::cout << "Creating kernel...\n" << std::flush;
  auto kernel =
      program.MakeKernel("FilterKernel", input_device, output_device, ratio);

  std::cout << "Executing kernel...\n" << std::flush;
  const auto elapsed = kernel.ExecuteTask();
  std::cout << "Finished in " << elapsed.second << " seconds.\n";

  std::cout << "Running reference implementation...\n" << std::flush;
  ReferenceImplementation(reference_input.data(), reference_output.data(),
                          ratio);

  std::cout << "Verifying results..." << std::endl;
  bool failed = false;
  for (int i = 0; i < kN / kMemoryWidth; ++i) {
    const auto pack = output[i];
    for (int j = 0; j < kMemoryWidth; ++j) {
      if (pack[j] != reference_output[i * kMemoryWidth + j]) {
        std::cerr << "Verification failed.\n" << std::flush;
        failed = true;
        break;
      }
    }
    if (failed) {
      break;
    }
  }
  constexpr int kNumPrint = 32;
  if (failed) {
    std::cout << "** Printing first " << kNumPrint << " elements:\nKernel:\n";
    for (int i = 0; i < kNumPrint / kMemoryWidth; ++i) {
      const auto pack = output[i];
      for (int j = 0; j < kMemoryWidth; ++j) {
        std::cout << pack[j] << " ";
      }
    }
    std::cout << "\nReference:\n";
    for (int i = 0; i < kNumPrint; ++i) {
      std::cout << reference_output[i] << " ";
    }
    std::cout << "\n\n** Printing last " << kNumPrint
              << " elements:\nKernel:\n";
    for (int i = (kN - kNumPrint) / kMemoryWidth; i < kN / kMemoryWidth; ++i) {
      const auto pack = output[i];
      for (int j = 0; j < kMemoryWidth; ++j) {
        std::cout << pack[j] << " ";
      }
    }
    std::cout << "\nReference:\n";
    for (int i = kN - kNumPrint; i < kN; ++i) {
      std::cout << reference_output[i] << " ";
    }
    std::cout << "\n\n";
    return 1;
  }

  std::cout << "Successfully verified.\n";

  return 0;
}
