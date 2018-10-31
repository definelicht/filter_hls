/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      October 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "Filter.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>

int main(int, char **argv) {

  const float ratio = std::stof(argv[1]); 

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

  std::cout << "Running simulation...\n" << std::flush;
  FilterKernel(input.data(), output.data(), ratio);

  std::cout << "Running reference implementation...\n" << std::flush;
  ReferenceImplementation(reference_input.data(), reference_output.data(),
                          ratio);

  std::cout << "Verifying results..." << std::endl;
  for (int i = 0; i < kN / kMemoryWidth; ++i) {
    const auto pack = output[i];
    for (int j = 0; j < kMemoryWidth; ++j) {
      if (pack[j] != reference_output[i * kMemoryWidth + j]) {
        std::cerr << "Verification failed.\n" << std::flush;
        return 1;
      }
    }
  }
  std::cout << "Successfully verified.\n";

  return 0;
}
