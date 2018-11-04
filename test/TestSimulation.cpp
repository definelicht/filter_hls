/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      October 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "Filter.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>

int main(int argc, char **argv) {

  if (argc < 3) {
    std::cout << "Required arguments: N ratio\n" << std::flush;
    return 1;
  }

  const unsigned N = std::stoi(argv[1]);
  const float ratio = std::stof(argv[2]); 

  std::default_random_engine rng(5);
  std::uniform_real_distribution<Data_t> dist(0, 1);

  std::cout << "Initializing memory...\n" << std::flush;
  std::vector<Data_t> reference_input(N);
  std::vector<MemoryPack_t> input(N / kMemoryWidth);

  for (unsigned i = 0; i < N / kMemoryWidth; ++i) {
    for (unsigned j = 0; j < kMemoryWidth; ++j) {
      reference_input[i * kMemoryWidth + j] = dist(rng);
    }
    input[i].Pack(&reference_input[i * kMemoryWidth]);
  }

  std::vector<Data_t> reference_output(N, 0);
  std::vector<MemoryPack_t> output(N / kMemoryWidth,
                                   MemoryPack_t(static_cast<Data_t>(0)));
  unsigned N_out = 0;
  unsigned N_out_reference = 0;

  std::cout << "Running simulation...\n" << std::flush;
  FilterKernel(input.data(), output.data(), N, ratio, &N_out);

  std::cout << "Running reference implementation...\n" << std::flush;
  ReferenceImplementation(reference_input.data(), reference_output.data(),
                          N, ratio, N_out_reference);

  bool failed = false;
  if (N_out != N_out_reference) {
    std::cerr << "Mismatch in number of filtered elements: " << N_out << " vs. "
              << N_out_reference << "\n";
    failed = true;
  }

  if (!failed) {
    std::cout << "Verifying results..." << std::endl;
    for (unsigned i = 0; i < N / kMemoryWidth; ++i) {
      const auto pack = output[i];
      for (unsigned j = 0; j < kMemoryWidth; ++j) {
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
  }
  constexpr int NumPrint = 32;
  if (failed) {
    std::cout << "\n** Printing first " << NumPrint << " elements:\nKernel:\n";
    for (unsigned i = 0; i < NumPrint / kMemoryWidth; ++i) {
      const auto pack = output[i];
      for (unsigned j = 0; j < kMemoryWidth; ++j) {
        std::cout << pack[j] << " ";
      }
    }
    std::cout << "\nReference:\n";
    for (unsigned i = 0; i < NumPrint; ++i) {
      std::cout << reference_output[i] << " ";
    }
    std::cout << "\n\n** Printing last " << NumPrint
              << " elements:\nKernel:\n";
    for (unsigned i = (N - NumPrint) / kMemoryWidth; i < N / kMemoryWidth; ++i) {
      const auto pack = output[i];
      for (unsigned j = 0; j < kMemoryWidth; ++j) {
        std::cout << pack[j] << " ";
      }
    }
    std::cout << "\nReference:\n";
    for (unsigned i = N - NumPrint; i < N; ++i) {
      std::cout << reference_output[i] << " ";
    }
    std::cout << "\n\n";
    return 1;
  }

  std::cout << "Successfully verified.\n";

  return 0;
}
