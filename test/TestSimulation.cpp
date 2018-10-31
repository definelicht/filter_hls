/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "Filter.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

int main() {

  std::vector<MemoryPack_t> data;
  std::vector<MemoryPack_t> output(8, MemoryPack_t(static_cast<Data_t>(0)));

  const Data_t data_raw[] = {0, 1, 2, 0, 0, 0, 3, 4,
                             5, 0, 0, 6, 7, 8, 0, 0,
                             9, 10, 11, 12, 13, 14, 15, 16,
                             17, 18, 19, 20, 21, 22, 23, 24,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             25, 26, 0, 0, 0, 0, 0, 27,
                             28, 29, 30, 31, 32, 33, 34, 35,
                             0, 0, 36, 37, 38, 39, 40, 0};
  for (int i = 0; i < 8; ++i) {
    data.emplace_back(data_raw + i * 8);
  }

  std::cout << "Running hardware emulation..." << std::flush;
  FilterKernel(data.data(), output.data(), 0.5);
  std::cout << " Done.\n";

  std::cout << "Verifying results..." << std::endl;
  for (int i = 0; i < 64; ++i) {
    std::cout << output[i / kMemoryWidth][i % kMemoryWidth] << " "; 
  }
  std::cout << "\n";
  // std::cout << "Successfully verified.\n";

  return 0;
}
