/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      October 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include "Config.h"
#include "hlslib/DataPack.h"
#include <type_traits>

constexpr int kMemoryWidth = kMemoryWidthBytes / sizeof(Data_t);
static_assert(kMemoryWidthBytes % sizeof(Data_t) == 0,
              "Memory width not divisable by size of data type.");

using MemoryPack_t = hlslib::DataPack<Data_t, kMemoryWidth>;

extern "C" void FilterKernel(MemoryPack_t const in[], MemoryPack_t out[],
                             Data_t ratio);

#ifndef HLSLIB_SYNTHESIS

inline void ReferenceImplementation(Data_t const in[], Data_t out[],
                                    Data_t const ratio) {
  int i_out = 0;
  for (int i_in = 0; i_in < kN; ++i_in) {
    const auto read = in[i_in];
    if (read >= ratio) {
      out[i_out++] = read;
    }
  }
}

#endif
