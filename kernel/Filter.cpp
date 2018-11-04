/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      October 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "hlslib/TreeReduce.h"
#include "hlslib/Utility.h"
#include "Filter.h"
#include <cassert>
#include <ap_int.h>

using hlslib::Stream;

void Read(MemoryPack_t const in[], Stream<MemoryPack_t> &pipe,
          const unsigned N) {
  for (unsigned i = 0; i < N / kMemoryWidth; ++i) {
    #pragma HLS PIPELINE II=1
    pipe.Push(in[i]);
  }
}

void Filter(const unsigned N, const Data_t ratio, Stream<MemoryPack_t> &pipe_in,
            Stream<MemoryPack_t> &pipe_out, Stream<bool> &valid_out) {

  constexpr int num_stages = 2 * kMemoryWidth - 1; 
  using Stage_t = ap_uint<hlslib::ConstLog2(num_stages)>;
  using Count_t = ap_uint<hlslib::ConstLog2(kMemoryWidth)>;

  MemoryPack_t stages[num_stages];
  #pragma HLS ARRAY_PARTITION variable=stages complete 

  Stage_t num_shifts[num_stages][kMemoryWidth];
  #pragma HLS ARRAY_PARTITION variable=num_shifts complete 

  bool non_zero[num_stages][kMemoryWidth];
  #pragma HLS ARRAY_PARTITION variable=non_zero complete 

  MemoryPack_t output, next;

  Count_t elements_in_output = 0;

  for (unsigned i = 0; i < N / kMemoryWidth + 1; ++i) {
    #pragma HLS PIPELINE II=1

    if (i < N / kMemoryWidth) {
      stages[0] = pipe_in.Pop();
    } else {
      stages[0] = MemoryPack_t(static_cast<Data_t>(0));
    }

    // Initialize shift counters
    Stage_t empty_slots_left = 0;
    const Count_t elements_in_output_local = elements_in_output;
    Count_t additional_elements = 0;
    for (unsigned w = 0; w < kMemoryWidth; ++w) {
      #pragma HLS UNROLL
      if (stages[0][w] < ratio) {
        ++empty_slots_left;
        non_zero[0][w] = false;
        num_shifts[0][w] = 0;
      } else {
        non_zero[0][w] = true;
        num_shifts[0][w] =
            kMemoryWidth - elements_in_output_local + empty_slots_left;
        ++additional_elements;
      }
    }
    elements_in_output =
        (elements_in_output_local + additional_elements) % kMemoryWidth;

    // Merge stages 
    for (int s = 1; s < num_stages; ++s) {
      #pragma HLS UNROLL
      for (unsigned w = 0; w < kMemoryWidth; ++w) {
        #pragma HLS UNROLL
        // If we're already in the correct place, just propagate forward
        if (num_shifts[s - 1][w] == 0 && non_zero[s - 1][w]) {
          stages[s][w] = stages[s - 1][w];
          num_shifts[s][w] = 0;
          non_zero[s][w] = true;
        } else {
          // Otherwise shift if the value is non-zero
          const Stage_t shifts = num_shifts[s - 1][(w + 1) % kMemoryWidth]; 
          if (shifts > 0) {
            stages[s][w] = stages[s - 1][(w + 1) % kMemoryWidth];
            num_shifts[s][w] = shifts - 1;
            non_zero[s][w] = true;
          } else {
            stages[s][w] = 0;
            num_shifts[s][w] = 0;
            non_zero[s][w] = false;
          }
        }
      }
    }

    // Fill up vector 
    Count_t num_curr = 0;
    Count_t num_next = 0;
    for (unsigned w = 0; w < kMemoryWidth; ++w) {
      #pragma HLS UNROLL
      const bool is_taken = w < elements_in_output_local;
      const bool is_non_zero = non_zero[num_stages - 1][w];
      if (!is_taken) {
        if (is_non_zero) {
          ++num_curr;
          output[w] = stages[num_stages - 1][w];  
        }
      } else {
        ++num_curr;
        if (is_non_zero) {
          next[w] = stages[num_stages - 1][w];  
          ++num_next;
        }
      }
    }

    const bool is_full = num_curr == kMemoryWidth;
    const bool last_iter = i == N / kMemoryWidth;
    pipe_out.Push(output);
    valid_out.Push(is_full || last_iter);
    if (is_full) {
      output = next;
      next = MemoryPack_t(Data_t(0));
    }

  } // End loop

}

void Write(Stream<MemoryPack_t> &pipe, Stream<bool> &pipe_valid,
           MemoryPack_t out[], const unsigned N) {
  unsigned i_out = 0;
  for (unsigned i = 0; i < N / kMemoryWidth + 1; ++i) {
    #pragma HLS PIPELINE II=1
    const auto rd = pipe.Pop();
    const auto valid = pipe_valid.Pop(); 
    if (valid && i_out < N / kMemoryWidth) {
      out[i_out++] = rd;
    }
  }
}

extern "C" void FilterKernel(MemoryPack_t const in[], MemoryPack_t out[],
                             unsigned N, Data_t ratio) {

  #pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1
  #pragma HLS INTERFACE s_axilite port=in bundle=control
  #pragma HLS INTERFACE s_axilite port=out bundle=control
  #pragma HLS INTERFACE s_axilite port=N bundle=control
  #pragma HLS INTERFACE s_axilite port=ratio bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  
  #pragma HLS DATAFLOW
  Stream<MemoryPack_t> pipe_in("pipe_in", 2048);
  Stream<MemoryPack_t> pipe_out("pipe_out", 2048);
  Stream<bool> pipe_valid("pipe_valid", 2048);

  HLSLIB_DATAFLOW_INIT();

  HLSLIB_DATAFLOW_FUNCTION(Read, in, pipe_in, N);
  HLSLIB_DATAFLOW_FUNCTION(Filter, N, ratio, pipe_in, pipe_out, pipe_valid);
  HLSLIB_DATAFLOW_FUNCTION(Write, pipe_out, pipe_valid, out, N);

  HLSLIB_DATAFLOW_FINALIZE();
}
