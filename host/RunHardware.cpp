/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      October 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include "hlslib/SDAccel.h"
#include "Filter.h"
#include "aligned_allocator.h"
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char **argv) {

  if (argc < 4) {
    std::cout << "Required arguments: <N> <ratio> <mode [hw_emu, hw]>\n"
              << std::flush;
    return 1;
  }

  const unsigned N = std::stoi(argv[1]);
  const float ratio = std::stof(argv[2]); 
  const std::string mode(argv[3]);

  bool emulation;
  if (mode == "hw") {
    emulation = false;
  } else if (mode == "hw_emu") {
    emulation = true;
  } else {
    std::cerr << "Invalid mode: " << mode << "\n" << std::flush;
    return 2;
  }

  std::default_random_engine rng(5);
  std::uniform_real_distribution<Data_t> dist(0, 1);

  std::cout << "Initializing memory...\n" << std::flush;
  std::vector<Data_t> reference_input(N);
  std::vector<MemoryPack_t, aligned_allocator<Data_t, 4096>> input(
      N / kMemoryWidth);

  for (unsigned i = 0; i < N / kMemoryWidth; ++i) {
    for (unsigned j = 0; j < kMemoryWidth; ++j) {
      reference_input[i * kMemoryWidth + j] = dist(rng);
    }
    input[i].Pack(&reference_input[i * kMemoryWidth]);
  }

  std::vector<Data_t> reference_output(N, 0);
  std::vector<MemoryPack_t, aligned_allocator<Data_t, 4096>> output(
      N / kMemoryWidth, MemoryPack_t(static_cast<Data_t>(0)));
  std::vector<unsigned, aligned_allocator<Data_t, 4096>> N_out_vec(1, 0);
  unsigned N_out_reference = 0;

  std::cout << "Creating context...\n" << std::flush;
  hlslib::ocl::Context context;

  std::cout << "Creating host memory...\n" << std::flush;
  auto input_device =
      context.MakeBuffer<MemoryPack_t, hlslib::ocl::Access::read>(
          hlslib::ocl::MemoryBank::bank0, input.cbegin(), input.cend());
  auto output_device =
      context.MakeBuffer<MemoryPack_t, hlslib::ocl::Access::write>(
          hlslib::ocl::MemoryBank::bank1, output.cbegin(), output.cend());
  auto N_out_device = context.MakeBuffer<unsigned, hlslib::ocl::Access::write>(
      hlslib::ocl::MemoryBank::bank0, N_out_vec.cbegin(),
      N_out_vec.cbegin() + 1);

  std::cout << "Creating program...\n" << std::flush;
  auto program = context.MakeProgram(emulation ? "Filter_hw_emu.xclbin"
                                               : "Filter_hw.xclbin");

  std::cout << "Creating kernel...\n" << std::flush;
  auto kernel = program.MakeKernel("FilterKernel", input_device, output_device,
                                   N, ratio, N_out_device);

  std::cout << "Executing kernel...\n" << std::flush;
  const auto elapsed = kernel.ExecuteTask();
  std::cout << "Finished in " << elapsed.second
            << " seconds, corresponding to a bandwidth of "
            << 1e-6 * (1.0 + ratio) * N * sizeof(Data_t) / elapsed.second
            << " MB/s\n";

  std::cout << "Running reference implementation...\n" << std::flush;
  ReferenceImplementation(reference_input.data(), reference_output.data(),
                          N, ratio, N_out_reference);

  std::cout << "Copying back result..." << std::endl;
  output_device.CopyToHost(output.begin());
  N_out_device.CopyToHost(N_out_vec.begin());
  unsigned N_out = N_out_vec[0];

  bool failed = false;
  if (N_out != N_out_reference) {
    std::cerr << "Mismatch in number of filtered elements: " << N_out << " vs. "
              << N_out_reference << "\n";
    failed = true;
  }

  if (!failed) {
    unsigned wrong_results = 0;
    std::cout << "Verifying results..." << std::endl;
    for (int i = 0; i < N_out; ++i) {
      const auto kernel_res = output[i / kMemoryWidth][i % kMemoryWidth];
      const auto reference_res = reference_output[i];
      if (kernel_res != reference_res) {
        std::cout << "Wrong result at " << i << ": " << kernel_res << " / "
                  << reference_res << "\n";
        ++wrong_results;
      }
    }
    if (wrong_results > 0) {
      failed = true;
      std::cerr << "Verification failed: " << wrong_results << " / " << N_out
                << " wrong elements.\n"
                << std::flush;
    }
  }
  constexpr unsigned NumPrint = 32;
  if (failed) {
    std::cout << "** Printing first " << NumPrint << " elements:\nKernel:\n";
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
    for (unsigned i = N_out - NumPrint; i < N_out; ++i) {
      std::cout << output[i / kMemoryWidth][i % kMemoryWidth] << " ";
    }
    std::cout << "\nReference:\n";
    for (unsigned i = N_out - NumPrint; i < N_out; ++i) {
      std::cout << reference_output[i] << " ";
    }
    std::cout << "\n\n";
    return 3;
  }

  std::cout << "Successfully verified.\n";

  return 0;
}
