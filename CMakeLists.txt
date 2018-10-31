# Author:  Johannes de Fine Licht (definelicht@inf.ethz.ch)
# Created: June 2017

cmake_minimum_required(VERSION 3.0)
project(sdaccel_filter)

# Target options 
set(FILTER_PART_NAME "xcvu9p-flgb2104-2-i" CACHE STRING "Part name for HLS.")
set(FILTER_DSA_NAME "xilinx_vcu1525_dynamic_5_1" CACHE STRING
    "DSA string for xocc.")
set(FILTER_TARGET_CLOCK 200 CACHE STRING
    "Target clock for kernel (<=0 uses DSA default).")
set(FILTER_TARGET_CLOCK_UNCERTAINTY 1.08 CACHE STRING "Clock uncertainty for HLS.")
set(FILTER_TMP_DIR /tmp CACHE STRING "Temporary directory for SDx build files")
set(FILTER_ENABLE_PROFILING OFF CACHE STRING "Collect profiling information")
set(FILTER_ENABLE_DEBUGGING OFF CACHE STRING "Inject debugging cores to design")

# Domain options
set(FILTER_DATA_TYPE "float" CACHE STRING "Data type.")
set(FILTER_MEMORY_BUS_WIDTH 64 CACHE STRING "Width of memory bus in bytes.")
set(FILTER_NUM_ELEMENTS 512 CACHE STRING "Number of elements.")

# Internal
set(FILTER_KERNEL_NAME FilterKernel)
set(FILTER_ENTRY_FUNCTION Filter)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake
    ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

find_package(SDAccel REQUIRED)
find_package(Threads REQUIRED)

include_directories(include hlslib/include ${CMAKE_BINARY_DIR} ${SDAccel_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# Allow specifying the floating point unit for adders and multipliers
if(FILTER_ADD_RESOURCE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHLSLIB_OPERATOR_ADD_RESOURCE=${FILTER_ADD_RESOURCE}") 
endif() 
if(FILTER_MULTIPLY_RESOURCE) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHLSLIB_OPERATOR_MULTIPLY_RESOURCE=${FILTER_MULTIPLY_RESOURCE}") 
endif() 

# Hardware configuration header for HLS
set(FILTER_GOLDEN_DIR ${CMAKE_SOURCE_DIR}/golden/)
configure_file(include/Config.h.in Config.h)

# C++ source code 
set(FILTER_KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/kernel/Filter.cpp)
add_library(filterkernel ${FILTER_KERNEL_SRC})
target_link_libraries(filterkernel ${CMAKE_THREAD_LIBS_INIT})

# Executables 
# add_executable(Stats src/Stats.cpp)

# Software test
add_executable(TestSimulation test/TestSimulation.cpp)
target_link_libraries(TestSimulation ${SDAccel_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} filterkernel)
add_test(TestSimulation TestSimulation)
enable_testing()

# Synthesis
set(FILTER_HLS_PROJECT_NAME "hls")
string(REPLACE ";" " " FILTER_SRC_SYNTHESIS "${FILTER_KERNEL_SRC}")
set(FILTER_ENTRY_FUNCTION "FilterKernel")
set(FILTER_SYNTHESIS_FLAGS "-std=c++11 -O3 -DFILTER_SYNTHESIS -DHLSLIB_SYNTHESIS")
configure_file(scripts/Synthesis.tcl.in Synthesis.tcl)
add_custom_target(synthesis COMMAND ${SDAccel_VIVADO_HLS} -f Synthesis.tcl) 

# Hardware kernel
set(FILTER_XOCC_COMMAND
  -s
  -O3
  ${FILTER_XOCC_FLAGS}
  -I${CMAKE_SOURCE_DIR}/include
  -I${CMAKE_SOURCE_DIR}/hlslib/include
  -I${CMAKE_BINARY_DIR}
  --kernel ${FILTER_KERNEL_NAME} 
  --platform ${FILTER_DSA_NAME}
  --xp prop:kernel.${FILTER_ENTRY_FUNCTION}.kernel_flags="${FILTER_SYNTHESIS_FLAGS}"
  --max_memory_ports all)
if(FILTER_ENABLE_PROFILING)
  set(FILTER_XOCC_COMMAND ${FILTER_XOCC_COMMAND}
    --profile_kernel "data:all:all:all"
    --profile_kernel "stall:all:all"
    --profile_kernel "exec:all:all")
endif()
if(FILTER_ENABLE_DEBUGGING)
  set(FILTER_XOCC_COMMAND ${FILTER_XOCC_COMMAND}
    --dk chipscope:FilterKernel_1:M_AXI_GMEM0
    --dk chipscope:FilterKernel_1:M_AXI_GMEM1)
endif()
if(FILTER_TARGET_CLOCK GREATER 0)
  set(FILTER_XOCC_COMMAND ${FILTER_XOCC_COMMAND}
      --kernel_frequency ${FILTER_TARGET_CLOCK})
endif()
set(FILTER_XOCC_COMMAND ${FILTER_XOCC_COMMAND}
  --sp ${FILTER_KERNEL_NAME}_1.m_axi_gmem0:bank0
  --sp ${FILTER_KERNEL_NAME}_1.m_axi_gmem1:bank1)

add_custom_target(compile_hardware_emulation
  COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -c -t hw_emu
  ${FILTER_XOCC_COMMAND} ${FILTER_XOCC_LINK_FLAGS}
  ${FILTER_KERNEL_SRC} -o FilterKernel_hw_emu.xo) 
add_custom_target(build_hardware_emulation
  COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -l -t hw_emu
  ${FILTER_XOCC_COMMAND} ${FILTER_XOCC_LINK_FLAGS}
  FilterKernel_hw_emu.xo -o Filter_hw_emu.xclbin) 
add_custom_target(compile_hardware
  COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -c -t hw
  ${FILTER_XOCC_COMMAND} ${FILTER_XOCC_LINK_FLAGS}
  ${FILTER_KERNEL_SRC} -o FilterKernel_hw.xo) 
add_custom_target(build_hardware
  COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -l -t hw
  ${FILTER_XOCC_COMMAND} ${FILTER_XOCC_LINK_FLAGS}
  FilterKernel_hw.xo -o Filter_hw.xclbin) 

# Host code to launch kernel
# add_executable(RunHardware.exe host/RunHardware.cpp)
# target_link_libraries(RunHardware.exe ${SDAccel_LIBRARIES} ${BLAS_LIBRARIES} filterkernel)
