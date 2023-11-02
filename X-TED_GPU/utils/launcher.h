#ifndef CUDAPLAYGROUND_LAUNCHER_H
#define CUDAPLAYGROUND_LAUNCHER_H
#include <cub/grid/grid_barrier.cuh>

#include "utils/stream.h"

template <typename F, typename... Args>
__global__ void KernelWrapper(F f, Args... args) {
  f(args...);
}

template <typename F, typename... Args>
__global__ void KernelWrapperFused(cub::GridBarrier barrier, F f,
                                   Args... args) {
  bool running;
  do {
    running = f(barrier, args...);
    barrier.Sync();
  } while (running);
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, F f, Args&&... args) {
  int grid_size, block_size;

  CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                KernelWrapper<F, Args...>, 0,
                                                (int) MAX_BLOCK_SIZE));

  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, size_t size, F f, Args&&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchFusedKernel(F f, Args&&... args) {
  int dev;
  int work_residency;
  cudaDeviceProp props{};
  int block_size = MAX_GRID_SIZE;

  CHECK_CUDA(cudaGetDevice(&dev));
  CHECK_CUDA(cudaGetDeviceProperties(&props, dev));

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &work_residency, KernelWrapperFused<F, Args...>, block_size, 0);

  size_t grid_size = props.multiProcessorCount * work_residency;
  dim3 grid_dims(grid_size), block_dims(block_size);
  cub::GridBarrierLifetime barrier;

  barrier.Setup(grid_size);
  KernelWrapperFused<<<grid_dims, block_dims>>>(barrier, f,
                                                std::forward<Args>(args)...);
  CHECK_CUDA(cudaDeviceSynchronize());
}
#endif  // CUDAPLAYGROUND_LAUNCHER_H
