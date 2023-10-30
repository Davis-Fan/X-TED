#ifndef CUDAPLAYGROUND_INCLUDE_UTILS_CUDA_UTILS_H_
#define CUDAPLAYGROUND_INCLUDE_UTILS_CUDA_UTILS_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <cub/util_ptx.cuh>
#include <iostream>
#include <vector>
#ifdef __CUDACC__
#define DEV_HOST __device__ __host__
#define DEV_HOST_INLINE __device__ __host__ __forceinline__
#define DEV_INLINE __device__ __forceinline__
#else
#define DEV_HOST_INLINE inline
#define DEV_HOST
#endif

#define MAX_BLOCK_SIZE 256
#define MAX_GRID_SIZE 768

#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)
#define TID_2D (threadIdx.y + blockIdx.y * blockDim.y)
#define TOTAL_THREADS_2D (gridDim.y * blockDim.y)
#define NUMBLOCKS (gridDim.x*gridDim.y)

static void HandleCudaError(const char* file, int line, cudaError_t err) {
  std::cout << "ERROR in " << file << ":" << line << ": "
            << cudaGetErrorString(err) << " (" << err << ")" << std::endl;
}

// CUDA assertions

#define CHECK_CUDA(err)                            \
  do {                                             \
    cudaError_t errr = (err);                      \
    if (errr != cudaSuccess) {                     \
      ::HandleCudaError(__FILE__, __LINE__, errr); \
    }                                              \
  } while (0)

template <typename SIZE_T>
inline void KernelSizing(int& block_num, int& block_size, SIZE_T work_size) {
  block_size = MAX_BLOCK_SIZE;
  block_num = std::min(MAX_GRID_SIZE, (int) round_up(work_size, block_size));
}

#endif  // CUDAPLAYGROUND_INCLUDE_UTILS_CUDA_UTILS_H_
