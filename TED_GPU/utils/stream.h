#ifndef CUDAPLAYGROUND_INCLUDE_STREAM_H_
#define CUDAPLAYGROUND_INCLUDE_STREAM_H_
#include "utils/cuda_utils.h"

enum class StreamPriority { kDefault, kHigh, kLow };

class Stream {
 public:
  explicit Stream(StreamPriority priority = StreamPriority::kDefault)
      : priority_(priority) {
    initStream();
  }

  Stream(const Stream& other) = delete;

  Stream(Stream&& other) noexcept
      : priority_(other.priority_), cuda_stream_(other.cuda_stream_) {
    other.cuda_stream_ = nullptr;
  }

  Stream& operator=(const Stream& other) = delete;

  Stream& operator=(Stream&& other) noexcept {
    if (this != &other) {
      this->cuda_stream_ = other.cuda_stream_;
      other.cuda_stream_ = nullptr;
    }
    return *this;
  }

  ~Stream() {
    if (cuda_stream_ != nullptr) {
      CHECK_CUDA(cudaStreamDestroy(cuda_stream_));
    }
  }

  void Sync() const { CHECK_CUDA(cudaStreamSynchronize(cuda_stream_)); }

  cudaStream_t cuda_stream() const { return cuda_stream_; }

 private:
  void initStream() {
    if (priority_ == StreamPriority::kDefault) {
      CHECK_CUDA(
          cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));
    } else {
      int leastPriority, greatestPriority;
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
      CHECK_CUDA(cudaStreamCreateWithPriority(
          &cuda_stream_, cudaStreamNonBlocking,
          priority_ == StreamPriority::kHigh ? greatestPriority
                                             : leastPriority));
    }
  }

  StreamPriority priority_;
  cudaStream_t cuda_stream_{};
};
#endif  // CUDAPLAYGROUND_INCLUDE_STREAM_H_
