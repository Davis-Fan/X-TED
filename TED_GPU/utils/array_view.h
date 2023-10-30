#ifndef CUDAPLAYGROUND_ARRAY_VIEW_H
#define CUDAPLAYGROUND_ARRAY_VIEW_H
#include <thrust/device_vector.h>
#include <thrust/swap.h>

#include "utils/cuda_utils.h"

template <typename T>
class ArrayView {
 public:
  ArrayView() = default;

  DEV_HOST ~ArrayView() {}

  explicit ArrayView(const thrust::device_vector<T>& vec)
      : data_(const_cast<T*>(thrust::raw_pointer_cast(vec.data()))),
        size_(vec.size()) {}

  DEV_HOST ArrayView(T* data, size_t size) : data_(data), size_(size) {}

  DEV_HOST_INLINE T* data() { return data_; }

  DEV_HOST_INLINE const T* data() const { return data_; }

  DEV_HOST_INLINE size_t size() const { return size_; }

  DEV_HOST_INLINE bool empty() const { return size_ == 0; }

  DEV_INLINE T& operator[](size_t i) { return data_[i]; }

  DEV_INLINE const T& operator[](size_t i) const { return data_[i]; }

  DEV_INLINE void Swap(ArrayView<T>& rhs) {
    thrust::swap(data_, rhs.data_);
    thrust::swap(size_, rhs.size_);
  }

 private:
  T* data_{};
  size_t size_{};
};
#endif  // CUDAPLAYGROUND_ARRAY_VIEW_H
