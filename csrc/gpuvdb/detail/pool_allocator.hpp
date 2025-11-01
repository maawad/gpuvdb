/*
 * GPU-VDB: GPU-native sparse voxel database
 * Copyright (c) 2025 Muhammad Awad
 * Licensed under the MIT License
 */

#pragma once
#include <gpuvdb/detail/macros.hpp>
#include <cstdint>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace gpuvdb {
namespace detail {

// Simple pool allocator for fixed-size nodes
// Uses atomic counter for allocation
template <typename NodeType>
struct pool_allocator {
  using node_type  = NodeType;
  using index_type = uint32_t;

  node_type* pool_;
  uint32_t* counter_;
  uint32_t capacity_;

  pool_allocator() : pool_(nullptr), counter_(nullptr), capacity_(0) {}

  // Initialize allocator with capacity
  void init(uint32_t capacity) {
    capacity_ = capacity;

#ifdef __HIP_PLATFORM_AMD__
    hipMalloc(&pool_, sizeof(node_type) * capacity_);
    hipMalloc(&counter_, sizeof(uint32_t));
    uint32_t init_val = 0;
    hipMemcpy(counter_, &init_val, sizeof(uint32_t), hipMemcpyHostToDevice);
#else
    cudaMalloc(&pool_, sizeof(node_type) * capacity_);
    cudaMalloc(&counter_, sizeof(uint32_t));
    uint32_t init_val = 0;
    cudaMemcpy(counter_, &init_val, sizeof(uint32_t), cudaMemcpyHostToDevice);
#endif
  }

  // Clean up
  void free() {
    if (pool_) {
#ifdef __HIP_PLATFORM_AMD__
      hipFree(pool_);
      hipFree(counter_);
#else
      cudaFree(pool_);
      cudaFree(counter_);
#endif
      pool_ = nullptr;
      counter_ = nullptr;
    }
  }

  // Allocate a new node (returns index)
  GPUVDB_DEVICE index_type allocate() {
    index_type idx = atomicAdd(counter_, 1);
    if (idx >= capacity_) {
      // Out of memory - return invalid index
      return 0xFFFFFFFF;
    }
    // Initialize node
    new (&pool_[idx]) node_type();
    return idx;
  }

  // Get pointer to node by index
  GPUVDB_DEVICE node_type* get_node(index_type idx) {
    return &pool_[idx];
  }

  GPUVDB_DEVICE const node_type* get_node(index_type idx) const {
    return &pool_[idx];
  }

  // Get current allocation count
  uint32_t get_count() const {
    uint32_t count;
#ifdef __HIP_PLATFORM_AMD__
    hipMemcpy(&count, counter_, sizeof(uint32_t), hipMemcpyDeviceToHost);
#else
    cudaMemcpy(&count, counter_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
#endif
    return count;
  }

  // Get raw device pool pointer (for host-side access)
  node_type* get_pool() { return pool_; }
  const node_type* get_pool() const { return pool_; }
};

}  // namespace detail
}  // namespace gpuvdb

