/*
 * GPU-VDB: GPU-native sparse voxel database
 * Copyright (c) 2025 Muhammad Awad
 * Licensed under the MIT License
 */

#pragma once
#include <gpuvdb/detail/config.hpp>
#include <gpuvdb/detail/macros.hpp>
#include <gpuvdb/detail/vdb_tree.hpp>

namespace gpuvdb {
namespace kernels {

// Kernel to set voxel values
template <typename TreeType, typename CoordType, typename ValueType>
__global__ void set_values_kernel(TreeType tree,
                                  const CoordType* coords,
                                  const ValueType* values,
                                  uint32_t num_voxels,
                                  uint32_t* success) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_voxels) {
    bool result = tree.set_value(coords[tid], values[tid]);
    if (!result && success) {
      atomicAnd(success, 0u);  // Mark failure
    }
  }
}

// Kernel to get voxel values
template <typename TreeType, typename CoordType, typename ValueType>
__global__ void get_values_kernel(TreeType tree,
                                  const CoordType* coords,
                                  ValueType* values,
                                  uint32_t num_voxels) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_voxels) {
    values[tid] = tree.get_value(coords[tid]);
  }
}

// Kernel to check if voxels are active
template <typename TreeType, typename CoordType>
__global__ void is_active_kernel(TreeType tree,
                                const CoordType* coords,
                                bool* active,
                                uint32_t num_voxels) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_voxels) {
    active[tid] = tree.is_active(coords[tid]);
  }
}

// Kernel to fill a sphere with voxels
template <typename TreeType, typename ValueType>
__global__ void fill_sphere_kernel(TreeType tree,
                                   int32_t center_x, int32_t center_y, int32_t center_z,
                                   float radius,
                                   ValueType value,
                                   int32_t start_x, int32_t start_y, int32_t start_z) {
  int32_t x = start_x + blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = start_y + blockIdx.y * blockDim.y + threadIdx.y;
  int32_t z = start_z + blockIdx.z * blockDim.z + threadIdx.z;

  // Check if within sphere
  float dx = static_cast<float>(x - center_x);
  float dy = static_cast<float>(y - center_y);
  float dz = static_cast<float>(z - center_z);
  float dist_sq = dx * dx + dy * dy + dz * dz;

  if (dist_sq <= radius * radius) {
    typename TreeType::coord3_type coord(x, y, z);
    tree.set_value(coord, value);
  }
}

}  // namespace kernels
}  // namespace gpuvdb

