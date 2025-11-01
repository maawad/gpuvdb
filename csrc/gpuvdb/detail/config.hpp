/*
 * GPU-VDB: GPU-native sparse voxel database
 * Copyright (c) 2025 Muhammad Awad
 * Licensed under the MIT License
 */

#pragma once
#include <cstdint>
#include <limits>
#include <gpuvdb/detail/macros.hpp>

namespace gpuvdb {

// VDB tree configuration following standard VDB layout
// Root: 5^3 (LOG2DIM=5), Internal: 4^3 (LOG2DIM=4), Leaf: 3^3 (LOG2DIM=3)
template <uint32_t RootLog2Dim = 5, uint32_t InternalLog2Dim = 4, uint32_t LeafLog2Dim = 3>
struct vdb_config {
  static constexpr uint32_t root_log2dim     = RootLog2Dim;
  static constexpr uint32_t internal_log2dim = InternalLog2Dim;
  static constexpr uint32_t leaf_log2dim     = LeafLog2Dim;

  static constexpr uint32_t root_dim     = 1u << root_log2dim;
  static constexpr uint32_t internal_dim = 1u << internal_log2dim;
  static constexpr uint32_t leaf_dim     = 1u << leaf_log2dim;

  static constexpr uint32_t root_size     = root_dim * root_dim * root_dim;
  static constexpr uint32_t internal_size = internal_dim * internal_dim * internal_dim;
  static constexpr uint32_t leaf_size     = leaf_dim * leaf_dim * leaf_dim;

  // Bit masks for coordinate decomposition
  static constexpr uint32_t leaf_mask     = (1u << leaf_log2dim) - 1u;
  static constexpr uint32_t internal_mask = (1u << internal_log2dim) - 1u;
  static constexpr uint32_t root_mask     = (1u << root_log2dim) - 1u;

  // Bit offsets for each level
  static constexpr uint32_t leaf_offset     = 0;
  static constexpr uint32_t internal_offset = leaf_log2dim;
  static constexpr uint32_t root_offset     = leaf_log2dim + internal_log2dim;

  using coord_type = int32_t;
  using index_type = uint32_t;

  // Invalid/sentinel values
  static constexpr index_type invalid_index = std::numeric_limits<index_type>::max();
};

// Coordinate type for 3D indexing
template <typename T>
struct coord3 {
  using value_type = T;
  T x, y, z;

  GPUVDB_HOST_DEVICE coord3() : x(0), y(0), z(0) {}
  GPUVDB_HOST_DEVICE coord3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

  GPUVDB_HOST_DEVICE bool operator==(const coord3& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  GPUVDB_HOST_DEVICE bool operator!=(const coord3& other) const {
    return !(*this == other);
  }
};

}  // namespace gpuvdb

