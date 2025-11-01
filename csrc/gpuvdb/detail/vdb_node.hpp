/*
 * GPU-VDB: GPU-native sparse voxel database
 * Copyright (c) 2025 Muhammad Awad
 * Licensed under the MIT License
 */

#pragma once
#include <gpuvdb/detail/config.hpp>
#include <gpuvdb/detail/macros.hpp>

namespace gpuvdb {
namespace detail {

// Forward declarations
template <typename ValueType, typename Config>
struct leaf_node;

template <typename ValueType, typename Config>
struct internal_node;

template <typename ValueType, typename Config>
struct root_node;

// Leaf node: stores actual voxel data
// Memory layout: flat array of values + bitmask for active states
template <typename ValueType, typename Config>
struct leaf_node {
  using value_type  = ValueType;
  using config_type = Config;
  using index_type  = typename Config::index_type;

  static constexpr uint32_t log2dim = Config::leaf_log2dim;
  static constexpr uint32_t dim     = Config::leaf_dim;
  static constexpr uint32_t size    = Config::leaf_size;

  // Number of 32-bit words needed for bitmask
  static constexpr uint32_t num_bitmask_words = (size + 31) / 32;

  value_type values[size];
  uint32_t active_mask[num_bitmask_words];

  GPUVDB_HOST_DEVICE leaf_node() {
    for (uint32_t i = 0; i < size; ++i) {
      values[i] = value_type{};
    }
    for (uint32_t i = 0; i < num_bitmask_words; ++i) {
      active_mask[i] = 0;
    }
  }

  // Convert local coordinate to linear index
  GPUVDB_DEVICE static uint32_t coord_to_offset(uint32_t x, uint32_t y, uint32_t z) {
    return (x << (2 * log2dim)) | (y << log2dim) | z;
  }

  // Check if voxel is active
  GPUVDB_DEVICE bool is_active(uint32_t offset) const {
    uint32_t word_idx = offset >> 5;  // divide by 32
    uint32_t bit_idx  = offset & 31;  // modulo 32
    return (active_mask[word_idx] & (1u << bit_idx)) != 0;
  }

  // Set voxel as active
  GPUVDB_DEVICE void set_active(uint32_t offset, bool active = true) {
    uint32_t word_idx = offset >> 5;
    uint32_t bit_idx  = offset & 31;
    if (active) {
      atomicOr(&active_mask[word_idx], 1u << bit_idx);
    } else {
      atomicAnd(&active_mask[word_idx], ~(1u << bit_idx));
    }
  }

  // Get voxel value
  GPUVDB_DEVICE value_type get_value(uint32_t offset) const {
    return values[offset];
  }

  // Set voxel value
  GPUVDB_DEVICE void set_value(uint32_t offset, const value_type& value) {
    values[offset] = value;
  }
};

// Internal node: stores child pointers (to internal nodes or leaf nodes)
template <typename ValueType, typename Config>
struct internal_node {
  using value_type  = ValueType;
  using config_type = Config;
  using index_type  = typename Config::index_type;

  static constexpr uint32_t log2dim = Config::internal_log2dim;
  static constexpr uint32_t dim     = Config::internal_dim;
  static constexpr uint32_t size    = Config::internal_size;

  static constexpr uint32_t num_bitmask_words = (size + 31) / 32;

  // Child node indices
  index_type children[size];

  // Bitmask: 1 if child exists, 0 otherwise
  uint32_t child_mask[num_bitmask_words];

  GPUVDB_HOST_DEVICE internal_node() {
    for (uint32_t i = 0; i < size; ++i) {
      children[i] = Config::invalid_index;
    }
    for (uint32_t i = 0; i < num_bitmask_words; ++i) {
      child_mask[i] = 0;
    }
  }

  GPUVDB_DEVICE static uint32_t coord_to_offset(uint32_t x, uint32_t y, uint32_t z) {
    return (x << (2 * log2dim)) | (y << log2dim) | z;
  }

  GPUVDB_DEVICE bool has_child(uint32_t offset) const {
    uint32_t word_idx = offset >> 5;
    uint32_t bit_idx  = offset & 31;
    return (child_mask[word_idx] & (1u << bit_idx)) != 0;
  }

  GPUVDB_DEVICE void set_child(uint32_t offset, index_type child_index) {
    children[offset] = child_index;
    uint32_t word_idx = offset >> 5;
    uint32_t bit_idx  = offset & 31;
    atomicOr(&child_mask[word_idx], 1u << bit_idx);
  }

  GPUVDB_DEVICE index_type get_child(uint32_t offset) const {
    return children[offset];
  }
};

// Root node: similar to internal node but at root level
template <typename ValueType, typename Config>
struct root_node {
  using value_type  = ValueType;
  using config_type = Config;
  using index_type  = typename Config::index_type;

  static constexpr uint32_t log2dim = Config::root_log2dim;
  static constexpr uint32_t dim     = Config::root_dim;
  static constexpr uint32_t size    = Config::root_size;

  static constexpr uint32_t num_bitmask_words = (size + 31) / 32;

  index_type children[size];
  uint32_t child_mask[num_bitmask_words];

  GPUVDB_HOST_DEVICE root_node() {
    for (uint32_t i = 0; i < size; ++i) {
      children[i] = Config::invalid_index;
    }
    for (uint32_t i = 0; i < num_bitmask_words; ++i) {
      child_mask[i] = 0;
    }
  }

  GPUVDB_DEVICE static uint32_t coord_to_offset(uint32_t x, uint32_t y, uint32_t z) {
    return (x << (2 * log2dim)) | (y << log2dim) | z;
  }

  GPUVDB_DEVICE bool has_child(uint32_t offset) const {
    uint32_t word_idx = offset >> 5;
    uint32_t bit_idx  = offset & 31;
    return (child_mask[word_idx] & (1u << bit_idx)) != 0;
  }

  GPUVDB_DEVICE void set_child(uint32_t offset, index_type child_index) {
    children[offset] = child_index;
    uint32_t word_idx = offset >> 5;
    uint32_t bit_idx  = offset & 31;
    atomicOr(&child_mask[word_idx], 1u << bit_idx);
  }

  GPUVDB_DEVICE index_type get_child(uint32_t offset) const {
    return children[offset];
  }
};

}  // namespace detail
}  // namespace gpuvdb

