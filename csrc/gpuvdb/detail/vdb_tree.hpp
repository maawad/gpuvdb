/*
 * GPU-VDB: GPU-native sparse voxel database
 * Copyright (c) 2025 Muhammad Awad
 * Licensed under the MIT License
 */

#pragma once
#include <gpuvdb/detail/config.hpp>
#include <gpuvdb/detail/macros.hpp>
#include <gpuvdb/detail/pool_allocator.hpp>
#include <gpuvdb/detail/vdb_node.hpp>

namespace gpuvdb {
namespace detail {

// GPU-native VDB tree implementation
template <typename ValueType, typename Config = vdb_config<>>
struct vdb_tree_impl {
  using value_type   = ValueType;
  using config_type  = Config;
  using coord_type   = typename Config::coord_type;
  using index_type   = typename Config::index_type;
  using coord3_type  = coord3<coord_type>;

  using leaf_node_type     = leaf_node<ValueType, Config>;
  using internal_node_type = internal_node<ValueType, Config>;
  using root_node_type     = root_node<ValueType, Config>;

  using leaf_allocator_type     = pool_allocator<leaf_node_type>;
  using internal_allocator_type = pool_allocator<internal_node_type>;

  // Root node (allocated on device)
  root_node_type* root_;

  // Allocators for internal and leaf nodes
  internal_allocator_type internal_alloc_;
  leaf_allocator_type leaf_alloc_;

  // Background value for inactive voxels
  value_type background_;

  vdb_tree_impl() : root_(nullptr), background_(value_type{}) {}

  // Initialize tree with specified capacity
  void init(uint32_t max_internal_nodes, uint32_t max_leaf_nodes) {
    // Allocate root node
#ifdef __HIP_PLATFORM_AMD__
    hipMalloc(&root_, sizeof(root_node_type));
    root_node_type host_root;
    hipMemcpy(root_, &host_root, sizeof(root_node_type), hipMemcpyHostToDevice);
#else
    cudaMalloc(&root_, sizeof(root_node_type));
    root_node_type host_root;
    cudaMemcpy(root_, &host_root, sizeof(root_node_type), cudaMemcpyHostToDevice);
#endif

    // Initialize allocators
    internal_alloc_.init(max_internal_nodes);
    leaf_alloc_.init(max_leaf_nodes);
  }

  // Clean up
  void free() {
    if (root_) {
#ifdef __HIP_PLATFORM_AMD__
      hipFree(root_);
#else
      cudaFree(root_);
#endif
      root_ = nullptr;
    }
    internal_alloc_.free();
    leaf_alloc_.free();
  }

  // Decompose global coordinate into (root, internal, leaf) local coordinates
  GPUVDB_DEVICE static void decompose_coord(const coord3_type& coord,
                                           uint32_t& root_x, uint32_t& root_y, uint32_t& root_z,
                                           uint32_t& internal_x, uint32_t& internal_y, uint32_t& internal_z,
                                           uint32_t& leaf_x, uint32_t& leaf_y, uint32_t& leaf_z) {
    // Extract leaf coordinates
    leaf_x = coord.x & Config::leaf_mask;
    leaf_y = coord.y & Config::leaf_mask;
    leaf_z = coord.z & Config::leaf_mask;

    // Extract internal coordinates
    internal_x = (coord.x >> Config::internal_offset) & Config::internal_mask;
    internal_y = (coord.y >> Config::internal_offset) & Config::internal_mask;
    internal_z = (coord.z >> Config::internal_offset) & Config::internal_mask;

    // Extract root coordinates
    root_x = (coord.x >> Config::root_offset) & Config::root_mask;
    root_y = (coord.y >> Config::root_offset) & Config::root_mask;
    root_z = (coord.z >> Config::root_offset) & Config::root_mask;
  }

  // Set voxel value at coordinate
  GPUVDB_DEVICE bool set_value(const coord3_type& coord, const value_type& value) {
    uint32_t root_x, root_y, root_z;
    uint32_t internal_x, internal_y, internal_z;
    uint32_t leaf_x, leaf_y, leaf_z;

    decompose_coord(coord, root_x, root_y, root_z,
                   internal_x, internal_y, internal_z,
                   leaf_x, leaf_y, leaf_z);

    // Traverse from root
    uint32_t root_offset = root_node_type::coord_to_offset(root_x, root_y, root_z);

    // Get or create internal node (with CAS to handle races)
    index_type internal_idx;
    while (true) {
      if (root_->has_child(root_offset)) {
        internal_idx = root_->get_child(root_offset);
        break;
      } else {
        // Try to allocate and install
        index_type new_idx = internal_alloc_.allocate();
        if (new_idx == Config::invalid_index) {
          return false;  // Out of memory
        }
        // Try to install it atomically
        index_type expected = Config::invalid_index;
        index_type* child_ptr = &(root_->children[root_offset]);
        index_type old = atomicCAS(child_ptr, expected, new_idx);
        if (old == Config::invalid_index) {
          // We won the race, mark it in the bitmask
          root_->set_child(root_offset, new_idx);
          internal_idx = new_idx;
          break;
        } else {
          // Someone else installed it, use theirs
          internal_idx = old;
          break;
        }
      }
    }

    internal_node_type* internal_node = internal_alloc_.get_node(internal_idx);
    uint32_t internal_offset = internal_node_type::coord_to_offset(internal_x, internal_y, internal_z);

    // Get or create leaf node (with CAS to handle races)
    index_type leaf_idx;
    while (true) {
      if (internal_node->has_child(internal_offset)) {
        leaf_idx = internal_node->get_child(internal_offset);
        break;
      } else {
        // Try to allocate and install
        index_type new_idx = leaf_alloc_.allocate();
        if (new_idx == Config::invalid_index) {
          return false;  // Out of memory
        }
        // Try to install it atomically
        index_type expected = Config::invalid_index;
        index_type* child_ptr = &(internal_node->children[internal_offset]);
        index_type old = atomicCAS(child_ptr, expected, new_idx);
        if (old == Config::invalid_index) {
          // We won the race, mark it in the bitmask
          internal_node->set_child(internal_offset, new_idx);
          leaf_idx = new_idx;
          break;
        } else {
          // Someone else installed it, use theirs
          leaf_idx = old;
          break;
        }
      }
    }

    leaf_node_type* leaf = leaf_alloc_.get_node(leaf_idx);
    uint32_t leaf_offset = leaf_node_type::coord_to_offset(leaf_x, leaf_y, leaf_z);

    // Set value and mark as active
    leaf->set_value(leaf_offset, value);
    leaf->set_active(leaf_offset, true);

    return true;
  }

  // Get voxel value at coordinate
  GPUVDB_DEVICE value_type get_value(const coord3_type& coord) const {
    uint32_t root_x, root_y, root_z;
    uint32_t internal_x, internal_y, internal_z;
    uint32_t leaf_x, leaf_y, leaf_z;

    decompose_coord(coord, root_x, root_y, root_z,
                   internal_x, internal_y, internal_z,
                   leaf_x, leaf_y, leaf_z);

    uint32_t root_offset = root_node_type::coord_to_offset(root_x, root_y, root_z);

    if (!root_->has_child(root_offset)) {
      return background_;
    }

    index_type internal_idx = root_->get_child(root_offset);
    const internal_node_type* internal_node = internal_alloc_.get_node(internal_idx);
    uint32_t internal_offset = internal_node_type::coord_to_offset(internal_x, internal_y, internal_z);

    if (!internal_node->has_child(internal_offset)) {
      return background_;
    }

    index_type leaf_idx = internal_node->get_child(internal_offset);
    const leaf_node_type* leaf = leaf_alloc_.get_node(leaf_idx);
    uint32_t leaf_offset = leaf_node_type::coord_to_offset(leaf_x, leaf_y, leaf_z);

    if (!leaf->is_active(leaf_offset)) {
      return background_;
    }

    return leaf->get_value(leaf_offset);
  }

  // Check if voxel is active
  GPUVDB_DEVICE bool is_active(const coord3_type& coord) const {
    uint32_t root_x, root_y, root_z;
    uint32_t internal_x, internal_y, internal_z;
    uint32_t leaf_x, leaf_y, leaf_z;

    decompose_coord(coord, root_x, root_y, root_z,
                   internal_x, internal_y, internal_z,
                   leaf_x, leaf_y, leaf_z);

    uint32_t root_offset = root_node_type::coord_to_offset(root_x, root_y, root_z);

    if (!root_->has_child(root_offset)) {
      return false;
    }

    index_type internal_idx = root_->get_child(root_offset);
    const internal_node_type* internal_node = internal_alloc_.get_node(internal_idx);
    uint32_t internal_offset = internal_node_type::coord_to_offset(internal_x, internal_y, internal_z);

    if (!internal_node->has_child(internal_offset)) {
      return false;
    }

    index_type leaf_idx = internal_node->get_child(internal_offset);
    const leaf_node_type* leaf = leaf_alloc_.get_node(leaf_idx);
    uint32_t leaf_offset = leaf_node_type::coord_to_offset(leaf_x, leaf_y, leaf_z);

    return leaf->is_active(leaf_offset);
  }
};

}  // namespace detail
}  // namespace gpuvdb

