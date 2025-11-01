/*
 * GPU-VDB: GPU-native sparse voxel database
 * Copyright (c) 2025 Muhammad Awad
 * Licensed under the MIT License
 */

#pragma once

#include <gpuvdb/detail/config.hpp>
#include <gpuvdb/detail/macros.hpp>
#include <gpuvdb/detail/pool_allocator.hpp>
#include <gpuvdb/detail/vdb_kernels.hpp>
#include <gpuvdb/detail/vdb_node.hpp>
#include <gpuvdb/detail/vdb_tree.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace gpuvdb {

// Main VDB tree class - host-side API
template <typename ValueType, typename Config = vdb_config<>>
class vdb_tree {
public:
  using value_type  = ValueType;
  using config_type = Config;
  using coord_type  = typename Config::coord_type;
  using coord3_type = coord3<coord_type>;
  using tree_impl_type = detail::vdb_tree_impl<ValueType, Config>;

  vdb_tree() : tree_impl_() {}

  // Initialize tree with capacity
  void initialize(uint32_t max_internal_nodes = 100000, uint32_t max_leaf_nodes = 1000000) {
    tree_impl_.init(max_internal_nodes, max_leaf_nodes);
  }

  // Clean up resources
  void free() {
    tree_impl_.free();
  }

  // Set voxel values (batch operation)
  void set_values(const coord3_type* coords,
                 const value_type* values,
                 uint32_t num_voxels,
                 void* stream = nullptr) {
    const uint32_t block_size = 256;
    const uint32_t num_blocks = (num_voxels + block_size - 1) / block_size;

    uint32_t* d_success = nullptr;
#ifdef __HIP_PLATFORM_AMD__
    hipMalloc(&d_success, sizeof(uint32_t));
    uint32_t h_success = 1u;
    hipMemcpy(d_success, &h_success, sizeof(uint32_t), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(
        (kernels::set_values_kernel<tree_impl_type, coord3_type, value_type>),
        dim3(num_blocks), dim3(block_size), 0, (hipStream_t)stream,
        tree_impl_, coords, values, num_voxels, d_success);

    hipFree(d_success);
#else
    cudaMalloc(&d_success, sizeof(uint32_t));
    uint32_t h_success = 1u;
    cudaMemcpy(d_success, &h_success, sizeof(uint32_t), cudaMemcpyHostToDevice);

    kernels::set_values_kernel<<<num_blocks, block_size, 0, (cudaStream_t)stream>>>(
        tree_impl_, coords, values, num_voxels, d_success);

    cudaFree(d_success);
#endif
  }

  // Get voxel values (batch operation)
  void get_values(const coord3_type* coords,
                 value_type* values,
                 uint32_t num_voxels,
                 void* stream = nullptr) const {
    const uint32_t block_size = 256;
    const uint32_t num_blocks = (num_voxels + block_size - 1) / block_size;

#ifdef __HIP_PLATFORM_AMD__
    hipLaunchKernelGGL(
        (kernels::get_values_kernel<tree_impl_type, coord3_type, value_type>),
        dim3(num_blocks), dim3(block_size), 0, (hipStream_t)stream,
        tree_impl_, coords, values, num_voxels);
#else
    kernels::get_values_kernel<<<num_blocks, block_size, 0, (cudaStream_t)stream>>>(
        tree_impl_, coords, values, num_voxels);
#endif
  }

  // Check if voxels are active (batch operation)
  void is_active(const coord3_type* coords,
                bool* active,
                uint32_t num_voxels,
                void* stream = nullptr) const {
    const uint32_t block_size = 256;
    const uint32_t num_blocks = (num_voxels + block_size - 1) / block_size;

#ifdef __HIP_PLATFORM_AMD__
    hipLaunchKernelGGL(
        (kernels::is_active_kernel<tree_impl_type, coord3_type>),
        dim3(num_blocks), dim3(block_size), 0, (hipStream_t)stream,
        tree_impl_, coords, active, num_voxels);
#else
    kernels::is_active_kernel<<<num_blocks, block_size, 0, (cudaStream_t)stream>>>(
        tree_impl_, coords, active, num_voxels);
#endif
  }

  // Fill a sphere with voxels
  void fill_sphere(int32_t center_x, int32_t center_y, int32_t center_z,
                  float radius,
                  value_type value,
                  void* stream = nullptr) {
    int32_t grid_size = static_cast<int32_t>(radius * 2.0f + 1.0f);
    int32_t start_x = center_x - static_cast<int32_t>(radius);
    int32_t start_y = center_y - static_cast<int32_t>(radius);
    int32_t start_z = center_z - static_cast<int32_t>(radius);

    dim3 block_size(8, 8, 8);
    dim3 num_blocks((grid_size + 7) / 8, (grid_size + 7) / 8, (grid_size + 7) / 8);

#ifdef __HIP_PLATFORM_AMD__
    hipLaunchKernelGGL(
        (kernels::fill_sphere_kernel<tree_impl_type, value_type>),
        num_blocks, block_size, 0, (hipStream_t)stream,
        tree_impl_, center_x, center_y, center_z, radius, value, start_x, start_y, start_z);
#else
    kernels::fill_sphere_kernel<<<num_blocks, block_size, 0, (cudaStream_t)stream>>>(
        tree_impl_, center_x, center_y, center_z, radius, value, start_x, start_y, start_z);
#endif
  }

  // Get memory usage statistics
  void get_memory_stats(uint32_t& num_internal_nodes, uint32_t& num_leaf_nodes) const {
    num_internal_nodes = tree_impl_.internal_alloc_.get_count();
    num_leaf_nodes = tree_impl_.leaf_alloc_.get_count();
  }

  // Accessor methods for tree export (used by export_quad_mesh)
  using root_node_type = typename tree_impl_type::root_node_type;
  using internal_node_type = typename tree_impl_type::internal_node_type;
  using leaf_node_type = typename tree_impl_type::leaf_node_type;
  using internal_allocator_type = typename tree_impl_type::internal_allocator_type;
  using leaf_allocator_type = typename tree_impl_type::leaf_allocator_type;

  root_node_type* get_root() const { return tree_impl_.root_; }
  internal_allocator_type& get_internal_allocator() { return tree_impl_.internal_alloc_; }
  leaf_allocator_type& get_leaf_allocator() { return tree_impl_.leaf_alloc_; }

private:
  tree_impl_type tree_impl_;
};

}  // namespace gpuvdb

