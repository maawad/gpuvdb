/*
 * GPU-VDB: GPU-native sparse voxel database
 * Copyright (c) 2025 Muhammad Awad
 * Licensed under the MIT License
 */

#pragma once

#ifdef __HIP_PLATFORM_AMD__
#define GPUVDB_DEVICE __device__
#define GPUVDB_HOST __host__
#define GPUVDB_HOST_DEVICE __host__ __device__
#define GPUVDB_FORCEINLINE __forceinline__
#else
#define GPUVDB_DEVICE __device__
#define GPUVDB_HOST __host__
#define GPUVDB_HOST_DEVICE __host__ __device__
#define GPUVDB_FORCEINLINE __forceinline__
#endif

// Cooperative groups - not currently used
// Uncomment if needed for future features:
// #ifdef __HIP_PLATFORM_AMD__
// #include <hip/hip_cooperative_groups.h>
// namespace cg = cooperative_groups;
// #endif

