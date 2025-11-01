/*
 * GPU-VDB Python Bindings
 * Copyright (c) 2025 Muhammad Awad
 * Licensed under the MIT License
 */

#include <gpuvdb/gpuvdb.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace py = pybind11;

using VDBTree = gpuvdb::vdb_tree<float>;
using Coord3 = gpuvdb::coord3<int32_t>;

class VDBTreeWrapper {
public:
    VDBTree tree;

    VDBTreeWrapper() {}

    void initialize(int64_t max_internal, int64_t max_leaf) {
        tree.initialize(max_internal, max_leaf);
    }

    void free() {
        tree.free();
    }

    void fill_sphere(int32_t cx, int32_t cy, int32_t cz, float radius, float value) {
        tree.fill_sphere(cx, cy, cz, radius, value);
        hipDeviceSynchronize();
    }

    // Python-friendly API accepting integers (from tensor.data_ptr())
    void insert(uintptr_t coords_ptr, uintptr_t values_ptr, int64_t n) {
        Coord3* d_coords = reinterpret_cast<Coord3*>(coords_ptr);
        float* d_values = reinterpret_cast<float*>(values_ptr);
        tree.set_values(d_coords, d_values, n);
        hipDeviceSynchronize();
    }

    void query(uintptr_t coords_ptr, uintptr_t values_ptr, int64_t n) {
        Coord3* d_coords = reinterpret_cast<Coord3*>(coords_ptr);
        float* d_values = reinterpret_cast<float*>(values_ptr);
        tree.get_values(d_coords, d_values, n);
        hipDeviceSynchronize();
    }

    void is_active(uintptr_t coords_ptr, uintptr_t active_ptr, int64_t n) {
        Coord3* d_coords = reinterpret_cast<Coord3*>(coords_ptr);
        bool* d_active = reinterpret_cast<bool*>(active_ptr);
        tree.is_active(d_coords, d_active, n);
        hipDeviceSynchronize();
    }

    py::tuple get_memory_stats() {
        uint32_t num_internal, num_leaf;
        tree.get_memory_stats(num_internal, num_leaf);
        return py::make_tuple(num_internal, num_leaf);
    }

    // Export tree structure as a quad mesh (CPU tensors)
    py::dict export_quad_mesh() {
        using Config = VDBTree::config_type;
        using RootNode = VDBTree::root_node_type;
        using InternalNode = VDBTree::internal_node_type;
        using LeafNode = VDBTree::leaf_node_type;

        std::vector<float> vertices;
        std::vector<int32_t> quads;
        std::vector<int32_t> levels;

        // Helper: add cube for a bounding box
        auto add_cube = [&](float min_x, float min_y, float min_z,
                           float max_x, float max_y, float max_z, int32_t level) {
            int32_t base_idx = vertices.size() / 3;

            // 8 vertices of cube
            vertices.push_back(min_x); vertices.push_back(min_y); vertices.push_back(min_z); // 0
            vertices.push_back(max_x); vertices.push_back(min_y); vertices.push_back(min_z); // 1
            vertices.push_back(max_x); vertices.push_back(max_y); vertices.push_back(min_z); // 2
            vertices.push_back(min_x); vertices.push_back(max_y); vertices.push_back(min_z); // 3
            vertices.push_back(min_x); vertices.push_back(min_y); vertices.push_back(max_z); // 4
            vertices.push_back(max_x); vertices.push_back(min_y); vertices.push_back(max_z); // 5
            vertices.push_back(max_x); vertices.push_back(max_y); vertices.push_back(max_z); // 6
            vertices.push_back(min_x); vertices.push_back(max_y); vertices.push_back(max_z); // 7

            // 6 quad faces (counter-clockwise winding)
            int32_t faces[6][4] = {
                {0, 1, 2, 3},  // front  (-Z)
                {5, 4, 7, 6},  // back   (+Z)
                {4, 0, 3, 7},  // left   (-X)
                {1, 5, 6, 2},  // right  (+X)
                {4, 5, 1, 0},  // bottom (-Y)
                {3, 2, 6, 7}   // top    (+Y)
            };

            for (int f = 0; f < 6; f++) {
                quads.push_back(base_idx + faces[f][0]);
                quads.push_back(base_idx + faces[f][1]);
                quads.push_back(base_idx + faces[f][2]);
                quads.push_back(base_idx + faces[f][3]);
                levels.push_back(level);
            }
        };

        // Copy root node from GPU to CPU
        RootNode host_root;
        hipMemcpy(&host_root, tree.get_root(), sizeof(RootNode), hipMemcpyDeviceToHost);

        // Iterate through root node's children (internal nodes)
        for (uint32_t rx = 0; rx < Config::root_dim; rx++) {
            for (uint32_t ry = 0; ry < Config::root_dim; ry++) {
                for (uint32_t rz = 0; rz < Config::root_dim; rz++) {
                    uint32_t root_offset = RootNode::coord_to_offset(rx, ry, rz);

                    if (!host_root.has_child(root_offset)) continue;

                    // This root cell has an internal node child
                    uint32_t internal_idx = host_root.get_child(root_offset);

                    // Compute global coordinates for this root cell
                    int32_t root_min_x = rx << Config::root_offset;
                    int32_t root_min_y = ry << Config::root_offset;
                    int32_t root_min_z = rz << Config::root_offset;
                    int32_t root_dim_voxels = 1 << Config::root_offset;

                    // Add cube for root node (level 2)
                    add_cube(root_min_x, root_min_y, root_min_z,
                            root_min_x + root_dim_voxels,
                            root_min_y + root_dim_voxels,
                            root_min_z + root_dim_voxels, 2);

                    // Copy internal node from GPU to CPU
                    InternalNode host_internal;
                    // Compute device pointer manually (pool base + index * sizeof)
                    InternalNode* d_internal_base = tree.get_internal_allocator().get_pool();
                    InternalNode* d_internal = d_internal_base + internal_idx;
                    hipMemcpy(&host_internal, d_internal, sizeof(InternalNode), hipMemcpyDeviceToHost);

                    // Iterate through internal node's children (leaf nodes)
                    for (uint32_t ix = 0; ix < Config::internal_dim; ix++) {
                        for (uint32_t iy = 0; iy < Config::internal_dim; iy++) {
                            for (uint32_t iz = 0; iz < Config::internal_dim; iz++) {
                                uint32_t internal_offset = InternalNode::coord_to_offset(ix, iy, iz);

                                if (!host_internal.has_child(internal_offset)) continue;

                                // This internal cell has a leaf node child
                                uint32_t leaf_idx = host_internal.get_child(internal_offset);

                                // Compute global coordinates for this internal cell
                                int32_t internal_min_x = root_min_x + (ix << Config::internal_offset);
                                int32_t internal_min_y = root_min_y + (iy << Config::internal_offset);
                                int32_t internal_min_z = root_min_z + (iz << Config::internal_offset);
                                int32_t internal_dim_voxels = 1 << Config::internal_offset;

                                // Add cube for internal node (level 1)
                                add_cube(internal_min_x, internal_min_y, internal_min_z,
                                        internal_min_x + internal_dim_voxels,
                                        internal_min_y + internal_dim_voxels,
                                        internal_min_z + internal_dim_voxels, 1);

                                // Compute leaf node bounding box
                                int32_t leaf_min_x = internal_min_x;
                                int32_t leaf_min_y = internal_min_y;
                                int32_t leaf_min_z = internal_min_z;
                                int32_t leaf_dim_voxels = Config::leaf_dim;

                                // Add cube for leaf node (level 0)
                                add_cube(leaf_min_x, leaf_min_y, leaf_min_z,
                                        leaf_min_x + leaf_dim_voxels,
                                        leaf_min_y + leaf_dim_voxels,
                                        leaf_min_z + leaf_dim_voxels, 0);
                            }
                        }
                    }
                }
            }
        }

        // Convert to NumPy arrays
        size_t num_verts = vertices.size() / 3;
        size_t num_quads = quads.size() / 4;

        auto vertices_array = py::array_t<float>(std::vector<ssize_t>{(ssize_t)num_verts, 3});
        auto quads_array = py::array_t<int32_t>(std::vector<ssize_t>{(ssize_t)num_quads, 4});
        auto levels_array = py::array_t<int32_t>((ssize_t)levels.size());

        // Copy data
        auto verts_buf = vertices_array.request();
        auto quads_buf = quads_array.request();
        auto levels_buf = levels_array.request();

        std::memcpy(verts_buf.ptr, vertices.data(), vertices.size() * sizeof(float));
        std::memcpy(quads_buf.ptr, quads.data(), quads.size() * sizeof(int32_t));
        std::memcpy(levels_buf.ptr, levels.data(), levels.size() * sizeof(int32_t));

        py::dict result;
        result["vertices"] = vertices_array;
        result["quads"] = quads_array;
        result["levels"] = levels_array;

        return result;
    }
};

PYBIND11_MODULE(_gpuvdb, m) {
    m.doc() = "GPU-VDB: GPU-native VDB implementation for AMD GPUs";
    m.attr("__version__") = "0.1.0";

    py::class_<VDBTreeWrapper>(m, "VDBTree")
        .def(py::init<>())
        .def("initialize", &VDBTreeWrapper::initialize,
             py::arg("max_internal") = 100000,
             py::arg("max_leaf") = 1000000,
             "Initialize VDB tree with capacity")
        .def("free", &VDBTreeWrapper::free,
             "Free GPU memory")
        .def("fill_sphere", &VDBTreeWrapper::fill_sphere,
             py::arg("center_x"), py::arg("center_y"), py::arg("center_z"),
             py::arg("radius"), py::arg("value"),
             "Fill a sphere with voxels")
        .def("insert", &VDBTreeWrapper::insert,
             py::arg("coords_ptr"), py::arg("values_ptr"), py::arg("n"),
             "Insert voxels at coordinates with values")
        .def("query", &VDBTreeWrapper::query,
             py::arg("coords_ptr"), py::arg("values_ptr"), py::arg("n"),
             "Query voxel values at coordinates")
        .def("is_active", &VDBTreeWrapper::is_active,
             py::arg("coords_ptr"), py::arg("active_ptr"), py::arg("n"),
             "Check if voxels are active")
        .def("get_memory_stats", &VDBTreeWrapper::get_memory_stats,
             "Get (num_internal_nodes, num_leaf_nodes)")
        .def("export_quad_mesh", &VDBTreeWrapper::export_quad_mesh,
             "Export tree structure as quad mesh (vertices, quads, levels)");
}

