/*
 * GPU-VDB Python Bindings
 * Copyright (c) 2025 Muhammad Awad
 * Licensed under the MIT License
 */

#include <gpuvdb/gpuvdb.hpp>
#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>

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
             "Get (num_internal_nodes, num_leaf_nodes)");
}

