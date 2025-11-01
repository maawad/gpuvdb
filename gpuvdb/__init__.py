"""
GPU-VDB: GPU-native sparse voxel database for AMD GPUs
Copyright (c) 2025 Muhammad Awad
Licensed under the MIT License
"""

__version__ = "0.1.0"
__author__ = "Muhammad Awad"

# Try to import the compiled extension
try:
    from ._gpuvdb import VDBTree as _VDBTreeBase

    # Wrap the C++ extension with a Pythonic interface
    class VDBTree:
        """GPU-VDB sparse voxel tree with PyTorch integration"""

        def __init__(self, max_internal=100000, max_leaf=1000000):
            """Initialize VDB tree with memory capacity

            Args:
                max_internal: Maximum internal nodes (default: 100000)
                max_leaf: Maximum leaf nodes (default: 1000000)
            """
            self._tree = _VDBTreeBase()
            self._tree.initialize(max_internal, max_leaf)

        def free(self):
            """Free GPU memory"""
            self._tree.free()

        def fill_sphere(self, center_x, center_y, center_z, radius, value):
            """Fill a spherical region with voxels

            Args:
                center_x, center_y, center_z: Sphere center coordinates
                radius: Sphere radius
                value: Voxel value to fill
            """
            self._tree.fill_sphere(center_x, center_y, center_z, radius, value)

        def insert(self, coords, values):
            """Insert voxels at specified coordinates

            Args:
                coords: torch.Tensor[N, 3] (int32, device='cuda')
                values: torch.Tensor[N] (float32, device='cuda')
            """
            import torch
            if not isinstance(coords, torch.Tensor) or not isinstance(values, torch.Tensor):
                raise TypeError("coords and values must be torch.Tensor")
            if not coords.is_cuda:
                raise ValueError("coords must be on GPU (device='cuda')")
            if not values.is_cuda:
                raise ValueError("values must be on GPU (device='cuda')")
            if coords.dtype != torch.int32:
                raise TypeError("coords must be int32")
            if values.dtype != torch.float32:
                raise TypeError("values must be float32")
            if coords.dim() != 2 or coords.size(1) != 3:
                raise ValueError("coords must be shape [N, 3]")
            if values.dim() != 1:
                raise ValueError("values must be shape [N]")
            if len(coords) != len(values):
                raise ValueError("coords and values must have same length")

            self._tree.insert(coords.data_ptr(), values.data_ptr(), len(coords))

        def query(self, coords):
            """Query voxel values at specified coordinates

            Args:
                coords: torch.Tensor[N, 3] (int32, device='cuda')

            Returns:
                torch.Tensor[N] (float32, device='cuda'): Voxel values
            """
            import torch
            if not isinstance(coords, torch.Tensor):
                raise TypeError("coords must be torch.Tensor")
            if not coords.is_cuda:
                raise ValueError("coords must be on GPU (device='cuda')")
            if coords.dtype != torch.int32:
                raise TypeError("coords must be int32")
            if coords.dim() != 2 or coords.size(1) != 3:
                raise ValueError("coords must be shape [N, 3]")

            values = torch.zeros(len(coords), dtype=torch.float32, device=coords.device)
            self._tree.query(coords.data_ptr(), values.data_ptr(), len(coords))
            return values

        def active(self, coords):
            """Check which voxels are active

            Args:
                coords: torch.Tensor[N, 3] (int32, device='cuda')

            Returns:
                torch.Tensor[N] (bool, device='cuda'): Active flags
            """
            import torch
            if not isinstance(coords, torch.Tensor):
                raise TypeError("coords must be torch.Tensor")
            if not coords.is_cuda:
                raise ValueError("coords must be on GPU (device='cuda')")
            if coords.dtype != torch.int32:
                raise TypeError("coords must be int32")
            if coords.dim() != 2 or coords.size(1) != 3:
                raise ValueError("coords must be shape [N, 3]")

            active = torch.zeros(len(coords), dtype=torch.bool, device=coords.device)
            self._tree.is_active(coords.data_ptr(), active.data_ptr(), len(coords))
            return active

        def get_memory_stats(self):
            """Get memory statistics

            Returns:
                tuple: (num_internal_nodes, num_leaf_nodes)
            """
            return self._tree.get_memory_stats()

        def export_quad_mesh(self):
            """Export tree structure as a quad mesh

            Returns:
                dict: {
                    'vertices': numpy.ndarray[V, 3] float32 - vertex positions
                    'quads': numpy.ndarray[F, 4] int32 - quad face indices
                    'levels': numpy.ndarray[F] int32 - hierarchy level per face (0=leaf, 1=internal, 2=root)
                }
            """
            return self._tree.export_quad_mesh()

        # Context manager support
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.free()
            return False

    __all__ = ['VDBTree']

except ImportError as e:
    import warnings
    warnings.warn(
        f"GPU-VDB native extension not found. Please install:\n"
        f"  pip install -e .\n"
        f"Error: {e}"
    )
    __all__ = []
