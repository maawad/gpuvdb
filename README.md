# GPU-VDB

**GPU-native VDB implementation for AMD GPUs**

A high-performance sparse voxel data structure designed from the ground up for GPU execution on AMD Instinct accelerators.

## Overview

GPU-VDB is a header-only C++ library implementing a sparse voxel database optimized for AMD GPUs using HIP. Unlike CPU-to-GPU ports, GPU-VDB is designed with GPU-first principles following BGHT/MVGpuBTree patterns.

**Key Features:**
- 🚀 GPU-native design (not a CPU port)
- 🎯 Lock-free concurrent operations with atomic CAS loops
- 💾 Sparse storage - only active voxels consume memory
- ⚡ Zero-copy Python bindings via raw device pointers
- 🔧 Header-only C++ library
- 📦 pip installable
- 🎨 Tested on AMD Instinct MI300X (gfx942)

## Installation

### Prerequisites
- ROCm 6.3+ with HIP compiler (hipcc)
- Python 3.8+
- PyTorch with ROCm support

### Install from source

```bash
# Clone repository
git clone https://github.com/maawad/gpuvdb.git
cd gpuvdb

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

## Quick Start

```python
import torch
import gpuvdb

# Initialize VDB tree
tree = gpuvdb.VDBTree(max_internal=100000, max_leaf=1000000)

# Fill a sphere (GPU kernel)
tree.fill_sphere(center_x=50, center_y=50, center_z=50,
                 radius=20.0, value=1.0)

# Insert voxels
coords = torch.tensor([[10,10,10], [20,20,20]], dtype=torch.int32, device='cuda')
values = torch.tensor([1.0, 2.0], dtype=torch.float32, device='cuda')
tree.insert(coords, values)

# Query voxels (returns GPU tensors)
query_coords = torch.tensor([[50,50,50], [60,60,60]], dtype=torch.int32, device='cuda')
result_values = tree.query(query_coords)
active_flags = tree.active(query_coords)

print(result_values)  # tensor([1., 0.], device='cuda:0')
print(active_flags)   # tensor([True, False], device='cuda:0')

# Cleanup
tree.free()

# Or use context manager
with gpuvdb.VDBTree() as tree:
    tree.fill_sphere(50, 50, 50, 20.0, 1.0)
    # ... automatically freed
```

## Architecture

### Hierarchical Structure
```
Root Node:     32³ cells → Internal nodes
Internal Node: 16³ cells → Leaf nodes
Leaf Node:     8³ cells  → Voxel data
```

### Design Principles
- **Flat memory layouts**: Arrays instead of pointers for coalesced access
- **Pool allocators**: O(1) atomic allocation with counters
- **Lock-free**: Atomic CAS loops prevent race conditions
- **Bitmasks**: Cache-friendly active voxel tracking

## API Reference

### VDBTree Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `__init__(max_internal=100000, max_leaf=1000000)` | Initialize tree with GPU memory | None |
| `free()` | Release GPU memory | None |
| `fill_sphere(cx, cy, cz, radius, value)` | Fill spherical region | None |
| `insert(coords, values)` | Insert voxels at coordinates | None |
| `query(coords)` | Query voxel values | Tensor[N] (float32) |
| `active(coords)` | Check which voxels are active | Tensor[N] (bool) |
| `get_memory_stats()` | Get node counts | (int, int) |

### Input/Output Format
- **Coordinates**: `torch.Tensor[N, 3]` (dtype=int32, device='cuda')
- **Values**: `torch.Tensor[N]` (dtype=float32, device='cuda')
- **Returns**: GPU tensors (no host-device copy!)

All tensors must be on GPU. The API handles tensor validation and provides helpful error messages.

## Examples

Eight visualization examples demonstrating different use cases:

| Example | Description | Voxels |
|---------|-------------|--------|
| [01_simple_sphere](examples/01_simple_sphere/) | Single sphere with accuracy validation | 1,256 |
| [02_multi_spheres](examples/02_multi_spheres/) | Multiple overlapping spheres | 2,724 |
| [03_fractal_voxels](examples/03_fractal_voxels/) | Mandelbulb fractal (power=8) | 388K |
| [04_level_set](examples/04_level_set/) | Signed distance field visualization | 46K |
| [05_procedural_noise](examples/05_procedural_noise/) | 3D Perlin noise field | 1M |
| [06_metaballs](examples/06_metaballs/) | Organic blob simulation with smooth blending | 122K |
| [07_conway_3d](examples/07_conway_3d/) | 3D cellular automaton (Game of Life) | Varies |
| [08_voxel_art](examples/08_voxel_art/) | Procedural castle with towers | Varies |

Run any example:
```bash
cd examples/01_simple_sphere
python3 simple_sphere.py
```

## Performance

Tested on AMD Instinct MI300X:
- **Sphere accuracy**: 99.95% (1256/1256.6 voxels)
- **Allocation**: Lock-free O(1) atomic counter
- **Memory**: Sparse - only active voxels stored
- **Compilation**: ~10 seconds for full library

## Development

```bash
# Install in development mode
pip install -e ".[dev,viz]"

# Run tests
pytest

# Format code
ruff format .

# Lint
ruff check .
```

## License

MIT License - Copyright (c) 2025 Muhammad Awad

See [LICENSE](LICENSE) file for details.

## Author

**Muhammad Awad** - [mawad@duck.com](mailto:mawad@duck.com)

GPU-VDB is designed for high-performance voxel processing on AMD accelerators.
