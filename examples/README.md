# GPU-VDB Examples

Each example generates two visualizations:
- **`*_2d.png`**: 2D slices with tree structure overlay
- **`*_3d.png`**: 3D point cloud of voxel data

---

## [01: Simple Sphere](01_simple_sphere/)
Single sphere with accuracy validation against analytical formula.

| 2D (with tree) | 3D (voxels) |
|:---:|:---:|
| ![](01_simple_sphere/sphere_2d.png) | ![](01_simple_sphere/sphere_3d.png) |

---

## [02: Multi Spheres](02_multi_spheres/)
Multiple overlapping spheres with different values demonstrating sparse storage.

| 2D (with tree) | 3D (voxels) |
|:---:|:---:|
| ![](02_multi_spheres/multisphere_2d.png) | ![](02_multi_spheres/multisphere_3d.png) |

---

## [03: Fractal Voxels](03_fractal_voxels/)
Mandelbulb fractal surface sampling showing complex procedural geometry.

| 2D (with tree) | 3D (voxels) |
|:---:|:---:|
| ![](03_fractal_voxels/fractal_2d.png) | ![](03_fractal_voxels/fractal_3d.png) |

---

## [04: Level Set](04_level_set/)
Signed distance field (SDF) narrow-band for sphere, box, and torus union.

| 2D (with tree) | 3D (voxels) |
|:---:|:---:|
| ![](04_level_set/levelset_2d.png) | ![](04_level_set/levelset_3d.png) |

---

## [05: Procedural Noise](05_procedural_noise/)
3D Perlin noise field with fractal Brownian motion (FBM) octaves.

| 2D (with tree) | 3D (voxels) |
|:---:|:---:|
| ![](05_procedural_noise/noise_2d.png) | ![](05_procedural_noise/noise_3d.png) |

---

## [06: Metaballs](06_metaballs/)
Organic blob field using implicit surface distance functions.

| 2D (with tree) | 3D (voxels) |
|:---:|:---:|
| ![](06_metaballs/metaballs_2d.png) | ![](06_metaballs/metaballs_3d.png) |

---

## [07: 3D Conway](07_conway_3d/)
3D cellular automaton with custom rules showing oscillating patterns.

| 2D (with tree) | 3D (voxels) |
|:---:|:---:|
| ![](07_conway_3d/conway_2d.png) | ![](07_conway_3d/conway_3d.png) |

---

## [08: Voxel Art](08_voxel_art/)
Procedurally generated skull demonstrating artistic voxel composition.

| 2D (with tree) | 3D (voxels) |
|:---:|:---:|
| ![](08_voxel_art/skull_2d.png) | ![](08_voxel_art/skull_3d.png) |

---

## [09: 3D Heart](09_heart/)
Romantic 3D heart shape using parametric equations with height-based gradient.

| 2D (with tree) | 3D (voxels) |
|:---:|:---:|
| ![](09_heart/heart_2d.png) | ![](09_heart/heart_3d.png) |

---

## [10: Showcase](10_showcase/)
Filled 3D heart (41K voxels) with hierarchical tree visualization - perfect demo!

| 2D (with tree) | 3D (voxels) |
|:---:|:---:|
| ![](10_showcase/showcase_2d.png) | ![](10_showcase/showcase_3d.png) |

---

## Running Examples

```bash
module load pytorch
cd examples/01_simple_sphere
python3 simple_sphere.py
```

All examples:
- Return 0 on success
- Include validation
- Generate `{name}_2d.png` and `{name}_3d.png`
- Show VDB tree structure (colored boxes: Blue=Root, Green=Internal, Red=Leaf)