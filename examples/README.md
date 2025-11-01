# GPU-VDB Examples

## [01: Simple Sphere](01_simple_sphere/)
Single sphere with accuracy validation against analytical formula.

![](01_simple_sphere/output.png)

---

## [02: Multi Spheres](02_multi_spheres/)
Multiple overlapping spheres with different values demonstrating sparse storage.

![](02_multi_spheres/output.png)

---

## [03: Fractal Voxels](03_fractal_voxels/)
Mandelbulb fractal surface sampling showing complex procedural geometry.

![](03_fractal_voxels/output.png)

---

## [04: Level Set](04_level_set/)
Signed distance field (SDF) narrow-band for sphere, box, and torus union.

![](04_level_set/output.png)

---

## [05: Procedural Noise](05_procedural_noise/)
3D Perlin noise field with fractal Brownian motion (FBM) octaves.

![](05_procedural_noise/output.png)

---

## [06: Metaballs](06_metaballs/)
Organic blob field using implicit surface distance functions.

![](06_metaballs/output.png)

---

## [07: 3D Conway](07_conway_3d/)
3D cellular automaton with custom rules showing oscillating patterns.

![](07_conway_3d/output.png)

---

## [08: Voxel Art](08_voxel_art/)
Procedurally generated skull demonstrating artistic voxel composition.

![](08_voxel_art/output.png)

---

## Running Examples

```bash
module load pytorch
cd examples/01_simple_sphere
python3 simple_sphere.py
```

All examples return 0 on success and include validation.

