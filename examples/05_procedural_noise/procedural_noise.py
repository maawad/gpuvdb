#!/usr/bin/env python3
"""
GPU-VDB Example 05: Procedural Noise
Copyright (c) 2025 Muhammad Awad
Licensed under the MIT License
"""
import sys
import torch
import gpuvdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def fade(t):
    """Perlin fade function"""
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, t):
    """Linear interpolation"""
    return a + t * (b - a)

def grad(hash_val, x, y, z):
    """Gradient function for Perlin noise"""
    h = hash_val & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

def perlin_noise_3d(x, y, z, perm):
    """3D Perlin noise"""
    xi = int(np.floor(x)) & 255
    yi = int(np.floor(y)) & 255
    zi = int(np.floor(z)) & 255

    xf = x - np.floor(x)
    yf = y - np.floor(y)
    zf = z - np.floor(z)

    u = fade(xf)
    v = fade(yf)
    w = fade(zf)

    aaa = perm[perm[perm[xi] + yi] + zi]
    aba = perm[perm[perm[xi] + yi + 1] + zi]
    aab = perm[perm[perm[xi] + yi] + zi + 1]
    abb = perm[perm[perm[xi] + yi + 1] + zi + 1]
    baa = perm[perm[perm[xi + 1] + yi] + zi]
    bba = perm[perm[perm[xi + 1] + yi + 1] + zi]
    bab = perm[perm[perm[xi + 1] + yi] + zi + 1]
    bbb = perm[perm[perm[xi + 1] + yi + 1] + zi + 1]

    x1 = lerp(grad(aaa, xf, yf, zf), grad(baa, xf - 1, yf, zf), u)
    x2 = lerp(grad(aba, xf, yf - 1, zf), grad(bba, xf - 1, yf - 1, zf), u)
    y1 = lerp(x1, x2, v)

    x1 = lerp(grad(aab, xf, yf, zf - 1), grad(bab, xf - 1, yf, zf - 1), u)
    x2 = lerp(grad(abb, xf, yf - 1, zf - 1), grad(bbb, xf - 1, yf - 1, zf - 1), u)
    y2 = lerp(x1, x2, v)

    return lerp(y1, y2, w)

def fbm_noise(x, y, z, octaves, persistence, lacunarity, perm):
    """Fractal Brownian Motion"""
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0

    for _ in range(octaves):
        total += perlin_noise_3d(x * frequency, y * frequency, z * frequency, perm) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / max_value

def main():
    print("GPU-VDB Example 05: Procedural Noise (Perlin)")

    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return 1

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tree = gpuvdb.VDBTree(200000, 2000000)
    print("VDB tree initialized")

    # Generate Perlin permutation table
    np.random.seed(42)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    perm = np.concatenate([p, p])

    print("Generating 3D Perlin noise field")
    size = 80
    scale = 0.12
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0
    threshold = 0.55

    points = []
    values = []

    for z in range(size):
        for y in range(size):
            for x in range(size):
                noise = fbm_noise(x * scale, y * scale, z * scale,
                                 octaves, persistence, lacunarity, perm)
                noise = (noise + 1.0) * 0.5

                if noise > threshold:
                    points.append([x, y, z])
                    values.append(noise)

    print(f"Found {len(points)} voxels above threshold")

    if len(points) == 0:
        print("ERROR: No points generated")
        return 1

    coords = torch.tensor(points, dtype=torch.int32, device=device)
    vals = torch.tensor(values, dtype=torch.float32, device=device)

    tree.insert(coords, vals)

    num_internal, num_leaf = tree.get_memory_stats()
    sparsity = 100 * (1 - len(points) / (size**3))
    print(f"Internal nodes: {num_internal}, Leaf nodes: {num_leaf}")
    print(f"Sparsity: {sparsity:.1f}%")

    # Validation
    print("\nValidation:")
    num_test = min(100, len(points))
    test_indices = np.random.choice(len(points), num_test, replace=False)
    test_coords_cpu = np.array(points)[test_indices]
    test_values_expected = np.array(values)[test_indices]

    test_coords = torch.tensor(test_coords_cpu, dtype=torch.int32, device=device)
    test_values_queried = tree.query(test_coords).cpu().numpy()
    test_active_queried = tree.active(test_coords).cpu().numpy()

    errors = 0
    active_failed = np.sum(test_active_queried != 1)
    if active_failed > 0:
        print(f"  FAIL: {active_failed}/{num_test} points are not active")
        errors += 1

    # Check if values are above threshold (should be since we only inserted those)
    below_threshold = np.sum(test_values_queried < threshold * 0.9)
    if below_threshold > num_test * 0.1:
        print(f"  FAIL: {below_threshold}/{num_test} values are below threshold")
        errors += 1

    if errors == 0:
        print("  All tests passed")
    else:
        print(f"  {errors} tests failed")
        tree.free()
        return 1

    # Visualize
    print("\nQuerying slices")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    z_slices = [20, 35, 50, 65, 80, 95]

    for idx, z_slice in enumerate(z_slices):
        ax = axes[idx // 3, idx % 3]

        coords_list = []
        for y in range(size):
            for x in range(size):
                coords_list.append([x, y, z_slice])

        coords = torch.tensor(coords_list, dtype=torch.int32, device=device)
        values_out = tree.query(coords)
        active_out = tree.active(coords)

        values_cpu = values_out.cpu().numpy().reshape((size, size))
        active_cpu = active_out.cpu().numpy().reshape((size, size))

        masked = np.ma.masked_where(~active_cpu, values_cpu)
        im = ax.imshow(masked, cmap='viridis', origin='lower',
                      extent=[0, size, 0, size], vmin=threshold, vmax=1.0)

        active_count = np.sum(active_cpu)
        ax.set_title(f'Z={z_slice} ({active_count} voxels)',
                    fontsize=12, weight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(False)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Noise Value')

    fig.suptitle(f'GPU-VDB Procedural Noise ({octaves} octaves, {len(points)} voxels, {sparsity:.1f}% sparse)',
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()

    output_path = Path(__file__).parent / 'output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")


    # Export tree structure
    print("\nExporting tree structure...")
    mesh = tree.export_quad_mesh()
    vertices = mesh['vertices']
    quads = mesh['quads']
    levels = mesh['levels']
    print(f"Tree: {len(vertices)} vertices, {len(quads)} quads")
    
    # 2D visualization with tree overlay
    print("\nGenerating 2D visualization...")
    from mpl_toolkits.mplot3d import Axes3D
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    size = 128
    z_slices = [50, 64, 78]
    
    for idx, z_val in enumerate(z_slices):
        ax = axes[idx]
        coords = torch.tensor([[x, y, z_val] for y in range(size) for x in range(size)],
                             dtype=torch.int32, device=device)
        vals = tree.query(coords).cpu().numpy().reshape(size, size)
        actv = tree.active(coords).cpu().numpy().reshape(size, size)
        
        masked = np.ma.masked_where(~actv, vals)
        ax.imshow(masked, cmap='viridis', origin='lower', extent=[0, size, 0, size])
        
        # Tree overlay
        for level in [2, 1, 0]:
            level_mask = levels == level
            level_quads = quads[level_mask]
            color = ['#ff0000', '#00ff00', '#0000ff'][level]
            alpha = [0.3, 0.4, 0.5][level]
            lw = [0.8, 1.2, 1.5][level]
            for quad in level_quads:
                pts = vertices[quad]
                if pts[:, 2].min() <= z_val + 4 and pts[:, 2].max() >= z_val - 4:
                    x_loop = np.append(pts[:, 0], pts[0, 0])
                    y_loop = np.append(pts[:, 1], pts[0, 1])
                    ax.plot(x_loop, y_loop, color=color, alpha=alpha, linewidth=lw)
        
        ax.set_title(f'Z={z_val}', fontsize=12, weight='bold')
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
    
    fig.suptitle('GPU-VDB Perlin Noise 2D (Blue=Root, Green=Internal, Red=Leaf)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'noise_2d.png', dpi=150, bbox_inches='tight')
    print("Saved noise_2d.png")
    
    # 3D visualization - show actual voxels
    print("\nGenerating 3D visualization...")
    
    # Sample 3D space (every Nth voxel for performance)
    step = 2
    coords_3d = torch.tensor([[x, y, z] for x in range(0, 128, step) 
                               for y in range(0, 128, step) 
                               for z in range(0, 128, step)],
                             dtype=torch.int32, device=device)
    vals_3d = tree.query(coords_3d)
    actv_3d = tree.active(coords_3d)
    
    coords_active = coords_3d[actv_3d].cpu().numpy()
    vals_active = vals_3d[actv_3d].cpu().numpy()
    
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(coords_active[:, 0], coords_active[:, 1], coords_active[:, 2],
                   c=vals_active, cmap='viridis', s=3, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Value', shrink=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'GPU-VDB Perlin Noise 3D ({len(coords_active)} voxels)', fontsize=14, weight='bold')
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 128)

    plt.savefig(Path(__file__).parent / 'noise_3d.png', dpi=150, bbox_inches='tight')
    print("Saved noise_3d.png")


    tree.free()
    return 0

if __name__ == '__main__':
    sys.exit(main())
