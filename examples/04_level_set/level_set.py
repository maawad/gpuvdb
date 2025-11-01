#!/usr/bin/env python3
"""
GPU-VDB Example 04: Level Set
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

def sphere_sdf(x, y, z, cx, cy, cz, radius):
    """Signed distance to sphere"""
    dx = x - cx
    dy = y - cy
    dz = z - cz
    return np.sqrt(dx*dx + dy*dy + dz*dz) - radius

def box_sdf(x, y, z, cx, cy, cz, half_size):
    """Signed distance to box"""
    dx = np.abs(x - cx) - half_size
    dy = np.abs(y - cy) - half_size
    dz = np.abs(z - cz) - half_size

    outside = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2 + np.maximum(dz, 0)**2)
    inside = np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0)

    return outside + inside

def torus_sdf(x, y, z, cx, cy, cz, major_r, minor_r):
    """Signed distance to torus"""
    dx = x - cx
    dy = y - cy
    dz = z - cz

    q_x = np.sqrt(dx*dx + dz*dz) - major_r
    return np.sqrt(q_x*q_x + dy*dy) - minor_r

def union_sdf(*sdfs):
    """Union of multiple SDFs (minimum)"""
    return np.minimum.reduce(sdfs)

def main():
    print("GPU-VDB Example 04: Level Set (SDF)")

    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return 1

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tree = gpuvdb.VDBTree()
    print("VDB tree initialized")

    print("Generating composite SDF")
    size = 100
    bandwidth = 3.0

    points = []
    values = []

    for z in range(size):
        for y in range(size):
            for x in range(size):
                sdf1 = sphere_sdf(x, y, z, 30, 30, 50, 15)
                sdf2 = box_sdf(x, y, z, 50, 50, 50, 12)
                sdf3 = torus_sdf(x, y, z, 70, 70, 50, 12, 5)

                sdf = union_sdf(sdf1, sdf2, sdf3)

                if abs(sdf) < bandwidth:
                    points.append([x, y, z])
                    value = (sdf + bandwidth) / (2 * bandwidth)
                    values.append(value)

    print(f"Found {len(points)} voxels in narrow band")

    if len(points) == 0:
        print("ERROR: No points generated")
        return 1

    coords = torch.tensor(points, dtype=torch.int32, device=device)
    vals = torch.tensor(values, dtype=torch.float32, device=device)

    tree.insert(coords, vals)

    num_internal, num_leaf = tree.get_memory_stats()
    print(f"Internal nodes: {num_internal}, Leaf nodes: {num_leaf}")

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

    # Check if values are in reasonable range for SDF (0-1)
    out_of_range = np.sum((test_values_queried < -0.1) | (test_values_queried > 1.1))
    if out_of_range > num_test * 0.1:
        print(f"  FAIL: {out_of_range}/{num_test} values are out of range")
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

    z_slices = [40, 45, 50, 55, 60, 65]

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
        im = ax.imshow(masked, cmap='RdBu_r', origin='lower',
                      extent=[0, size, 0, size], vmin=0, vmax=1)

        contour = ax.contour(values_cpu, levels=[0.5], colors='yellow',
                            linewidths=2, extent=[0, size, 0, size],
                            origin='lower')

        ax.set_title(f'Z={z_slice} ({np.sum(active_cpu)} voxels)',
                    fontsize=12, weight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(False)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SDF (0.5=surface)')
        cbar.ax.axhline(0.5, color='yellow', linewidth=2, linestyle='--')

    fig.suptitle(f'GPU-VDB Level Set / SDF (Sphere U Box U Torus, {len(points)} voxels)',
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
        ax.imshow(masked, cmap='coolwarm', origin='lower', extent=[0, size, 0, size])
        
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
    
    fig.suptitle('GPU-VDB Level Set 2D (Blue=Root, Green=Internal, Red=Leaf)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'levelset_2d.png', dpi=150, bbox_inches='tight')
    print("Saved levelset_2d.png")
    
    # 3D visualization - show actual voxels
    print("\nGenerating 3D visualization...")
    
    # Sample 3D space (every Nth voxel for performance)
    step = 1
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
                   c=vals_active, cmap='coolwarm', s=3, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Value', shrink=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'GPU-VDB Level Set 3D ({len(coords_active)} voxels)', fontsize=14, weight='bold')
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 128)

    plt.savefig(Path(__file__).parent / 'levelset_3d.png', dpi=150, bbox_inches='tight')
    print("Saved levelset_3d.png")


    tree.free()
    return 0

if __name__ == '__main__':
    sys.exit(main())
