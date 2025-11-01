#!/usr/bin/env python3
"""
GPU-VDB Example 02: Multi Spheres
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

def main():
    print("GPU-VDB Example 02: Multi Spheres")

    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return 1

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tree = gpuvdb.VDBTree()
    print("VDB tree initialized")

    # Create multiple spheres with different values
    spheres = [
        (30, 30, 50, 15.0, 1.0),
        (50, 50, 50, 20.0, 5.0),
        (70, 70, 50, 12.0, 9.0),
        (50, 30, 50, 10.0, 3.0),
        (30, 70, 50, 10.0, 7.0),
    ]

    print(f"Filling {len(spheres)} spheres")
    for i, (cx, cy, cz, r, v) in enumerate(spheres, 1):
        print(f"  Sphere {i}: center=({cx},{cy},{cz}), r={r:.1f}, value={v:.1f}")
        tree.fill_sphere(cx, cy, cz, r, v)

    num_internal, num_leaf = tree.get_memory_stats()
    print(f"Internal nodes: {num_internal}, Leaf nodes: {num_leaf}")

    # Validation
    print("\nValidation:")
    errors = 0
    for i, (cx, cy, cz, r, expected_val) in enumerate(spheres, 1):
        test_coord = torch.tensor([[cx, cy, cz]], dtype=torch.int32, device=device)
        queried_val = tree.query(test_coord)[0].item()
        is_active = tree.active(test_coord)[0].item()

        if is_active != 1 or abs(queried_val - expected_val) > 0.01:
            print(f"  FAIL: Sphere {i} center")
            errors += 1

    # Test point far from all spheres
    test_coord = torch.tensor([[5, 5, 50]], dtype=torch.int32, device=device)
    if tree.active(test_coord)[0].item() != 0:
        print("  FAIL: Far outside should be inactive")
        errors += 1

    if errors == 0:
        print("  All tests passed")
    else:
        print(f"  {errors} tests failed")
        tree.free()
        return 1

    # Visualize
    z_slice = 50
    print(f"\nQuerying slice at z={z_slice}")

    size = 100
    coords_list = []
    for y in range(size):
        for x in range(size):
            coords_list.append([x, y, z_slice])

    coords = torch.tensor(coords_list, dtype=torch.int32, device=device)
    values_out = tree.query(coords)
    active_out = tree.active(coords)

    values_cpu = values_out.cpu().numpy().reshape((size, size))
    active_cpu = active_out.cpu().numpy().reshape((size, size))

    active_count = np.sum(active_cpu)
    print(f"Active voxels: {active_count}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    im1 = ax.imshow(active_cpu, cmap='binary', origin='lower', extent=[0, size, 0, size])
    ax.set_title(f'Active Voxels (z={z_slice} slice)', fontsize=14, weight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for (cx, cy, cz, r, v), color in zip(spheres, colors):
        circle = plt.Circle((cx, cy), r, fill=False,
                           color=color, linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(circle)
        ax.text(cx, cy, f'${v:.1f}$', ha='center', va='center',
               fontsize=10, weight='bold', color=color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.colorbar(im1, ax=ax, label='Active')

    ax = axes[1]
    masked = np.ma.masked_where(~active_cpu, values_cpu)
    im2 = ax.imshow(masked, cmap='hot', origin='lower', extent=[0, size, 0, size],
                   vmin=0, vmax=10)
    ax.set_title(f'Voxel Values (z={z_slice} slice)', fontsize=14, weight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    for (cx, cy, cz, r, v), color in zip(spheres, colors):
        circle = plt.Circle((cx, cy), r, fill=False,
                           color='cyan', linewidth=1.5, linestyle='--', alpha=0.5)
        ax.add_patch(circle)
    plt.colorbar(im2, ax=ax, label='Value')

    ax = axes[2]
    im3 = ax.imshow(masked, cmap='tab10', origin='lower', extent=[0, size, 0, size],
                   vmin=0, vmax=10)
    ax.set_title(f'Discrete Regions', fontsize=14, weight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    for (cx, cy, cz, r, v), color in zip(spheres, colors):
        circle = plt.Circle((cx, cy), r, fill=False,
                           color='white', linewidth=1, linestyle=':', alpha=0.6)
        ax.add_patch(circle)
    plt.colorbar(im3, ax=ax, label='Value', ticks=range(10))

    fig.suptitle(f'GPU-VDB Multi-Spheres ({len(spheres)} spheres, {active_count} total voxels)',
                 fontsize=16, weight='bold', y=0.98)
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
    z_slices = [30, 50, 70]
    
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
    
    fig.suptitle('GPU-VDB Multi-Spheres 2D (Blue=Root, Green=Internal, Red=Leaf)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'multisphere_2d.png', dpi=150, bbox_inches='tight')
    print("Saved multisphere_2d.png")
    
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
                   c=vals_active, cmap='viridis', s=3, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Value', shrink=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'GPU-VDB Multi-Spheres 3D ({len(coords_active)} voxels)', fontsize=14, weight='bold')
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 128)

    plt.savefig(Path(__file__).parent / 'multisphere_3d.png', dpi=150, bbox_inches='tight')
    print("Saved multisphere_3d.png")


    tree.free()
    return 0

if __name__ == '__main__':
    sys.exit(main())
