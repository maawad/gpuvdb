#!/usr/bin/env python3
"""
GPU-VDB Example 01: Simple Sphere
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
    print("GPU-VDB Example 01: Simple Sphere")

    if not torch.cuda.is_available():
        print("ERROR: GPU not available")
        return 1

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tree = gpuvdb.VDBTree()
    print("VDB tree initialized")

    # Create sphere
    center_x, center_y, center_z = 50, 50, 50
    radius = 20.0
    value = 1.0

    print(f"\nFilling sphere at ({center_x}, {center_y}, {center_z}), radius={radius}")
    tree.fill_sphere(center_x, center_y, center_z, radius, value)

    num_internal, num_leaf = tree.get_memory_stats()
    print(f"Internal nodes: {num_internal}, Leaf nodes: {num_leaf}")

    # Query 2D slice
    size = 100
    coords_list = [[x, y, center_z] for y in range(size) for x in range(size)]
    coords = torch.tensor(coords_list, dtype=torch.int32, device=device)

    values_out = tree.query(coords)
    active_out = tree.active(coords)

    values_cpu = values_out.cpu().numpy().reshape((size, size))
    active_cpu = active_out.cpu().numpy().reshape((size, size))

    active_count = np.sum(active_cpu)
    expected_count = np.pi * radius**2
    accuracy = (active_count / expected_count) * 100
    print(f"Active voxels: {active_count}, Expected: {expected_count:.1f}, Accuracy: {accuracy:.2f}%")

    # Validation
    print("\nValidation:")
    test_coords = torch.tensor([
        [center_x, center_y, center_z],
        [center_x + 10, center_y, center_z],
        [center_x + 25, center_y, center_z],
    ], dtype=torch.int32, device=device)

    test_values = tree.query(test_coords)
    test_active = tree.active(test_coords)

    errors = 0
    if test_active[0].item() != 1 or abs(test_values[0].item() - 1.0) > 0.01:
        print("  FAIL: center query failed")
        errors += 1
    if test_active[1].item() != 1 or abs(test_values[1].item() - 1.0) > 0.01:
        print("  FAIL: inside query failed")
        errors += 1
    if test_active[2].item() != 0:
        print("  FAIL: outside query failed")
        errors += 1

    if errors == 0:
        print("  All tests passed")

    # Export tree structure
    print("\nExporting tree structure...")
    mesh = tree.export_quad_mesh()
    vertices = mesh['vertices']
    quads = mesh['quads']
    levels = mesh['levels']
    print(f"Tree: {len(vertices)} vertices, {len(quads)} quads")

    # 2D visualization with tree overlay
    print("\nGenerating 2D visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for subplot_idx, z_slice in enumerate([center_z - 10, center_z]):
        ax = axes[subplot_idx]

        # Query slice
        coords_slice = torch.tensor([[x, y, z_slice] for y in range(size) for x in range(size)],
                                    dtype=torch.int32, device=device)
        values_slice = tree.query(coords_slice).cpu().numpy().reshape(size, size)
        active_slice = tree.active(coords_slice).cpu().numpy().reshape(size, size)

        masked = np.ma.masked_where(~active_slice, values_slice)
        ax.imshow(masked, cmap='viridis', origin='lower', extent=[0, size, 0, size])

        # Overlay tree structure
        for level in [2, 1, 0]:
            level_mask = levels == level
            level_quads = quads[level_mask]
            color = ['#ff0000', '#00ff00', '#0000ff'][level]
            alpha = [0.3, 0.4, 0.5][level]
            lw = [0.8, 1.2, 1.5][level]

            for quad in level_quads:
                pts = vertices[quad]
                z_coords = pts[:, 2]
                if z_coords.min() <= z_slice + 4 and z_coords.max() >= z_slice - 4:
                    x_loop = np.append(pts[:, 0], pts[0, 0])
                    y_loop = np.append(pts[:, 1], pts[0, 1])
                    ax.plot(x_loop, y_loop, color=color, alpha=alpha, linewidth=lw)

        ax.set_title(f'Z={z_slice} slice', fontsize=12, weight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)

    fig.suptitle(f'GPU-VDB Sphere 2D (r={radius}, Blue=Root, Green=Internal, Red=Leaf)',
                 fontsize=14, weight='bold')
    plt.tight_layout()

    output_2d = Path(__file__).parent / 'sphere_2d.png'
    plt.savefig(output_2d, dpi=150, bbox_inches='tight')
    print(f"Saved 2D to: {output_2d}")

    # 3D visualization - show actual voxels
    print("\nGenerating 3D visualization...")

    # Sample 3D space
    coords_3d = torch.tensor([[x, y, z] for x in range(100) for y in range(100) for z in range(100)],
                             dtype=torch.int32, device=device)
    vals_3d = tree.query(coords_3d)
    actv_3d = tree.active(coords_3d)

    coords_active = coords_3d[actv_3d].cpu().numpy()
    vals_active = vals_3d[actv_3d].cpu().numpy()

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(coords_active[:, 0], coords_active[:, 1], coords_active[:, 2],
                   c=vals_active, cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Value', shrink=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'GPU-VDB Sphere 3D ({len(coords_active)} voxels)', fontsize=14, weight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)

    output_3d = Path(__file__).parent / 'sphere_3d.png'
    plt.savefig(output_3d, dpi=150, bbox_inches='tight')
    print(f"Saved 3D to: {output_3d}")

    tree.free()
    return errors

if __name__ == '__main__':
    sys.exit(main())
