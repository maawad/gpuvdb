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
        [center_x, center_y, center_z],  # center
        [center_x + 10, center_y, center_z],  # inside
        [center_x + 25, center_y, center_z],  # outside
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
    else:
        print(f"  {errors} tests failed")
        tree.free()
        return errors

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im1 = ax.imshow(active_cpu, cmap='viridis', origin='lower', extent=[0, size, 0, size])
    ax.set_title(f'Active Voxels (z={center_z} slice)', fontsize=14, weight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    circle = plt.Circle((center_x, center_y), radius, fill=False,
                       color='red', linewidth=2, linestyle='--', label='Expected')
    ax.add_patch(circle)
    ax.legend()
    plt.colorbar(im1, ax=ax, label='Active')

    ax = axes[1]
    im2 = ax.imshow(values_cpu, cmap='plasma', origin='lower', extent=[0, size, 0, size])
    ax.set_title(f'Voxel Values (z={center_z} slice)', fontsize=14, weight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    circle = plt.Circle((center_x, center_y), radius, fill=False,
                       color='cyan', linewidth=2, linestyle='--', label='Expected')
    ax.add_patch(circle)
    ax.legend()
    plt.colorbar(im2, ax=ax, label='Value')

    fig.suptitle(f'GPU-VDB Simple Sphere (r={radius}, {active_count} voxels, {accuracy:.1f}% accurate)',
                 fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()

    print("\nGenerating visualization...")
    output_path = Path(__file__).parent / 'output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")

    tree.free()
    return 0

if __name__ == '__main__':
    sys.exit(main())
