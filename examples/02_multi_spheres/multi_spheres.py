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

    tree.free()
    return 0

if __name__ == '__main__':
    sys.exit(main())
