#!/usr/bin/env python3
"""
GPU-VDB Example 06: Metaballs
Copyright (c) 2025 Muhammad Awad
Licensed under the MIT License
"""
import sys
import torch
import gpuvdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def metaball_field(x, y, z, balls):
    """Compute metaball field (sum of inverse squared distances)"""
    field = 0.0
    for bx, by, bz, radius in balls:
        dx = x - bx
        dy = y - by
        dz = z - bz
        dist_sq = dx*dx + dy*dy + dz*dz + 0.01
        field += (radius * radius) / dist_sq
    return field

def main():
    print("GPU-VDB Example 06: Metaballs (Organic Blobs)")

    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return 1

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tree = gpuvdb.VDBTree(200000, 2000000)
    print("VDB tree initialized")

    metaballs = [
        (40, 40, 50, 15),
        (60, 60, 50, 12),
        (50, 50, 50, 18),
        (35, 65, 50, 10),
        (65, 35, 50, 10),
    ]

    print(f"Generating {len(metaballs)} metaballs")
    size = 100
    threshold = 1.0

    points = []
    values = []

    for z in range(size):
        for y in range(size):
            for x in range(size):
                field = metaball_field(x, y, z, metaballs)

                if field > threshold:
                    points.append([x, y, z])
                    value = min(field / 3.0, 1.0)
                    values.append(value)

    print(f"Found {len(points)} voxels in metaball field")

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

    # Check if values are reasonable (above threshold, not too high)
    out_of_range = np.sum((test_values_queried < 0) | (test_values_queried > 1.5))
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

    z_slices = [35, 40, 45, 50, 55, 60]

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
        im = ax.imshow(masked, cmap='magma', origin='lower',
                      extent=[0, size, 0, size], vmin=0, vmax=1,
                      interpolation='bilinear')

        for bx, by, bz, radius in metaballs:
            if abs(bz - z_slice) < 5:
                alpha = 1.0 - abs(bz - z_slice) / 5.0
                ax.plot(bx, by, 'o', color='cyan', markersize=8,
                       alpha=alpha, markeredgecolor='white', markeredgewidth=2)

        active_count = np.sum(active_cpu)
        ax.set_title(f'Z={z_slice} ({active_count} voxels)',
                    fontsize=12, weight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(False)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Field Strength')

    fig.suptitle(f'GPU-VDB Metaballs ({len(metaballs)} organic blobs, {len(points)} voxels)',
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()

    output_path = Path(__file__).parent / 'output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")

    tree.free()
    return 0

if __name__ == '__main__':
    sys.exit(main())
