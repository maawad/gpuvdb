#!/usr/bin/env python3
"""
GPU-VDB Example 07: Conway's Game of Life in 3D
Copyright (c) 2025 Muhammad Awad
Licensed under the MIT License
"""
import torch
import gpuvdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def count_neighbors_3d(grid, x, y, z):
    """Count alive neighbors in 3D (26-neighborhood)"""
    count = 0
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx, ny, nz = x+dx, y+dy, z+dz
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2]:
                    if grid[nx, ny, nz]:
                        count += 1
    return count

def conway_step_3d(grid):
    """One step of 3D Conway's Game of Life (custom rules)"""
    new_grid = np.zeros_like(grid)
    for z in range(grid.shape[2]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                neighbors = count_neighbors_3d(grid, x, y, z)
                # 3D Conway rules (adjusted for 26-neighborhood, more permissive)
                if grid[x, y, z]:
                    # Alive cell survives with 5-10 neighbors
                    if 5 <= neighbors <= 10:
                        new_grid[x, y, z] = 1
                else:
                    # Dead cell becomes alive with 6-8 neighbors
                    if 6 <= neighbors <= 8:
                        new_grid[x, y, z] = 1
    return new_grid

def main():
    print("GPU-VDB Example 07: 3D Conway's Game of Life")

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return 1

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tree = gpuvdb.VDBTree()
    print("VDB tree initialized")

    # Create initial pattern (glider-like structure in 3D)
    size = 80
    grid = np.zeros((size, size, size), dtype=bool)

    # Seed with stable 3D pattern (multiple cubes and structures)
    center = size // 2

    # Create a 3x3x3 cube
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                x, y, z = center + dx, center + dy, center + dz
                grid[x, y, z] = 1

    # Add satellite patterns around it
    offsets = [(10, 0, 0), (-10, 0, 0), (0, 10, 0), (0, -10, 0), (0, 0, 10), (0, 0, -10)]
    for ox, oy, oz in offsets:
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 or dy == 0 or dz == 0:  # Make it a cross shape
                        x, y, z = center + ox + dx, center + oy + dy, center + oz + dz
                        if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                            grid[x, y, z] = 1

    print(f"\nSimulating 3D cellular automaton...")
    print(f"  Initial alive cells: {np.sum(grid)}")

    # Run simulation for multiple steps
    steps = 10
    snapshots = []

    for step in range(steps):
        grid = conway_step_3d(grid)
        alive = np.sum(grid)
        print(f"  Step {step+1}: {alive} alive cells")
        if step % 2 == 0:  # Save every 2 steps
            snapshots.append(grid.copy())

    # Store final state in VDB
    points = []
    values = []

    for z in range(size):
        for y in range(size):
            for x in range(size):
                if grid[x, y, z]:
                    points.append([x, y, z])
                    # Color by distance from center for visual interest
                    dist = np.sqrt((x-center)**2 + (y-center)**2 + (z-center)**2)
                    value = 1.0 - min(dist / (size * 0.5), 1.0)
                    values.append(value)

    print(f"\nFinal state: {len(points)} alive cells")

    failed = 0
    # Upload to VDB
    if len(points) > 0:
        coords = torch.tensor(points, dtype=torch.int32, device=device)
        vals = torch.tensor(values, dtype=torch.float32, device=device)

        print(f"  Uploading to VDB...")
        tree.insert(coords, vals)

        num_internal, num_leaf = tree.get_memory_stats()
        print(f"Cellular automaton state stored")
        print(f"  Internal nodes: {num_internal}")
        print(f"  Leaf nodes: {num_leaf}")

        # Validation: Re-query a subset of points
        print(f"\nValidation:")
        num_test = min(100, len(points))
        test_indices = np.random.choice(len(points), num_test, replace=False)
        test_coords_cpu = np.array(points)[test_indices]
        test_values_expected = np.array(values)[test_indices]

        test_coords = torch.tensor(test_coords_cpu, dtype=torch.int32, device=device)
        test_values_queried = tree.query(test_coords).cpu().numpy()
        test_active_queried = tree.active(test_coords).cpu().numpy()

        if not np.all(test_active_queried == 1):
            print("  Active check failed")
            failed = 1

        if not np.allclose(test_values_queried, test_values_expected, atol=0.01):
            print("  Value check failed")
            failed = 1

        if failed == 0:
            print("  All tests passed")

    # Visualize evolution
    print(f"\nGenerating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Show 3 snapshots at middle slice
    z_slice = center

    for idx in range(6):
        ax = axes[idx // 3, idx % 3]

        if idx < len(snapshots):
            # Show snapshot
            snapshot = snapshots[idx]
            slice_2d = snapshot[:, :, z_slice].T

            im = ax.imshow(slice_2d, cmap='plasma', origin='lower',
                          extent=[0, size, 0, size], vmin=0, vmax=1)
            ax.set_title(f'Step {idx*2} (z={z_slice})', fontsize=12, weight='bold')
        else:
            # Query final state from VDB
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
            im = ax.imshow(masked, cmap='plasma', origin='lower',
                          extent=[0, size, 0, size], vmin=0, vmax=1)
            ax.set_title(f'Final (z={z_slice}, from VDB)', fontsize=12, weight='bold')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(False)
        plt.colorbar(im, ax=ax, label='Intensity')

    fig.suptitle(f'GPU-VDB 3D Cellular Automaton ({len(points)} alive cells after {steps} steps)',
                 fontsize=16, weight='bold', y=0.995)

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent / 'output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")

    # Cleanup
    tree.free()

    return failed

if __name__ == '__main__':
    import sys
    sys.exit(main())
