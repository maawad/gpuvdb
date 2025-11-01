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
from mpl_toolkits.mplot3d import Axes3D

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
    z_slices = [30, 40, 50]
    
    for idx, z_val in enumerate(z_slices):
        ax = axes[idx]
        coords = torch.tensor([[x, y, z_val] for y in range(size) for x in range(size)],
                             dtype=torch.int32, device=device)
        vals = tree.query(coords).cpu().numpy().reshape(size, size)
        actv = tree.active(coords).cpu().numpy().reshape(size, size)
        
        masked = np.ma.masked_where(~actv, vals)
        ax.imshow(masked, cmap='binary', origin='lower', extent=[0, size, 0, size])
        
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
    
    fig.suptitle('GPU-VDB Conway 3D 2D (Blue=Root, Green=Internal, Red=Leaf)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'conway_2d.png', dpi=150, bbox_inches='tight')
    print("Saved conway_2d.png")
    
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
                   c=vals_active, cmap='binary', s=3, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Value', shrink=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'GPU-VDB Conway 3D 3D ({len(coords_active)} voxels)', fontsize=14, weight='bold')
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 128)

    plt.savefig(Path(__file__).parent / 'conway_3d.png', dpi=150, bbox_inches='tight')
    print("Saved conway_3d.png")


    tree.free()

    return failed

if __name__ == '__main__':
    import sys
    sys.exit(main())
