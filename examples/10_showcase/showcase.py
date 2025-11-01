#!/usr/bin/env python3
"""
GPU-VDB Example 10: Showcase
Copyright (c) 2025 Muhammad Awad
Licensed under the MIT License

A visually striking combination of shapes demonstrating GPU-VDB capabilities.
"""
import sys
import torch
import gpuvdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def heart_coords(center_x, center_y, center_z, scale=1.0):
    """Generate filled heart shape coordinates (interior + surface)"""
    num_samples = 300
    x_range = torch.linspace(-1.2, 1.2, num_samples)
    y_range = torch.linspace(-1.2, 1.5, num_samples)
    z_range = torch.linspace(-0.5, 0.5, 60)

    xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    zz_flat = zz.flatten()

    # Heart equation: x² + (5y/4 - √|x|)² <= 1 (FILLED interior)
    lhs = xx_flat**2 + ((5.0 * yy_flat / 4.0) - torch.sqrt(torch.abs(xx_flat)))**2
    mask = lhs <= 1.0  # Everything INSIDE the heart

    xx_heart = xx_flat[mask]
    yy_heart = yy_flat[mask]
    zz_heart = zz_flat[mask]

    # Transform to voxel coordinates
    vx = (center_x + xx_heart * scale).round().to(torch.int32)
    vy = (center_y + yy_heart * scale).round().to(torch.int32)
    vz = (center_z + zz_heart * scale).round().to(torch.int32)

    # Filter bounds
    valid = (vx >= 0) & (vx < 128) & (vy >= 0) & (vy < 128) & (vz >= 0) & (vz < 128)
    coords = torch.stack([vx[valid], vy[valid], vz[valid]], dim=1)
    coords_unique = torch.unique(coords, dim=0)

    return coords_unique

def main():
    print("GPU-VDB Example 10: Showcase")

    if not torch.cuda.is_available():
        print("ERROR: GPU not available")
        return 1

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tree = gpuvdb.VDBTree(500000, 3000000)
    print("VDB tree initialized")

    print("\nGenerating showcase scene...")

    # Big filled heart in the center
    print("  - Heart (filled interior)...")
    heart = heart_coords(64, 70, 64, scale=25.0)

    # Color gradient based on Y coordinate (top = bright, bottom = dark)
    vy = heart[:, 1].float()
    heart_vals = 0.5 + 0.5 * ((vy - vy.min()) / (vy.max() - vy.min() + 1e-6))
    heart_vals = heart_vals.to(torch.float32)

    # Insert heart (no extra spheres - just the heart!)
    if len(heart) > 0:
        tree.insert(heart.to(device), heart_vals.to(device))

    num_internal, num_leaf = tree.get_memory_stats()
    print(f"\nScene created:")
    print(f"  Heart voxels: {len(heart)}")
    print(f"  Internal nodes: {num_internal}")
    print(f"  Leaf nodes: {num_leaf}")

    # Validation
    print("\nValidation:")
    test_coords = torch.tensor([
        [64, 70, 64],   # Heart center
        [60, 80, 64],   # Top of heart
        [70, 60, 64],   # Bottom of heart
    ], dtype=torch.int32, device=device)

    test_active = tree.active(test_coords)
    errors = 0
    if test_active.sum().item() < 2:  # At least 2 should be active
        print("  FAIL: Not enough test points active")
        errors += 1
    else:
        print("  All tests passed")

    # Export tree
    print("\nExporting tree structure...")
    mesh = tree.export_quad_mesh()
    vertices = mesh['vertices']
    quads = mesh['quads']
    levels = mesh['levels']

    # 2D visualization - slices through the heart
    print("\nGenerating 2D visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    size = 128
    z_slices = [54, 64, 74]  # Through the heart's depth

    for idx, z_val in enumerate(z_slices):
        ax = axes[idx]
        coords = torch.tensor([[x, y, z_val] for y in range(size) for x in range(size)],
                             dtype=torch.int32, device=device)
        vals = tree.query(coords).cpu().numpy().reshape(size, size)
        actv = tree.active(coords).cpu().numpy().reshape(size, size)

        masked = np.ma.masked_where(~actv, vals)
        im = ax.imshow(masked, cmap='Reds', origin='lower', extent=[0, size, 0, size],
                      vmin=0.4, vmax=1.0, interpolation='bilinear')

        # Tree overlay - show ALL tree structure projected onto this slice
        for level in [2, 1, 0]:
            level_mask = levels == level
            level_quads = quads[level_mask]
            if level == 0:
                color, alpha, lw = '#ff0000', 0.2, 0.5  # Leaf - red
            elif level == 1:
                color, alpha, lw = '#00ff00', 0.3, 0.8  # Internal - green
            else:
                color, alpha, lw = '#0000ff', 0.4, 1.2   # Root - blue

            # Draw all quads (project entire tree onto 2D slice)
            for quad in level_quads:
                pts = vertices[quad]
                # Only draw if the quad has X-Y coordinates (not purely Z-aligned faces)
                # This filters out vertical faces and shows horizontal structure
                x_range = pts[:, 0].max() - pts[:, 0].min()
                y_range = pts[:, 1].max() - pts[:, 1].min()

                if x_range > 1 or y_range > 1:  # Has X or Y extent
                    x_loop = np.append(pts[:, 0], pts[0, 0])
                    y_loop = np.append(pts[:, 1], pts[0, 1])
                    ax.plot(x_loop, y_loop, color=color, alpha=alpha, linewidth=lw)

        active_count = actv.sum()
        ax.set_title(f'Z={z_val} ({active_count} voxels)', fontsize=14, weight='bold')
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        plt.colorbar(im, ax=ax, label='Intensity')

    fig.suptitle(f'GPU-VDB Showcase: Filled Heart ({len(heart)} voxels)\\nBlue=Root, Green=Internal, Red=Leaf',
                 fontsize=16, weight='bold', color='#FF1744')
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'showcase_2d.png', dpi=150, bbox_inches='tight')
    print("Saved showcase_2d.png")

    # 3D visualization
    print("\nGenerating 3D visualization...")
    step = 2
    coords_3d = torch.tensor([[x, y, z] for x in range(0, 128, step)
                               for y in range(0, 128, step)
                               for z in range(0, 128, step)],
                             dtype=torch.int32, device=device)
    vals_3d = tree.query(coords_3d)
    actv_3d = tree.active(coords_3d)

    coords_active = coords_3d[actv_3d].cpu().numpy()
    vals_active = vals_3d[actv_3d].cpu().numpy()

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the heart with nice gradient
    sc = ax.scatter(coords_active[:, 0], coords_active[:, 1], coords_active[:, 2],
                   c=vals_active, cmap='Reds', s=3, alpha=0.7, vmin=0.4, vmax=1.0)

    cbar = plt.colorbar(sc, ax=ax, label='Height Gradient', shrink=0.5, pad=0.1)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('X', fontsize=13, labelpad=10)
    ax.set_ylabel('Y', fontsize=13, labelpad=10)
    ax.set_zlabel('Z', fontsize=13, labelpad=10)
    ax.set_title(f'GPU-VDB Showcase: Filled Heart\\n({len(coords_active)} sampled voxels, {len(heart)} total)',
                fontsize=17, weight='bold', color='#FF1744', pad=20)
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 128)

    # Better viewing angle - rotated to show heart shape
    ax.view_init(elev=20, azim=135)

    # Grid styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

    plt.savefig(Path(__file__).parent / 'showcase_3d.png', dpi=150, bbox_inches='tight')
    print("Saved showcase_3d.png")

    tree.free()
    return errors

if __name__ == '__main__':
    sys.exit(main())

