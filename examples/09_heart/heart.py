#!/usr/bin/env python3
"""
GPU-VDB Example 09: 3D Heart
Copyright (c) 2025 Muhammad Awad
Licensed under the MIT License

Heart equation (2D): x² + (5y/4 - √|x|)² = 1
Extended to 3D by adding Z depth
"""
import sys
import torch
import gpuvdb
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def main():
    print("GPU-VDB Example 09: 3D Heart")

    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return 1

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tree = gpuvdb.VDBTree(500000, 3000000)
    print("VDB tree initialized")

    print("\nGenerating 3D heart shape...")

    # Sample the 2D heart curve: x² + (5y/4 - √|x|)² = 1
    # Then extrude it along Z axis

    # Dense sampling for smooth heart
    num_samples = 300
    x_range = torch.linspace(-1.2, 1.2, num_samples)
    y_range = torch.linspace(-1.2, 1.5, num_samples)
    z_range = torch.linspace(-0.5, 0.5, 80)  # Depth extrusion

    # Create 3D grid
    xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    zz_flat = zz.flatten()

    print(f"Evaluating heart equation on {len(xx_flat)} points...")

    # Evaluate 2D heart equation (ignoring Z, which is just depth)
    # x² + (5y/4 - √|x|)² ≈ 1
    lhs = xx_flat**2 + ((5.0 * yy_flat / 4.0) - torch.sqrt(torch.abs(xx_flat)))**2

    # Keep points close to the heart curve (with some thickness for 3D volume)
    threshold = 0.05  # Thickness of the heart
    mask = torch.abs(lhs - 1.0) <= threshold

    xx_heart = xx_flat[mask]
    yy_heart = yy_flat[mask]
    zz_heart = zz_flat[mask]

    print(f"Found {len(xx_heart)} points on heart surface")

    if len(xx_heart) == 0:
        print("ERROR: No heart voxels generated")
        return 1

    # Transform to voxel coordinates [0, 127]
    scale = 35.0
    center = 64

    vx = (center + xx_heart * scale).round().to(torch.int32)
    vy = (center + yy_heart * scale).round().to(torch.int32)
    vz = (center + zz_heart * scale).round().to(torch.int32)

    # Filter to valid bounds
    valid_mask = (vx >= 0) & (vx < 128) & (vy >= 0) & (vy < 128) & (vz >= 0) & (vz < 128)
    vx = vx[valid_mask]
    vy = vy[valid_mask]
    vz = vz[valid_mask]

    print(f"After bounds filtering: {len(vx)} valid voxels")

    # Stack coordinates and remove duplicates
    coords = torch.stack([vx, vy, vz], dim=1)
    coords_unique = torch.unique(coords, dim=0)

    print(f"After deduplication: {len(coords_unique)} unique voxels")

    # Color based on height (Y coordinate) - red gradient
    vy_unique = coords_unique[:, 1].float()
    vals = 0.3 + 0.7 * ((vy_unique - vy_unique.min()) / (vy_unique.max() - vy_unique.min()))
    vals = vals.to(torch.float32)

    print(f"Y range: {vy_unique.min().item():.0f} to {vy_unique.max().item():.0f}")

    # Move to GPU
    coords_gpu = coords_unique.to(device)
    vals_gpu = vals.to(device)

    print("Uploading to VDB...")
    tree.insert(coords_gpu, vals_gpu)

    num_internal, num_leaf = tree.get_memory_stats()
    print(f"Heart created. Active voxels: {len(coords_unique)}, Internal nodes: {num_internal}, Leaf nodes: {num_leaf}")

    # Validation - check if inserted points are active
    print("\nValidation:")
    num_test = min(200, len(coords_unique))
    test_indices = torch.randperm(len(coords_unique))[:num_test]
    test_coords = coords_unique[test_indices].to(device)

    test_active = tree.active(test_coords)

    errors = 0
    active_count = test_active.sum().item()
    success_rate = (active_count / num_test) * 100

    if success_rate < 95.0:
        print(f"  FAIL: Only {active_count}/{num_test} points are active ({success_rate:.1f}%)")
        errors += 1
    else:
        print(f"  All tests passed ({active_count}/{num_test} points active, {success_rate:.1f}%)")

    # Export tree structure
    print("\nExporting VDB tree structure...")
    mesh = tree.export_quad_mesh()
    print(f"Tree mesh: {len(mesh['vertices'])} vertices, {len(mesh['quads'])} quads")
    print(f"  Level 0 (leaf): {(mesh['levels'] == 0).sum()} faces")
    print(f"  Level 1 (internal): {(mesh['levels'] == 1).sum()} faces")
    print(f"  Level 2 (root): {(mesh['levels'] == 2).sum()} faces")

    # Visualize 2D slices with tree structure overlay
    print("\nGenerating heart slice visualization with tree overlay...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    size = 128
    # Sample Z slices through the heart's depth
    vz_unique = coords_unique[:, 2]
    z_min = int(vz_unique.min().item())
    z_max = int(vz_unique.max().item())
    z_center = (z_min + z_max) // 2
    z_slices = [z_center - 8, z_center - 4, z_center, z_center + 2, z_center + 4, z_center + 8]

    # Extract tree structure for overlay
    vertices = mesh['vertices']
    quads = mesh['quads']
    levels = mesh['levels']

    for idx, z_slice in enumerate(z_slices):
        ax = axes[idx // 3, idx % 3]

        y_coords = torch.arange(size, device=device)
        x_coords = torch.arange(size, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        coords_list = torch.stack([
            xx.flatten(),
            yy.flatten(),
            torch.full((size * size,), z_slice, device=device)
        ], dim=1).to(torch.int32)

        values_out = tree.query(coords_list)
        active_out = tree.active(coords_list)

        values_cpu = values_out.cpu().reshape(size, size)
        active_cpu = active_out.cpu().reshape(size, size)

        masked = np.ma.masked_where(~active_cpu.numpy(), values_cpu.numpy())

        im = ax.imshow(masked, cmap='Reds', origin='lower',
                      extent=[0, size, 0, size], vmin=0.3, vmax=1.0,
                      interpolation='bilinear')

        # Overlay tree structure (quads that intersect this Z slice)
        for level in [2, 1, 0]:
            level_mask = levels == level
            level_quads = quads[level_mask]

            # Color for each level
            if level == 0:
                color, alpha, lw = '#ff0000', 0.3, 0.8  # Leaf - red
            elif level == 1:
                color, alpha, lw = '#00ff00', 0.4, 1.2  # Internal - green
            else:
                color, alpha, lw = '#0000ff', 0.5, 1.5  # Root - blue

            for quad in level_quads[::1]:  # Draw all quads
                pts = vertices[quad]
                # Check if this quad intersects the Z slice (within ±4 voxels)
                z_coords = pts[:, 2]
                if z_coords.min() <= z_slice + 4 and z_coords.max() >= z_slice - 4:
                    # Draw quad edges in X-Y plane
                    x_coords = pts[:, 0]
                    y_coords = pts[:, 1]
                    # Close the loop
                    x_loop = np.append(x_coords, x_coords[0])
                    y_loop = np.append(y_coords, y_coords[0])
                    ax.plot(x_loop, y_loop, color=color, alpha=alpha, linewidth=lw)

        active_count = active_out.sum().item()
        ax.set_title(f'Z={z_slice} ({active_count} voxels)',
                    fontsize=12, weight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.grid(False)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity')

    fig.suptitle(f'GPU-VDB 3D Heart with Tree Structure\n({len(coords_unique)} voxels, Blue=Root, Green=Internal, Red=Leaf)',
                 fontsize=14, weight='bold', y=0.995, color='#FF1744')
    plt.tight_layout()

    output_path = Path(__file__).parent / 'heart_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heart_2d.png")

    # Visualize 3D heart voxels
    print("\nGenerating 3D heart visualization...")
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Sample every Nth voxel for performance
    sample_rate = max(1, len(coords_unique) // 5000)  # ~5000 points max
    coords_sample = coords_unique[::sample_rate].cpu().numpy()
    vals_sample = vals[::sample_rate].cpu().numpy()

    # Color based on height (Y coordinate)
    ax.scatter(coords_sample[:, 0], coords_sample[:, 1], coords_sample[:, 2],
              c=vals_sample, cmap='Reds', s=2, alpha=0.6, vmin=0.3, vmax=1.0)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('GPU-VDB 3D Heart Point Cloud\n(Sampled voxels colored by height)',
                fontsize=14, weight='bold')
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 128)
    ax.view_init(elev=20, azim=45)

    heart_3d_path = Path(__file__).parent / 'heart_3d.png'
    plt.savefig(heart_3d_path, dpi=150, bbox_inches='tight')
    print(f"Saved heart_3d.png")

    tree.free()
    return errors


if __name__ == '__main__':
    sys.exit(main())
