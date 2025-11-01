#!/usr/bin/env python3
"""
GPU-VDB Example 08: Voxel Art (Procedural Skull)
Copyright (c) 2025 Muhammad Awad
Licensed under the MIT License
"""
import torch
import gpuvdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def add_sphere(voxels, cx, cy, cz, radius, value):
    """Add a filled sphere to voxel array"""
    for z in range(max(0, int(cz-radius)), min(voxels.shape[2], int(cz+radius+1))):
        for y in range(max(0, int(cy-radius)), min(voxels.shape[1], int(cy+radius+1))):
            for x in range(max(0, int(cx-radius)), min(voxels.shape[0], int(cx+radius+1))):
                dx = x - cx
                dy = y - cy
                dz = z - cz
                if dx*dx + dy*dy + dz*dz <= radius*radius:
                    voxels[x, y, z] = value

def add_box(voxels, x0, y0, z0, x1, y1, z1, value):
    """Add a filled box to voxel array"""
    for z in range(max(0, z0), min(voxels.shape[2], z1+1)):
        for y in range(max(0, y0), min(voxels.shape[1], y1+1)):
            for x in range(max(0, x0), min(voxels.shape[0], x1+1)):
                voxels[x, y, z] = value

def main():
    print("GPU-VDB Example 08: Voxel Art (Procedural Skull)")

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return 1

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tree = gpuvdb.VDBTree()
    print("VDB tree initialized")

    # Create procedural skull
    size = 100
    voxels = np.zeros((size, size, size), dtype=float)

    print(f"\nBuilding skull...")

    cx, cy = size // 2, size // 2

    # Main cranium (large sphere)
    add_sphere(voxels, cx, cy, 60, 22, 0.8)

    # Face/jaw area (ellipsoid - front of skull)
    for z in range(30, 50):
        scale = 1.0 - (50 - z) / 20.0
        r = 15 * scale
        add_sphere(voxels, cx, cy, z, r, 0.7)

    # Eye sockets (hollow)
    add_sphere(voxels, cx-10, cy, 50, 6, 0.0)
    add_sphere(voxels, cx+10, cy, 50, 6, 0.0)

    # Eye socket depth
    add_sphere(voxels, cx-10, cy, 48, 5, 0.0)
    add_sphere(voxels, cx+10, cy, 48, 5, 0.0)

    # Nose cavity (triangle shape)
    for z in range(38, 48):
        size_nose = (48 - z) * 0.3
        add_sphere(voxels, cx, cy, z, size_nose, 0.0)

    # Teeth (upper jaw)
    tooth_y = cy - 8
    for i, tooth_x in enumerate([cx-8, cx-4, cx, cx+4, cx+8]):
        add_box(voxels, tooth_x-1, tooth_y, 32, tooth_x+1, tooth_y+2, 38, 0.9)

    # Jaw bone (lower)
    add_box(voxels, cx-12, cy-10, 28, cx+12, cy-8, 35, 0.7)

    # Cheekbones (zygomatic arch)
    add_box(voxels, cx-15, cy-3, 48, cx-10, cy+3, 52, 0.75)
    add_box(voxels, cx+10, cy-3, 48, cx+15, cy+3, 52, 0.75)

    # Forehead ridge
    add_box(voxels, cx-12, cy-2, 58, cx+12, cy+2, 62, 0.85)

    # Temple indents (slight hollows)
    add_sphere(voxels, cx-18, cy, 62, 4, 0.6)
    add_sphere(voxels, cx+18, cy, 62, 4, 0.6)

    # Convert to sparse representation
    points = []
    values = []

    for z in range(size):
        for y in range(size):
            for x in range(size):
                if voxels[x, y, z] > 0:
                    points.append([x, y, z])
                    values.append(float(voxels[x, y, z]))

    print(f"Skull built: {len(points)} voxels")

    errors = 0
    # Upload to VDB
    if len(points) > 0:
        coords = torch.tensor(points, dtype=torch.int32, device=device)
        vals = torch.tensor(values, dtype=torch.float32, device=device)

        print(f"  Uploading to VDB...")
        tree.insert(coords, vals)

        num_internal, num_leaf = tree.get_memory_stats()
        print(f"\nSkull stored in VDB")
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

        errors = 0
        active_failed = np.sum(test_active_queried != 1)
        if active_failed > 0:
            print(f"  FAIL: {active_failed}/{num_test} points are not active")
            errors += 1

        # Check if values are in reasonable range [0, 1]
        out_of_range = np.sum((test_values_queried < -0.1) | (test_values_queried > 1.1))
        if out_of_range > num_test * 0.1:
            print(f"  FAIL: {out_of_range}/{num_test} values are out of range")
            errors += 1

        if errors == 0:
            print("  All tests passed")

    # Visualize from multiple angles
    print(f"\nRendering views...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Different views showing skull features
    views = [
        ('XY', 50, 'Front view (eyes/nose)'),
        ('XY', 45, 'Nose cavity'),
        ('XY', 60, 'Cranium top'),
        ('XZ', cy, 'Side profile'),
        ('YZ', cx, 'Side profile (other)'),
        ('XY', 35, 'Jaw/teeth'),
    ]

    for idx, (plane, slice_val, title) in enumerate(views):
        ax = axes[idx // 3, idx % 3]

        if plane == 'XY':
            coords_list = []
            for y in range(size):
                for x in range(size):
                    coords_list.append([x, y, slice_val])
            w, h = size, size

        elif plane == 'XZ':
            coords_list = []
            for z in range(size):
                for x in range(size):
                    coords_list.append([x, slice_val, z])
            w, h = size, size

        else:  # YZ
            coords_list = []
            for z in range(size):
                for y in range(size):
                    coords_list.append([slice_val, y, z])
            w, h = size, size

        coords = torch.tensor(coords_list, dtype=torch.int32, device=device)
        values_out = tree.query(coords)
        active_out = tree.active(coords)

        values_cpu = values_out.cpu().numpy().reshape((h, w))
        active_cpu = active_out.cpu().numpy().reshape((h, w))

        masked = np.ma.masked_where(~active_cpu, values_cpu)
        im = ax.imshow(masked, cmap='bone', origin='lower',
                      extent=[0, w, 0, h], vmin=0, vmax=1)

        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel('X' if plane in ['XY', 'XZ'] else 'Y')
        ax.set_ylabel('Y' if plane == 'XY' else 'Z')
        ax.grid(False)
        plt.colorbar(im, ax=ax, label='Bone Density')

    fig.suptitle(f'GPU-VDB Voxel Art: Procedural Skull ({len(points)} voxels)',
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
    z_slices = [40, 50, 60]
    
    for idx, z_val in enumerate(z_slices):
        ax = axes[idx]
        coords = torch.tensor([[x, y, z_val] for y in range(size) for x in range(size)],
                             dtype=torch.int32, device=device)
        vals = tree.query(coords).cpu().numpy().reshape(size, size)
        actv = tree.active(coords).cpu().numpy().reshape(size, size)
        
        masked = np.ma.masked_where(~actv, vals)
        ax.imshow(masked, cmap='bone', origin='lower', extent=[0, size, 0, size])
        
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
    
    fig.suptitle('GPU-VDB Voxel Art (Skull) 2D (Blue=Root, Green=Internal, Red=Leaf)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'skull_2d.png', dpi=150, bbox_inches='tight')
    print("Saved skull_2d.png")
    
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
                   c=vals_active, cmap='bone', s=3, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Value', shrink=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'GPU-VDB Voxel Art 3D ({len(coords_active)} voxels)', fontsize=14, weight='bold')
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 128)

    plt.savefig(Path(__file__).parent / 'skull_3d.png', dpi=150, bbox_inches='tight')
    print("Saved skull_3d.png")


    tree.free()

    return errors

if __name__ == '__main__':
    import sys
    sys.exit(main())
