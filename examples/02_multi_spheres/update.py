#!/usr/bin/env python3
"""Update script to add export_quad_mesh visualizations"""
import re

# Read original file
with open('multi_spheres.py', 'r') as f:
    content = f.read()

# Add imports
if 'from mpl_toolkits.mplot3d import Axes3D' not in content:
    content = content.replace(
        'from pathlib import Path',
        'from pathlib import Path\nfrom mpl_toolkits.mplot3d import Axes3D'
    )

# Find where visualization starts and replace
viz_start = content.find('print("\\nQuerying slice')
if viz_start > 0:
    # Keep everything before visualization
    before = content[:viz_start]
    
    # New visualization code
    new_viz = '''print("\\nExporting tree structure...")
    mesh = tree.export_quad_mesh()
    vertices = mesh['vertices']
    quads = mesh['quads']
    levels = mesh['levels']
    
    print("Querying slices...")
    size = 100
    z_slices = [30, 50, 70]
    
    # 2D visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, z_val in enumerate(z_slices):
        ax = axes[idx]
        coords = torch.tensor([[x, y, z_val] for y in range(size) for x in range(size)],
                             dtype=torch.int32, device=device)
        vals = tree.query(coords).cpu().numpy().reshape(size, size)
        actv = tree.active(coords).cpu().numpy().reshape(size, size)
        
        masked = np.ma.masked_where(~actv, vals)
        ax.imshow(masked, cmap='viridis', origin='lower', extent=[0, size, 0, size], vmin=0, vmax=10)
        
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
    
    # 3D visualization
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')
    for level in [2, 1, 0]:
        level_mask = levels == level
        level_quads = quads[level_mask]
        color = ['#ff4444', '#44ff44', '#4444ff'][level]
        for quad in level_quads[::2]:
            pts = vertices[quad]
            for i in range(4):
                edge = np.array([pts[i], pts[(i+1)%4]])
                ax.plot3D(edge[:, 0], edge[:, 1], edge[:, 2], color=color, alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('GPU-VDB Multi-Spheres Tree\\n(Red=Leaf, Green=Internal, Blue=Root)', fontsize=14, weight='bold')
    plt.savefig(Path(__file__).parent / 'multisphere_3d.png', dpi=150, bbox_inches='tight')
    print("Saved multisphere_3d.png")
'''
    
    # Write new file
    with open('multi_spheres.py', 'w') as f:
        f.write(before + new_viz + '\n\n    tree.free()\n    return 0\n\nif __name__ == "__main__":\n    sys.exit(main())\n')
    print("Updated multi_spheres.py")

