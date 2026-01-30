
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from splinebox.jax_spline_curves import JaxSpline
from splinebox.jax_basis_functions import JaxB3

def visualize_tube():
    M = 20
    basis = JaxB3()
    
    # Create a 3D knot like shape
    t = np.linspace(0, 2*np.pi, M, endpoint=False)
    x = (2 + np.cos(3*t)) * np.cos(2*t)
    y = (2 + np.cos(3*t)) * np.sin(2*t)
    z = np.sin(3*t)
    
    points_3d = np.stack([x, y, z], axis=1) * 3.0 # Scale up
    
    spline = JaxSpline(M, basis, closed=True, control_points=points_3d, is_ribbon=False)
    
    # Generate Centerline
    t_eval = np.linspace(0, M, 200)
    centerline = spline(t_eval)
    
    # Generate Tube Mesh
    radius = 0.5
    # step_t=0.2 (20/0.2 = 100 slices), step_angle=30 (12 pts around)
    points_flat, conn = spline.mesh(step_t=0.2, radius=radius, step_angle=30)
    
    # Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Centerline
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], 'k-', linewidth=3, label='Centerline')
    
    # Plot Tube Mesh
    ax.plot_trisurf(points_flat[:, 0], points_flat[:, 1], points_flat[:, 2], 
                    triangles=conn, cmap='coolwarm', edgecolor='none', alpha=0.6)
    
    # Plot frames at a few points
    sample_t = np.array([0., 5., 10., 15.])
    frames = spline.moving_frame(sample_t, method="bishop")
    pts = spline(sample_t)
    
    normals = frames[:, 1, :]
    binormals = frames[:, 2, :]
    
    # Verify orthogonality
    ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2], 
              normals[:, 0], normals[:, 1], normals[:, 2], length=1.0, color='r', label='Normal')
    ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2], 
              binormals[:, 0], binormals[:, 1], binormals[:, 2], length=1.0, color='g', label='Binormal')

    ax.set_title(f"Tube Mesh Visualization (Radius={radius})\nTorus Knot")
    ax.legend()
    
    # Equal aspect ratio hack
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() * 3.0 / 2.0
    mid_x = (x.max()+x.min()) * 3.0 * 0.5
    mid_y = (y.max()+y.min()) * 3.0 * 0.5
    mid_z = (z.max()+z.min()) * 3.0 * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    output_path = "/home/magna/.gemini/antigravity/brain/af24cecc-ed2d-4c00-9935-231d0f4c321e/tube_visualization.png"
    plt.savefig(output_path, dpi=100)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_tube()
