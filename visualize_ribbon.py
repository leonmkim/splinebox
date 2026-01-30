
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from splinebox.jax_spline_curves import JaxSpline
from splinebox.jax_basis_functions import JaxB3

def visualize_ribbon():
    M = 20
    basis = JaxB3()
    
    # Create a 3D circle with a twist
    t = np.linspace(0, 2*np.pi, M, endpoint=False)
    radius = 5.0
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = 2.0 * np.sin(2 * t) # Wavy in Z
    theta = t * 2.0 # twist twice around
    
    points_4d = np.stack([x, y, z, theta], axis=1)
    
    spline = JaxSpline(M, basis, closed=True, control_points=points_4d, is_ribbon=True, ribbon_width=2.0)
    
    # Generate high-res Centerline
    t_eval = np.linspace(0, M, 200)
    centerline = spline.centerline(t_eval)
    
    # Generate Mesh
    # step_t=0.1 gives 20*10 = 200 steps roughly
    points_flat, conn = spline.mesh(step_t=0.2, u_resolution=5)
    
    # Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Centerline
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], 'k-', linewidth=2, label='Centerline')
    
    # Plot Ribbon Mesh using Trisurf
    # Normalize connectivity to 0-based index if not already? 
    # mesh returns flat points and connectivity indices.
    # conn shape (N_tris, 3)
    
    ax.plot_trisurf(points_flat[:, 0], points_flat[:, 1], points_flat[:, 2], 
                    triangles=conn, cmap='viridis', edgecolor='none', alpha=0.8)

    # Plot frames at a few points to verify twist
    # Sample every M/2 points
    sample_t = np.array([0, M/4, M/2, 3*M/4])
    frames = spline.moving_frame(sample_t, method="bishop")
    
    pts = spline.centerline(sample_t)
    normals = frames[:, 1, :] # Bishop N
    binormals = frames[:, 2, :] # Bishop B
    
    thetas = spline.theta(sample_t)
    
    # Plot actual surface normal (twisted)
    cos_th = np.cos(thetas)[:, None]
    sin_th = np.sin(thetas)[:, None]
    n_surf = cos_th * normals + sin_th * binormals
    
    ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2], 
              n_surf[:, 0], n_surf[:, 1], n_surf[:, 2], length=3.0, color='r', label='Twisted Normal')
    
    ax.set_title("Ribbon Spline Visualization\nTwisted 720 deg along circle")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Equal aspect ratio hack
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save
    output_path = "/home/magna/.gemini/antigravity/brain/af24cecc-ed2d-4c00-9935-231d0f4c321e/ribbon_visualization.png"
    plt.savefig(output_path, dpi=100)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_ribbon()
