
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from splinebox.jax_spline_curves import JaxSpline
from splinebox.jax_basis_functions import JaxB3
import os

def generate_random_ribbons(num_trials=10):
    M = 20
    basis = JaxB3()
    
    # Dedicated output directory in the artifact folder
    base_dir = "/home/magna/.gemini/antigravity/brain/af24cecc-ed2d-4c00-9935-231d0f4c321e"
    output_dir = os.path.join(base_dir, "ribbon_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42) # Reproducibility
    
    for i in range(num_trials):
        print(f"Generating Ribbon #{i+1}...")
        
        # 1. Base shape: Circle
        t = np.linspace(0, 2*np.pi, M, endpoint=False)
        radius = 3.0 + np.random.rand() * 2.0 # Random radius 3-5
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.zeros_like(t)
        
        # 2. Perturbation
        noise_scale = 1.0
        x += np.random.uniform(-noise_scale, noise_scale, size=M)
        y += np.random.uniform(-noise_scale, noise_scale, size=M)
        z += np.random.uniform(-1.5, 1.5, size=M) 
        
        # 3. Constrained Theta
        theta_center = np.random.uniform(0, 2*np.pi)
        freq = np.random.randint(1, 3)
        theta_variation = 0.5 * np.pi * np.sin(freq * t + np.random.rand()*2*np.pi)
        theta = theta_center + theta_variation
        
        points_4d = np.stack([x, y, z, theta], axis=1)
        
        # Create Spline
        spline = JaxSpline(M, basis, closed=True, control_points=points_4d, is_ribbon=True, ribbon_width=1.0)
        
        # Mesh
        points_flat, conn = spline.mesh(step_t=0.2, u_resolution=5)
        t_eval = np.linspace(0, M, 200)
        centerline = spline.centerline(t_eval)
        
        # Plot Setup - BIGGER FIGURE
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot data once
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], 'k-', linewidth=2.0, label='Centerline')
        ax.plot_trisurf(points_flat[:, 0], points_flat[:, 1], points_flat[:, 2], 
                        triangles=conn, cmap='viridis', edgecolor='none', alpha=0.9)
        
        # Limits
        max_range = 6.0
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_title(f"Random Ribbon #{i+1}\nTwist Range ~ $\pi$")
        
        # Animation
        def update(angle):
            ax.view_init(elev=30, azim=angle)
            return fig,

        # Create animation (360 degrees, 36 frames = 10 deg per frame to keep file size manageable)
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 5), interval=50)
        
        filename = f"ribbon_{i+1}.gif"
        path = os.path.join(output_dir, filename)
        
        # Save as GIF using Pillow writer
        anim.save(path, writer='pillow', fps=20)
        plt.close(fig)
        print(f"Saved animation to {path}")

if __name__ == "__main__":
    generate_random_ribbons()
