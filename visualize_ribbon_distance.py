
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from splinebox.jax_spline_curves import JaxSpline
from splinebox.jax_basis_functions import JaxB3

def run_distance_validation():
    print("Generating Ribbon Distance Validations...")
    
    # Output dir
    out_dir = "/home/magna/.gemini/antigravity/brain/af24cecc-ed2d-4c00-9935-231d0f4c321e/distance_visualizations"
    os.makedirs(out_dir, exist_ok=True)
    num_trials = 10
    # 3 Trials
    for trial_idx in range(num_trials):
        print(f"Trial {trial_idx+1}...")
        key = jax.random.PRNGKey(trial_idx * 100 + 42)
        
        # 1. Random Spline
        spline = JaxSpline(M=10, basis_function=JaxB3(), is_ribbon=True, ribbon_width=1.0)
        # Generate "better behaved" ribbons (convex-ish loop with mild 3D)
        # 1. Circle in XY
        theta = jnp.linspace(0, 2*jnp.pi, 10, endpoint=False)
        radius = 4.0
        x = radius * jnp.cos(theta)
        y = radius * jnp.sin(theta)
        z = jnp.zeros_like(x)
        base_cps = jnp.stack([x, y, z], axis=1)
        
        # 2. Add mild noise
        k1, k2, k3 = jax.random.split(key, 3)
        noise = jax.random.uniform(k1, (10, 3), minval=-0.5, maxval=0.5)
        # Add slight Z variation for 3Dness
        z_noise = jax.random.uniform(k3, (10, 1), minval=-1.0, maxval=1.0)
        noise = noise.at[:, 2].add(z_noise[:, 0])
        
        cps = base_cps + noise
        spline.control_points = cps
        
        # 2. Random Query Points
        # Generate points in bounding box of spline + noise
        min_vals = jnp.min(cps, axis=0) - 2.0
        max_vals = jnp.max(cps, axis=0) + 2.0
        
        n_queries = 15
        queries = jax.random.uniform(k2, (n_queries, 3), minval=min_vals, maxval=max_vals)
        
        # 3. Compute Distance
        # Expected return: dists, (t_opt, u_opt)
        dists, args = spline.distance(queries, return_arg=True)
        t_opt, u_opt = args
        
        # 4. Compute Closest Points
        closest_pts = spline.ribbon_surface(t_opt, u_opt)
        
        # Verify shape
        # print(f"Queries: {queries.shape}, Closest: {closest_pts.shape}")
        
        # 5. Visualize
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Spline Ribbon Mesh
        # We need a mesh grid for visualization
        t_vis = jnp.linspace(0, 10, 200)
        u_vis = jnp.linspace(-1, 1, 10)
        # ribbon_surface supports grid?
        # spline.ribbon_surface(t, u) with t(N), u(W) -> (N, W, 3)
        # Check signature: u: scalar or (N,) or (W,)
        mesh_pts = spline.ribbon_surface(t_vis, u_vis) # (200, 10, 3)
        
        # Plot surface wireframe or scatter
        # Wireframe manually
        X = mesh_pts[:, :, 0]
        Y = mesh_pts[:, :, 1]
        Z = mesh_pts[:, :, 2]
        
        ax.plot_surface(X, Y, Z, alpha=0.3, color='cyan', edgecolor='none')
        # Add edges for wireframe effect
        for i in range(0, 200, 20):
            ax.plot(X[i, :], Y[i, :], Z[i, :], 'b-', lw=0.5, alpha=0.5)
        for j in range(0, 10, 9): # borders
            ax.plot(X[:, j], Y[:, j], Z[:, j], 'b-', lw=1.0)

        # Plot Queries
        ax.scatter(queries[:, 0], queries[:, 1], queries[:, 2], c='red', s=50, label='Query')
        
        # Plot Closest
        ax.scatter(closest_pts[:, 0], closest_pts[:, 1], closest_pts[:, 2], c='lime', s=50, label='Closest')
        
        # Plot Lines
        for i in range(n_queries):
            ax.plot([queries[i, 0], closest_pts[i, 0]],
                    [queries[i, 1], closest_pts[i, 1]],
                    [queries[i, 2], closest_pts[i, 2]],
                    'k--', alpha=0.6, lw=1.5)
            
        ax.set_title(f"Ribbon Distance Validation (Trial {trial_idx+1})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        # Animation
        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return fig,
            
        ani = FuncAnimation(fig, update, frames=range(0, 360, 4), interval=50)
        
        save_path = f"{out_dir}/distance_trial_{trial_idx+1}.mp4"
        print(f"Saving {save_path}...")
        # Use FFMpegWriter for mp4
        writer =  jax.tree_util.Partial(lambda *args, **kwargs: None) # Dummy default to import
        from matplotlib.animation import FFMpegWriter
        ani.save(save_path, writer=FFMpegWriter(fps=20))
        plt.close(fig)

if __name__ == "__main__":
    run_distance_validation()
