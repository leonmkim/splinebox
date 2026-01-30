
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import os
from splinebox.jax_spline_curves import JaxSpline
from splinebox.jax_basis_functions import JaxB3

def benchmark_caching():
    M = 20
    basis = JaxB3()
    output_dir = "/home/magna/.gemini/antigravity/brain/af24cecc-ed2d-4c00-9935-231d0f4c321e"
    
    # Setup Random Ribbon
    np.random.seed(42)
    t = np.linspace(0, 2*np.pi, M, endpoint=False)
    x = (3 + np.cos(3*t)) * np.cos(t)
    y = (3 + np.cos(3*t)) * np.sin(t)
    z = np.sin(2*t)
    theta = 0.5 * np.sin(t)
    points_4d = np.stack([x, y, z, theta], axis=1)
    
    spline = JaxSpline(M, basis, closed=True, control_points=points_4d, is_ribbon=True, ribbon_width=1.0)
    
    # --- 1. Speedup Benchmark ---
    print("Running Speedup Benchmark...")
    N_values = [10, 100, 1000, 10000, 100000]
    times_cached = []
    times_exact = []
    
    # Warmup
    _ = spline.moving_frame(jnp.array([0., 1.]), method="bishop").block_until_ready()
    spline._frame_cache = None
    _ = spline.moving_frame(jnp.array([0., 1.]), method="bishop").block_until_ready()
    # Restore cache
    spline._frame_cache = spline._compute_dense_bishop_frames()
    
    for N in N_values:
        print(f"  N={N}...")
        t_eval = jnp.linspace(0, M, N)
        
        # Cached
        # Force cache usage (it is populated)
        start = time.time()
        res_cached = spline.moving_frame(t_eval, method="bishop").block_until_ready()
        times_cached.append(time.time() - start)
        
        # Exact (No Cache)
        # We simulate "fresh" computation by temporarily clearing cache
        cache_backup = spline._frame_cache
        spline._frame_cache = None
        
        start = time.time()
        res_exact = spline.moving_frame(t_eval, method="bishop").block_until_ready()
        times_exact.append(time.time() - start)
        
        # Restore
        spline._frame_cache = cache_backup

    # Plot Speedup
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(N_values, times_cached, 'b-o', label='Cached (Interpolation)')
    ax.loglog(N_values, times_exact, 'r-x', label='Exact (Fresh RMF Scan)')
    ax.set_xlabel('Number of Query Points (N)')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Moving Frame Computation Time: Cached vs Exact')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    
    # Improve y-tick resolution
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%.4f'))
    # Ensure minor ticks are shown
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=20))
    
    speedup_path = os.path.join(output_dir, "caching_speedup.png")
    plt.savefig(speedup_path)
    print(f"Saved {speedup_path}")
    plt.close(fig)

    # --- 2. Error Benchmark ---
    print("Running Error Benchmark...")
    # We compare interpolated frames against "Ground Truth".
    # Ground Truth: High-quality RMF integration.
    # We'll use the 'res_exact' from N=10000 (dense enough for RMF to be accurate)
    # and compare it to 'res_cached' from same N.
    
    # Re-run for N=10000 to be sure
    N_err = 10000
    t_eval = jnp.linspace(0, M, N_err)
    
    res_cached = spline.moving_frame(t_eval, method="bishop")
    
    cache_backup = spline._frame_cache
    spline._frame_cache = None
    res_exact = spline.moving_frame(t_eval, method="bishop") # Fresh RMF
    spline._frame_cache = cache_backup
    
    # Calculate angular error (angle between Normal vectors)
    # N_cached, N_exact
    N_c = res_cached[:, 1, :]
    N_e = res_exact[:, 1, :]
    
    # Dot product clamped to [-1, 1]
    dot = jnp.sum(N_c * N_e, axis=-1)
    dot = jnp.clip(dot, -1.0, 1.0)
    angles_rad = jnp.arccos(dot)
    angles_deg = jnp.degrees(angles_rad)
    
    # Plot Error
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_eval, angles_deg, 'k-', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Spline Parameter t')
    ax.set_ylabel('Normal Vector Error (degrees)')
    ax.set_title(f'Interpolation Error vs Ground Truth (N={N_err})')
    ax.grid(True)
    
    # Add stats
    max_err = jnp.max(angles_deg)
    mean_err = jnp.mean(angles_deg)
    ax.text(0.02, 0.95, f'Max Error: {max_err:.4f}°\nMean Error: {mean_err:.4f}°', 
            transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    error_path = os.path.join(output_dir, "caching_error.png")
    plt.savefig(error_path)
    print(f"Saved {error_path}")
    plt.close(fig)

    # --- 3. Sparse Query Accuracy Benchmark ---
    print("Running Sparse Query Accuracy Benchmark...")
    # Verify intuition: For sparse t, "from-scratch" causes large integration steps -> high error.
    # Cache (interpolated) -> low error (based on dense pre-integration).
    
    # Generate Ground Truth (Dense)
    N_gt = 10000
    t_gt = jnp.linspace(0, M, N_gt)
    # Ensure cache is dense
    spline._frame_cache = spline._compute_dense_bishop_frames(samples_per_knot=100) 
    frames_gt = spline.moving_frame(t_gt, method="bishop")
    
    # Select sparse points randomly (e.g., 20 points)
    # We choose points that align with GT grid to avoid GT interpolation error dominating
    # idxs = np.random.choice(N_gt, 20, replace=False)
    # Actually, let's just pick strictly sparse points (e.g. integer t + 0.5)
    # and compare against the corresponding GT frame found via dense lookup.
    
    num_sparse = 20
    t_sparse = np.sort(np.random.uniform(0, M, num_sparse))
    
    # A. Cached Method
    # Cache is already populated
    frames_cached_sparse = spline.moving_frame(t_sparse, method="bishop")
    
    # B. Scratch Method
    # Disable cache
    cache_backup = spline._frame_cache
    spline._frame_cache = None
    frames_scratch_sparse = spline.moving_frame(t_sparse, method="bishop") # computes on t_sparse only
    spline._frame_cache = cache_backup # Restore
    
    # Ground Truth at t_sparse
    # We re-compute GT specifically at these points? No, that would be "scratch".
    # GT is "limit of dense integration".
    # We can get GT by asking for moving_frame on (t_sparse UNION dense_grid) and sorting?
    # Or just computing moving_frame(dense_grid) and interpolating to t_sparse (very accurately)?
    # Since we established Cached -> GT is accurate, we can treat "Cached" as proxy for GT?
    # No, we want to show Cache is Better.
    # We need a Reference that is "Exact Dense Integration".
    # We can compute moving_frame on a very dense grid (100k) and interpolate?
    
    t_super_dense = jnp.union1d(jnp.linspace(0, M, 10000), t_sparse)
    t_super_dense = jnp.sort(t_super_dense)
    # Force fresh computation on super dense grid (highly accurate)
    spline._frame_cache = None
    frames_super = spline.moving_frame(t_super_dense, method="bishop")
    spline._frame_cache = cache_backup
    
    # Extract frames at t_sparse from frames_super
    # specific indices? t_super_dense has t_sparse values exactly.
    # Find indices
    idxs_map = jnp.searchsorted(t_super_dense, t_sparse)
    frames_gt_sparse = frames_super[idxs_map]
    
    # Calculate Errors
    def get_max_angle_error(f1, f2):
        n1 = f1[:, 1, :]
        n2 = f2[:, 1, :]
        dot = jnp.sum(n1 * n2, axis=-1)
        dot = jnp.clip(dot, -1.0, 1.0)
        deg = jnp.degrees(jnp.arccos(dot))
        return deg

    err_cached = get_max_angle_error(frames_cached_sparse, frames_gt_sparse)
    err_scratch = get_max_angle_error(frames_scratch_sparse, frames_gt_sparse)
    
    # Plot Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x_axis = np.arange(num_sparse)
    
    rects1 = ax.bar(x_axis - width/2, err_cached, width, label='Cached (Dense Interp)')
    rects2 = ax.bar(x_axis + width/2, err_scratch, width, label='From Scratch (Sparse Step)', color='r')
    
    ax.set_ylabel('Error vs Ground Truth (degrees)')
    ax.set_xlabel('Sparse Query Index')
    ax.set_title('Accuracy for Sparse Queries: Cached vs From-Scratch')
    ax.set_yscale('log') # Log scale because difference might be huge
    ax.legend()
    ax.grid(True, which="both", axis='y', alpha=0.3)
    
    # Stats
    mean_bias = jnp.mean(err_scratch) / (jnp.mean(err_cached) + 1e-9)
    print(f"Mean Error Ratio (Scratch/Cached): {mean_bias:.1f}x")
    
    sparse_path = os.path.join(output_dir, "sparse_query_error.png")
    plt.savefig(sparse_path)
    print(f"Saved {sparse_path}")
    plt.close(fig)

if __name__ == "__main__":
    benchmark_caching()
