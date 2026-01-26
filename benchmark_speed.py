
import time
import jax
import jax.numpy as jnp
import numpy as np
from splinebox.jax_spline_curves_2d import JaxSpline
from splinebox.jax_basis_functions import JaxB3

def old_moving_frame_simulation(spline, t, initial_vector=None):
    """
    Simulates the old moving_frame method without correction.
    """
    # Logic copied/adapted from JaxSpline.moving_frame before changes
    t_arr, single_value = spline._convert_to_array(t)
    
    sort_idx = jnp.argsort(t_arr)
    t_sorted = t_arr[sort_idx]
    
    d1 = spline(t_sorted, derivative=1)
    d2 = spline(t_sorted, derivative=2)
    d0 = spline(t_sorted, derivative=0)
    
    T_norm = jnp.linalg.norm(d1, axis=-1, keepdims=True)
    T = d1 / (T_norm + 1e-12)
    
    # Bishop method specific
    if initial_vector is None:
        guess = jnp.cross(jnp.cross(T[0], d2[0]), T[0])
        guess_norm = jnp.linalg.norm(guess)
        guess_is_degen = jnp.isclose(guess_norm, 0.0) or jnp.any(jnp.isnan(guess))
        initial_vector = spline._initial_vector_guess(T[0], guess, guess_is_degen)
        
    # closed=False for simpler benchmark, or True to test overhead?
    # Original benchmark compared "old" vs "new". "Old" simulation was essentially just raw transport.
    # New optimized version includes correction inside JIT.
    # Let's pass closed=True to measure full cost including correction logic (which is compiled).
    
    frames = spline._moving_frame_bishop(d0, T, initial_vector, t_sorted, spline.M, True)
    
    # Correction and binormal calculation is now inside _moving_frame_bishop
    
    # Skip correction block (that was the old behavior for closed=True or False)
    
    if single_value: return frames[0]
    inv_sort_idx = jnp.argsort(sort_idx)
    frames = frames[inv_sort_idx]
    
    return frames

def benchmark():
    try:
        import matplotlib.pyplot as plt
        HAS_PLOT = True
    except ImportError:
        HAS_PLOT = False
        print("Matplotlib not found, skipping plot generation.")

    print("Setting up benchmark...")
    M = 20 
    spline = JaxSpline(M=M, basis_function=JaxB3(), closed=True)
    
    key = jax.random.PRNGKey(0)
    cps = jax.random.uniform(key, (M, 3), minval=-2.0, maxval=2.0)
    spline.control_points = cps
    
    n_iter = 20
    query_sizes = [10, 50, 100, 200]
    
    results = {
        "N": query_sizes,
        "old_mean": [],
        "new_mean": [],
        "overhead": []
    }
    
    print(f"Running benchmark with {n_iter} iterations for N={query_sizes}...")
    
    # Warmup once
    print("Warming up JIT...")
    t_warm = jnp.linspace(0, M, 10)
    _ = old_moving_frame_simulation(spline, t_warm)
    _ = spline.moving_frame(t_warm, method="bishop")
    jax.block_until_ready(_)
    
    print(f"{'N':<10} | {'Old (ms)':<15} | {'New (ms)':<15} | {'Overhead (ms)':<15} | {'Factor':<10}")
    print("-" * 75)
    
    for N in query_sizes:
        t = jnp.linspace(0, M, N)
        
        # Measure Old
        start_time = time.time()
        for _ in range(n_iter):
            res = old_moving_frame_simulation(spline, t)
            jax.block_until_ready(res)
        end_time = time.time()
        avg_old = (end_time - start_time) / n_iter * 1000
        
        # Measure New
        start_time = time.time()
        for _ in range(n_iter):
            res = spline.moving_frame(t, method="bishop")
            jax.block_until_ready(res)
        end_time = time.time()
        avg_new = (end_time - start_time) / n_iter * 1000
        
        overhead = avg_new - avg_old
        factor = avg_new / avg_old if avg_old > 0 else 0
        
        results["old_mean"].append(avg_old)
        results["new_mean"].append(avg_new)
        results["overhead"].append(overhead)
        
        print(f"{N:<10} | {avg_old:<15.4f} | {avg_new:<15.4f} | {overhead:<15.4f} | {factor:<10.2f}x")

    if HAS_PLOT:
        plt.figure(figsize=(10, 6))
        plt.plot(results["N"], results["old_mean"], marker='o', label='Old (No Correction)')
        plt.plot(results["N"], results["new_mean"], marker='o', label='New (Corrected)')
        plt.xlabel('Number of Query Points (N)')
        plt.ylabel('Execution Time (ms)')
        plt.title('Bishop Frame Computation Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig('benchmark_results.png')
        print("\nPlot saved to benchmark_results.png")

if __name__ == "__main__":
    benchmark()
