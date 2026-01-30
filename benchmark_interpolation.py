
import jax
import jax.numpy as jnp
import numpy as np
import time
from jax.scipy.ndimage import map_coordinates

# --- Implementations ---

# 1. Current Manual Vectorized
@jax.jit
def interp_manual_vectorized(t, cache):
    t_grid, frames_grid = cache
    # frames_grid shape: (N_grid, 3, 3)
    N_grid = frames_grid.shape[0]
    limit = t_grid[-1]
    
    # Indices
    factor = (N_grid - 1) / (limit + 1e-12)
    idx_float = t * factor
    idx_float = jnp.clip(idx_float, 0, N_grid - 1.0001)
    
    idx_floor = jnp.floor(idx_float).astype(jnp.int32)
    idx_ceil = idx_floor + 1
    idx_ceil = jnp.minimum(idx_ceil, N_grid - 1)
    
    alpha = idx_float - idx_floor
    
    frames_flat = frames_grid.reshape(N_grid, -1)
    
    f0 = jnp.take(frames_flat, idx_floor, axis=0) 
    f1 = jnp.take(frames_flat, idx_ceil, axis=0)
    
    interpolated_flat = f0 * (1 - alpha[:, None]) + f1 * alpha[:, None]
    
    # Post-process (Orthonormalize) omitted for fair memory/gather bench? 
    # Or include it to catch overhead? Include it.
    interpolated_frames = interpolated_flat.reshape(-1, 3, 3)
    T_approx = interpolated_frames[:, 0, :]
    N_approx = interpolated_frames[:, 1, :]
    
    def safe_normalize(v):
        return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

    T_new = safe_normalize(T_approx)
    dot_nt = jnp.sum(N_approx * T_new, axis=-1, keepdims=True)
    N_ortho = N_approx - dot_nt * T_new
    N_new = safe_normalize(N_ortho)
    B_new = jnp.cross(T_new, N_new)
    
    return jnp.stack([T_new, N_new, B_new], axis=1)

# 2. Vmap Scalar
@jax.jit
def interp_vmap(t_arr, cache):
    t_grid, frames_grid = cache
    N_grid = frames_grid.shape[0]
    limit = t_grid[-1]
    frames_flat = frames_grid.reshape(N_grid, -1)
    
    def single_point(ti):
        factor = (N_grid - 1) / (limit + 1e-12)
        idx_f = ti * factor
        idx_f = jnp.clip(idx_f, 0, N_grid - 1.0001)
        
        idx_fl = jnp.floor(idx_f).astype(jnp.int32)
        idx_ce = jnp.minimum(idx_fl + 1, N_grid - 1)
        alpha = idx_f - idx_fl
        
        # Take single
        # Although frames_flat is large, we index it.
        # Inside vmap, this becomes gather.
        f0 = frames_flat[idx_fl]
        f1 = frames_flat[idx_ce]
        val = f0 * (1 - alpha) + f1 * alpha
        
        # Orthonormalize scalar
        frame = val.reshape(3, 3)
        T = frame[0]
        N = frame[1]
        
        # safe norm for scalar
        def n(v): return v / (jnp.linalg.norm(v) + 1e-12)
        
        T_n = n(T)
        N_n = n(N - jnp.dot(N, T_n) * T_n)
        B_n = jnp.cross(T_n, N_n)
        return jnp.stack([T_n, N_n, B_n], axis=0)

    return jax.vmap(single_point, in_axes=(0))(t_arr)

# 3. Map Coordinates
@jax.jit
def interp_map_coordinates(t, cache):
    t_grid, frames_grid = cache
    # frames_grid shape: (N_grid, 3, 3) -> (N_grid, 9)
    # map_coordinates wants input: (ndim, point_coords)
    # But here input is 1D array (frames_flat column)
    # We want to interpolate along axis 0 for each of 9 channels?
    # This might be slow if we loop over 9 channels?
    # Or map_coordinates supports "vector" output? No.
    # We treat image as (N_grid, 9)?
    # We want to interp at (coord, channel). 
    # But channel is integer grid 0..8.
    
    N_grid = frames_grid.shape[0]
    limit = t_grid[-1]
    factor = (N_grid - 1) / (limit + 1e-12)
    coords = t * factor # (K,)
    
    # We have 9 features.
    # Efficient way: vmap over features? 
    # Or Reshape to (Height, Width) and interp?
    # map_coordinates works on volume.
    # Data: (N_grid, 9).
    # Coords: We want (t_mapped, 0), (t_mapped, 1) ... (t_mapped, 8).
    # This seems complex to setup for map_coordinates.
    # Simpler: Transpose to (9, N_grid). vmap map_coordinates over the 9 channels.
    
    data = frames_grid.reshape(N_grid, 9).T # (9, N_grid)
    
    def map_channel(channel_data):
        # channel_data: (N_grid,)
        return map_coordinates(channel_data, [coords], order=1, mode='nearest')
        
    interpolated_flat = jax.vmap(map_channel)(data) # (9, K)
    interpolated_flat = interpolated_flat.T # (K, 9)
    
    # Post process
    interpolated_frames = interpolated_flat.reshape(-1, 3, 3)
    
    # Standard vectorization for ortho (same as manual)
    T_approx = interpolated_frames[:, 0, :]
    N_approx = interpolated_frames[:, 1, :]
    def safe_normalize(v): return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    T_new = safe_normalize(T_approx)
    dot_nt = jnp.sum(N_approx * T_new, axis=-1, keepdims=True)
    N_ortho = N_approx - dot_nt * T_new
    N_new = safe_normalize(N_ortho)
    B_new = jnp.cross(T_new, N_new)
    return jnp.stack([T_new, N_new, B_new], axis=1)



# 4. User Suggested (Manual Fast Indexing)
@jax.jit
def interp_user_suggested(t, cache):
    t_grid, frames_grid = cache
    N = frames_grid.shape[0]
    limit = t_grid[-1]

    # Assume t checks or closed handled externally for benchmark parity
    
    factor = (N - 1) / (limit + 1e-12)
    x = t * factor
    
    # Clip to match behavior
    x = jnp.clip(x, 0, N - 1.0001)

    i0 = jnp.floor(x).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, N - 1)
    
    w = (x - i0.astype(x.dtype))[:, None, None]

    # Gather full frames (K, 3, 3) 
    F0 = frames_grid[i0]
    F1 = frames_grid[i1]
    F  = (1.0 - w) * F0 + w * F1

    # Re-orthonormalize
    T_approx = F[:, 0, :]
    N_approx = F[:, 1, :]

    def safe_normalize(v): return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    T_new = safe_normalize(T_approx)
    dot_nt = jnp.sum(N_approx * T_new, axis=-1, keepdims=True)
    N_ortho = N_approx - dot_nt * T_new
    N_new = safe_normalize(N_ortho)

    B_new = jnp.cross(T_new, N_new)
    return jnp.stack([T_new, N_new, B_new], axis=1)


def run_benchmark():
    print("Benchmarking Interpolation Variants...")
    
    # Setup Data
    N_grid = 1000
    t_grid = jnp.linspace(0, 20, N_grid)
    frames_grid = jax.random.normal(jax.random.PRNGKey(0), (N_grid, 3, 3))
    cache = (t_grid, frames_grid)
    
    # Evaluation Points
    N_eval = 100000
    t_eval = jnp.linspace(0, 20, N_eval)
    
    # Warmup & Run
    variants = [
        ("Manual Vectorized", interp_manual_vectorized),
        ("Vmap Scalar", interp_vmap),
        ("Map Coordinates", interp_map_coordinates),
        ("User Suggested", interp_user_suggested)
    ]
    
    for name, func in variants:
        # Warmup
        _ = func(t_eval[:10], cache).block_until_ready()
        
        # Run
        start = time.time()
        for _ in range(10): # 10 loops to avg
             res = func(t_eval, cache).block_until_ready()
        duration = (time.time() - start) / 10.0
        
        print(f"{name}: {duration:.6f}s")
        
        # Verify result shape
        assert res.shape == (N_eval, 3, 3)

if __name__ == "__main__":
    run_benchmark()
