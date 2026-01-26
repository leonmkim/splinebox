
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from splinebox.jax_spline_curves_2d import JaxSpline
from splinebox.jax_basis_functions import JaxB3

def generate_random_spline(key, M=12):
    # Base loop
    base_points = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [1.0, 2.0, -1.0],
        [0.0, 2.0, 0.0],
        [-1.0, 2.0, 1.0],
        [-2.0, 1.0, 0.0],
        [-1.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [1.0, -1.0, 1.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.0, -0.5] 
    ])
    # Perturb
    perturbation = jax.random.normal(key, base_points.shape) * 0.3
    control_points = base_points + perturbation
    
    # Ensure closedness implicitly by how we treat it? 
    # JaxSpline with closed=True treats the control points as periodic.
    # But usually we want first and last to be same? 
    # No, JaxSpline(closed=True) handles wrapping indices.
    # The control points are just M unique points.
    return JaxSpline(M=M, basis_function=JaxB3(), closed=True, control_points=control_points)

def get_bishop_mismatch(spline, frames, T, M, t_sorted):
    # We reproduce the logic from moving_frame to compute alpha
    # frames: (N, 3, 3) [T, N, B]
    normals = frames[:, 1, :]
    
    v = normals[0] 
    u1 = normals[-1]
    u2 = jnp.cross(T[-1], u1)
    
    c = jnp.dot(v, u1)
    s = jnp.dot(v, u2)
    alpha = jnp.arctan2(s, c)
    return alpha

def run_trial(key, n_values):
    spline = generate_random_spline(key)
    
    # Ground Truth: N=4000
    N_segments_ref = 4000
    t_ref = jnp.linspace(0, spline.M, N_segments_ref + 1)
    
    # Uncorrected RMF for GT
    # We call with closed=False to get the raw transport without twist
    frames_ref = spline.moving_frame(t_ref, method="bishop")
    # Wait, the public API with closed=True automatically corrects it.
    # We need to bypass the correction to measure the "uncorrected" error.
    # or just use the private method?
    # Public API 'moving_frame' calls private '_moving_frame_bishop'.
    # If we pass closed=False to moving_frame, it raises checks?
    # No, JaxSpline(closed=True) instance. moving_frame checks "if self.closed and method...".
    # So we can temporarily set spline.closed = False? Or call private method.
    
    # Let's call private method directly for full control.
    # _moving_frame_bishop(d0, T, initial_vector, t_sorted, M, closed=False)
    
    d1 = spline(t_ref, derivative=1)
    d2 = spline(t_ref, derivative=2)
    T_ref = d1 / jnp.linalg.norm(d1, axis=-1, keepdims=True)
    
    # Initial vector
    guess = jnp.cross(jnp.cross(T_ref[0], d2[0]), T_ref[0])
    guess_norm = jnp.linalg.norm(guess)
    is_degen = jnp.isclose(guess_norm, 0.0) or jnp.any(jnp.isnan(guess))
    init_vec_ref = spline._initial_vector_guess(T_ref[0], guess, is_degen)
    
    # GT Uncorrected
    frames_ref_uncorrected = spline._moving_frame_bishop(
        spline(t_ref), T_ref, init_vec_ref, t_ref, spline.M, closed=False
    )
    normals_ref_uncorrected = frames_ref_uncorrected[:, 1, :]
    
    # GT Alpha
    alpha_ref = get_bishop_mismatch(spline, frames_ref_uncorrected, T_ref, spline.M, t_ref)
    
    rmf_errors = []
    alpha_errors = []
    
    for n in n_values:
        step = int(N_segments_ref / n)
        t_coarse = t_ref[::step]
        
        # d1, d2, T for coarse
        d1_c = d1[::step]
        d2_c = d2[::step]
        T_c = T_ref[::step]
        
        # Ensure we use exactly the same initial vector for fair comparison?
        # Ideally we recalculate init vector from coarse data to see full pipeline error.
        # But if T[0] is same, init vector should be same.
        guess_c = jnp.cross(jnp.cross(T_c[0], d2_c[0]), T_c[0])
        guess_norm_c = jnp.linalg.norm(guess_c)
        is_degen_c = jnp.isclose(guess_norm_c, 0.0) or jnp.any(jnp.isnan(guess_c))
        init_vec_c = spline._initial_vector_guess(T_c[0], guess_c, is_degen_c)
        
        # Coarse Uncorrected
        frames_c_uncorrected = spline._moving_frame_bishop(
            spline(t_coarse), T_c, init_vec_c, t_coarse, spline.M, closed=False
        )
        normals_c = frames_c_uncorrected[:, 1, :]
        
        # 1. RMF Error (Uncorrected)
        normals_truth_sub = normals_ref_uncorrected[::step]
        dot = jnp.sum(normals_c * normals_truth_sub, axis=1)
        dot = jnp.clip(dot, -1.0, 1.0)
        angles_deg = jnp.degrees(jnp.arccos(dot))
        rmf_errors.append(jnp.max(angles_deg))
        
        # 2. Alpha Error
        alpha_c = get_bishop_mismatch(spline, frames_c_uncorrected, T_c, spline.M, t_coarse)
        diff_rad = jnp.abs(alpha_c - alpha_ref)
        # Handle wrap around? alpha is in [-pi, pi].
        # Mismatch should be small.
        alpha_errors.append(jnp.degrees(diff_rad))
        
    return jnp.array(rmf_errors), jnp.array(alpha_errors)

def main():
    key = jax.random.PRNGKey(42)
    n_trials = 20
    # Divisors of 4000
    n_values = [10, 20, 25, 40, 50, 80, 100, 200, 400, 500, 800, 1000]
    
    all_rmf_errors = []
    all_alpha_errors = []
    
    print(f"Running {n_trials} randomized trials...")
    
    for i in range(n_trials):
        key, subkey = jax.random.split(key)
        rmf, alpha = run_trial(subkey, n_values)
        all_rmf_errors.append(rmf)
        all_alpha_errors.append(alpha)
        
    all_rmf_errors = jnp.stack(all_rmf_errors) # (Trials, N_vals)
    all_alpha_errors = jnp.stack(all_alpha_errors)
    
    mean_rmf = jnp.mean(all_rmf_errors, axis=0)
    std_rmf = jnp.std(all_rmf_errors, axis=0)
    
    mean_alpha = jnp.mean(all_alpha_errors, axis=0)
    std_alpha = jnp.std(all_alpha_errors, axis=0)
    
    # --- Plot 1: Uncorrected RMF Error ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.loglog(n_values, mean_rmf, 'o-', label='Mean Max Angular Error')
    ax1.fill_between(n_values, mean_rmf - std_rmf, mean_rmf + std_rmf, alpha=0.2)
    
    ax1.set_xlabel('Samples N')
    ax1.set_ylabel('Error (deg)')
    ax1.set_title('Uncorrected RMF Approximation Error vs. Sampling')
    ax1.grid(True, which="both", ls="-")
    ax1.legend()
    plt.savefig('rmf_error_plot.png')
    print("Saved rmf_error_plot.png")
    
    # --- Plot 2: Holonomy Mismatch Error ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.loglog(n_values, mean_alpha, 's-', color='orange', label='Mean Holonomy Angle Error')
    ax2.fill_between(n_values, mean_alpha - std_alpha, mean_alpha + std_alpha, color='orange', alpha=0.2)
    
    ax2.set_xlabel('Samples N')
    ax2.set_ylabel('Error (deg)')
    ax2.set_title('Holonomy Mismatch Calculation Error vs. Sampling')
    ax2.grid(True, which="both", ls="-")
    ax2.legend()
    plt.savefig('holonomy_error_plot.png')
    print("Saved holonomy_error_plot.png")

if __name__ == "__main__":
    main()
