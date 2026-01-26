
import jax
import jax.numpy as jnp
import numpy as np
from splinebox.jax_spline_curves_2d import JaxSpline
from splinebox.jax_basis_functions import JaxB3

def run_test(seed, test_id):
    # Create a closed spline with torsion (3D)
    M = 10
    spline = JaxSpline(M=M, basis_function=JaxB3(), closed=True)
    
    # Random 3D control points
    key = jax.random.PRNGKey(seed)
    cps = jax.random.uniform(key, (M, 3), minval=-2.0, maxval=2.0)
    
    # Add some structure to ensure it's a loop and has torsion (perturb a basic loop)
    # We want varied shapes, so we add random perturbations to a base circle/ellipse+wave
    t = np.linspace(0, 2*np.pi, M, endpoint=False)
    x = np.sin(t) * 2.0
    y = np.cos(t) * 2.0
    z = np.sin(3*t) * 1.0 # Significant torsion
    base_shape = jnp.column_stack([x, y, z])
    
    spline.control_points = base_shape + cps * 0.5
    
    # Dense sampling
    # We rely on the internal correction which uses a dense grid.
    # We verify on another dense grid.
    t_dense = jnp.linspace(0, M, 1000)
    
    try:
        frames = spline.moving_frame(t_dense, method="bishop")
    except Exception as e:
        print(f"Test {test_id} (Seed {seed}): ERROR - Exception during moving_frame: {e}")
        return False

    f0 = frames[0]
    f_end = frames[-1]
    
    # Check alignment
    u0 = f0[1]
    u_end = f_end[1]
    dot_u = jnp.dot(u0, u_end)
    
    # Check Curvature
    # Torsion integral relates to Bishop frame rotation only if Frenet frame is well-defined everywhere (kappa > 0).
    # If kappa passes through 0 (inflection point), Frenet frame flips, causing a jump in the angle relation.
    # curvature = |r' x r''| / |r'|^3
    d1 = spline(t_dense, derivative=1)
    d2 = spline(t_dense, derivative=2)
    cross = jnp.cross(d1, d2)
    kappa_num = jnp.linalg.norm(cross, axis=-1)
    speed = jnp.linalg.norm(d1, axis=-1)
    kappa = kappa_num / (speed**3 + 1e-12)
    
    print(f"Min Curvature: {jnp.min(kappa)}")
    print(f"Max Curvature: {jnp.max(kappa)}")
    print(f"Points with curvature < 1e-3: {jnp.sum(kappa < 1e-3)}")
    
    if jnp.any(kappa < 1e-4):
        print("WARNING: Curve has near-zero curvature (inflection points). Torsion integral may be invalid due to Frenet frame flips.")
        
    start_mismatch = jnp.arccos(jnp.clip(dot_u, -1, 1))
    print(f"Measured Bishop Mismatch: {np.degrees(start_mismatch)} deg ({start_mismatch} rad)")
    
    success = dot_u > 0.99
    
    return success

def main():
    n_tests = 20
    failures = 0
    print(f"Running {n_tests} randomized tests...")
    
    for i in range(n_tests):
        if not run_test(42 + i, i):
            failures += 1
            
    print("-" * 30)
    if failures == 0:
        print(f"ALL {n_tests} TESTS PASSED.")
    else:
        print(f"{failures}/{n_tests} TESTS FAILED.")
        exit(1)

if __name__ == "__main__":
    main()
