
import jax
import jax.numpy as jnp
import numpy as np
from splinebox.jax_basis_functions import JaxB3
from splinebox.jax_spline_curves import JaxSpline

def test_jax_spline():
    print("Testing JaxSpline...")
    M = 5
    b3 = JaxB3()
    spline = JaxSpline(M, b3, closed=True)
    
    # Set control points
    # Circle in 2D
    t = jnp.linspace(0, 2*np.pi, M, endpoint=False)
    cp = jnp.stack([jnp.sin(t), jnp.cos(t)], axis=1)
    spline.control_points = cp
    
    # Evaluate
    test_t = jnp.linspace(0, M, 10)
    vals = spline(test_t)
    print("Spline evaluation successful.")
    print("Values shape:", vals.shape)
    print("Values sample:", vals[:2])
    
    # Test derivatives
    d1 = spline(test_t, derivative=1)
    print("Derivative evaluation successful.")
    
    # Test arc length
    length = spline.arc_length()
    print("Arc length:", length)
    
    return vals, d1, length

if __name__ == "__main__":
    test_jax_spline()
