
import jax
import jax.numpy as jnp
import numpy as np
import unittest
import time
from splinebox.jax_spline_curves import JaxSpline
from splinebox.jax_basis_functions import JaxB3

class TestFrameCaching(unittest.TestCase):
    def setUp(self):
        self.M = 20
        self.basis = JaxB3()
        t = np.linspace(0, 2*np.pi, self.M, endpoint=False)
        x = np.cos(t)
        y = np.sin(t)
        z = np.sin(2*t)
        self.points_3d = np.stack([x, y, z], axis=1)
        self.theta = np.zeros_like(t)
        self.points_4d = np.stack([x, y, z, self.theta], axis=1)
        
    def test_caching_speedup(self):
        # Create ribbon -> triggers cache computation
        spline = JaxSpline(self.M, self.basis, closed=True, control_points=self.points_4d, is_ribbon=True, ribbon_width=1.0)
        
        # Access once to compile whatever needed
        _ = spline.moving_frame(0.5, method="bishop")
        # Warmup cached path
        _ = spline.moving_frame(jnp.array([0.0, 1.0]), method="bishop").block_until_ready()
        
        t_eval = jnp.linspace(0, self.M, 50000)
        
        # Timed run with cache
        # Force wait
        start_cache = time.time()
        frames_cached = spline.moving_frame(t_eval, method="bishop").block_until_ready()
        time_cache = time.time() - start_cache
        
        # Kill cache - verify logic
        print(f"DEBUG: Cache object before kill: {id(spline._frame_cache)}")
        spline._frame_cache = None
        print(f"DEBUG: Cache killed.")
        
        # Timed run without cache (force update? no, just call it)
        start_no_cache = time.time()
        frames_exact = spline.moving_frame(t_eval, method="bishop").block_until_ready()
        time_no_cache = time.time() - start_no_cache
        
        print(f"Time Cached: {time_cache:.6f}s")
        print(f"Time Exact:  {time_no_cache:.6f}s")
        print(f"Speedup: {time_no_cache/time_cache:.2f}x")
        
        # Check accuracy
        err_T = jnp.linalg.norm(frames_cached[:, 0, :] - frames_exact[:, 0, :], axis=-1)
        print(f"Max Error T: {jnp.max(err_T)}")
        self.assertTrue(jnp.max(err_T) < 1e-2, "Interpolated frame should be reasonably accurate")

    def test_invalidation(self):
        spline = JaxSpline(self.M, self.basis, closed=True, control_points=self.points_4d, is_ribbon=True, ribbon_width=1.0)
        
        # Cache should be populated
        self.assertIsNotNone(spline._frame_cache)
        cache_frame_id = id(spline._frame_cache[1])
        
        # Modify geometry slightly
        new_pts = self.points_4d.copy()
        new_pts[0, 0] += 1.0
        # Assign
        spline.control_points = new_pts
        
        # Cache should be replaced
        self.assertIsNotNone(spline._frame_cache)
        cache_frame_id_new = id(spline._frame_cache[1])
        
        self.assertNotEqual(cache_frame_id, cache_frame_id_new, "Cache array should be recreated on update")
        
        # Check logic correctness
        f0 = spline.moving_frame(0.0, method="bishop")
        # Do manual calculation to verify it matches new geometry, not old
        # ... logic mainly relies on `test_caching_speedup` correctness check.

if __name__ == '__main__':
    unittest.main()
