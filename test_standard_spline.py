
import jax
import jax.numpy as jnp
import numpy as np
import unittest
from splinebox.jax_spline_curves import JaxSpline
from splinebox.jax_basis_functions import JaxB3

class TestStandardSpline(unittest.TestCase):
    def setUp(self):
        self.M = 50
        self.basis = JaxB3()
        # Create 3D control points: Circle
        t = np.linspace(0, 2*np.pi, self.M, endpoint=False)
        x = np.cos(t)
        y = np.sin(t)
        z = np.zeros_like(t)
        
        self.points_3d = np.stack([x, y, z], axis=1)
        self.spline = JaxSpline(self.M, self.basis, closed=True, control_points=self.points_3d, is_ribbon=False)
        
    def test_init(self):
        self.assertFalse(self.spline.is_ribbon)
        self.assertEqual(self.spline.control_points.shape, (50, 3))
        
    def test_length(self):
        # Circle length approx 2*pi*r = 6.28
        length = self.spline.arc_length()
        print(f"Standard Spline Length: {length}")
        self.assertAlmostEqual(length, 2*np.pi, places=1) # Low places due to M=10 discretization

    def test_distance(self):
        # Point at origin -> distance should be radius = 1.0
        dist = self.spline.distance(np.array([0.0, 0.0, 0.0]))
        print(f"Distance to origin: {dist}")
        self.assertAlmostEqual(dist, 1.0, places=2)
        
        # Point on curve
        pt = self.spline(0.5)
        dist_zero = self.spline.distance(pt)
        self.assertAlmostEqual(dist_zero, 0.0, places=4)

    def test_tube_mesh(self):
        radius = 0.2
        # Should generate a tube
        pts, conn = self.spline.mesh(radius=radius, step_t=0.5, step_angle=45)
        print(f"Tube Mesh - Pts: {pts.shape}, Conn: {conn.shape}")
        
        # Check basic properties
        # step_angle=45 -> 360/45 = 8 points around
        # (N_t) * 8 points approx
        # Conn should be triangles
        self.assertEqual(conn.shape[1], 3)
        self.assertTrue(pts.shape[0] > 0)

if __name__ == '__main__':
    unittest.main()
