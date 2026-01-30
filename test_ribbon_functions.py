
import jax
import jax.numpy as jnp
import numpy as np
import unittest
from splinebox.jax_spline_curves import JaxSpline
from splinebox.jax_basis_functions import JaxB3

class TestRibbonSpline(unittest.TestCase):
    def setUp(self):
        self.M = 10
        self.basis = JaxB3()
        # Create 4D control points: 3D circle + varying theta
        t = np.linspace(0, 2*np.pi, self.M, endpoint=False)
        x = np.cos(t)
        y = np.sin(t)
        z = np.zeros_like(t)
        theta = t * 0.5 # Some twist
        
        self.points_3d = np.stack([x, y, z], axis=1)
        self.points_4d = np.stack([x, y, z, theta], axis=1)
        
        self.spline_3d = JaxSpline(self.M, self.basis, closed=True, control_points=self.points_3d)
        self.spline_4d = JaxSpline(self.M, self.basis, closed=True, control_points=self.points_4d, is_ribbon=True, ribbon_width=1.0)
        
    def test_length(self):
        # Length should be calculated based on x,y,z only
        len_3d = self.spline_3d.arc_length()
        len_4d = self.spline_4d.arc_length()
        
        print(f"Length 3D: {len_3d}")
        print(f"Length 4D: {len_4d}")
        
        self.assertAlmostEqual(len_3d, len_4d, places=4, msg="Length should ignore theta")

    def test_centroid(self):
        cent_3d = self.spline_3d._control_points_centroid()
        cent_4d = self.spline_4d._control_points_centroid()
        
        print(f"Centroid 3D: {cent_3d.shape}")
        print(f"Centroid 4D: {cent_4d.shape}")
        
        self.assertEqual(cent_4d.shape, (3,))
        np.testing.assert_allclose(cent_3d, cent_4d, err_msg="Spatial centroid should match")
        
    def test_rotate(self):
        # Rotate 90 deg around value z
        rot_matrix = jnp.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        # Clone to not finish original
        s4d = self.spline_4d.copy()
        s4d.rotate(rot_matrix, centered=False)
        
        rotated_4d = s4d.control_points
        print(f"Rotated shape: {rotated_4d.shape}")
        
        # Check if theta is preserved
        np.testing.assert_allclose(rotated_4d[:, 3], self.points_4d[:, 3], err_msg="Theta should be preserved in rotation")

    def test_scale(self):
        scale = 2.0
        s4d = self.spline_4d.copy()
        s4d.scale(scale)
        
        scaled_4d = s4d.control_points
        
        original_theta = self.points_4d[:, 3]
        scaled_theta = scaled_4d[:, 3]
        
        np.testing.assert_allclose(original_theta, scaled_theta, err_msg="Theta should not be scaled")

    def test_centerline_theta(self):
        t = jnp.array([0.0, 1.0, 2.5])
        c = self.spline_4d.centerline(t)
        th = self.spline_4d.theta(t)
        
        self.assertEqual(c.shape, (3, 3))
        self.assertEqual(th.shape, (3,))
        
        # Check values against 3D spline
        c_3d = self.spline_3d(t)
        np.testing.assert_allclose(c, c_3d, err_msg="Centerline should match 3D spline")

    def test_ribbon_surface(self):
        t = 0.5
        u = 0.0
        # u=0 should be centerline
        surf_pt = self.spline_4d.ribbon_surface(t, u)
        c_pt = self.spline_4d.centerline(t)
        np.testing.assert_allclose(surf_pt, c_pt, err_msg="u=0 should be centerline")
        
        # u=1 should be dist w/2
        u = 1.0
        surf_pt_edge = self.spline_4d.ribbon_surface(t, u)
        dist = np.linalg.norm(surf_pt_edge - c_pt)
        self.assertAlmostEqual(dist, 0.5 * 1.0, places=5)
        
    def test_distance(self):
        # Test point exactly on centerline
        pt = self.spline_4d.centerline(0.5)
        dist = self.spline_4d.distance(pt)
        self.assertAlmostEqual(dist, 0.0, places=4)
        
        # Test point on edge
        edge_pt = self.spline_4d.ribbon_surface(0.5, 1.0)
        dist_edge = self.spline_4d.distance(edge_pt)
        self.assertAlmostEqual(dist_edge, 0.0, places=4)
        
    def test_mesh(self):
        # Just check it runs for ribbon
        pts, conn = self.spline_4d.mesh(u_resolution=5, step_t=0.5)
        print(f"Mesh points: {pts.shape}, Conn: {conn.shape}")
        
    def test_control_point_setter_promotion(self):
        # Passing 3D points to ribbon spline should promote to 4D with theta=0
        s = JaxSpline(self.M, self.basis, closed=True, control_points=self.points_3d, is_ribbon=True, ribbon_width=1.0)
        self.assertEqual(s.control_points.shape, (self.M, 4))
        np.testing.assert_allclose(s.control_points[:, 3], 0.0)

if __name__ == '__main__':
    unittest.main()
