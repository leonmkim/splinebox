#%%
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import math
from functools import partial
import collections
import json
import copy
from jax.tree_util import register_pytree_node_class

from quadax import quadgk # gauss-kronrod integration in JAX, same used by scipy.integrate.quad
from splinebox.jax_basis_functions import JaxBasisFunction, JaxB3, basis_function_from_name
#%%

# --- Main Class ---
@register_pytree_node_class
class JaxSpline:
    """
    JAX-compatible Spline class.
    Heavy computations are delegated to static JIT functions.
    """
    _wrong_dimension_msg = "It looks like control_points is a 2D array with second dimension different than two. I don't know how to handle this yet."
    _wrong_array_size_msg = (
        "It looks like control_points is neither a 1 nor a 2D array. I don't know how to handle this yet."
    )
    _no_control_points_msg = "This spline object doesn't have any control points yet."
    _unimplemented_msg = "This function is not implemented."

    def __init__(self, 
                 M: int, 
                 basis_function: JaxBasisFunction, 
                # basis_function_name: str,
                 closed=True, 
                 control_points=None, 
                 padding_function=None,
                 is_ribbon: bool = False,
                 ):
        # basis_function = JaxB3()
        if basis_function.support <= M:
            self.M = M
        else:
            raise RuntimeError("M must be greater or equal than the spline generator support size.")

        self.basis_function = basis_function
        self._half_support = self.basis_function.support / 2
        self._pad = math.ceil(self._half_support) - 1
        if not closed: 
            raise NotImplementedError("Open splines are not implemented yet in JaxSpline.")
        self.closed = closed
        self.is_ribbon = is_ribbon

        
        # JAX arrays preferred
        self._control_points = control_points
        if control_points is not None:
            # if not isinstance(control_points, jnp.ndarray):
            if isinstance(control_points, (list, tuple, np.ndarray)):
                control_points = jnp.array(control_points)
            
            self._control_points = control_points
        
        if padding_function is None:
            padding_function = self._padding_function
        self.padding_function = padding_function

    # #####################################
    # Static JIT-compiled methods
    # #####################################

    @staticmethod
    @jax.jit
    def _padding_function(knots, pad_length):
        """JIT-able padding function."""
        if knots.ndim == 1:
            knots = knots[:, None]
        # jnp.pad works identically to np.pad
        return jnp.pad(knots, ((pad_length, pad_length), (0, 0)), mode="edge")

    @staticmethod
    @partial(jax.jit, static_argnames=['M', 'half_support', 'closed'])
    def _get_tval(t, M, half_support, pad, closed):
        """
        Calculates the t-values relative to knot indices. 
        Replaces _wrap_index logic with vectorized operations.
        """
        t = jnp.atleast_1d(t)
        
        if closed:
            k = jnp.arange(M)
            # We broadcast t against k to get a matrix (len(t), M)
            # t is (N, 1), k is (1, M)
            ts = t[:, None]
            ks = k[None, :]
            
            # Logic from _wrap_index vectorized:
            # Case 1: Close to end, wrapping to beginning
            cond1 = ts >= (M + ks - half_support)
            val1 = ts - M - ks
            
            # Case 2: Close to beginning, wrapping from end
            cond2 = ts <= (half_support - (M - ks))
            val2 = ts + M - ks
            
            # Case 3: Outside support
            # Note: In JAX we usually don't use nan for "outside", strictly speaking,
            # but if the basis function handles bounds checking, we can leave it or set to a dummy.
            # Here we follow the logic:
            cond3 = (ts > ks + half_support) | (ts < ks - half_support)
            val3 = half_support + 1.0 # "outside_tvalue"
            
            # Case 4: Normal
            val4 = ts - ks
            
            # Combine using select/where
            # We check conditions in priority order
            tval = jnp.select(
                [cond1, cond2, cond3],
                [val1, val2, val3],
                default=val4
            )
            return tval
        else:
            raise NotImplementedError("Open splines are not implemented yet in JaxSpline.")
            # Non-closed case is much simpler
            k = jnp.arange(M + 2 * pad) - pad
            return t[:, None] - k[None, :]

    @staticmethod
    @jax.jit
    def _eval_spline(basis_vals, control_points):
        """Pure JIT-able matrix multiplication for spline evaluation."""
        return jnp.matmul(basis_vals, control_points)

    @staticmethod
    @jax.jit
    def _fit_spline(basis_vals, points):
        """Least squares fitting using JAX's linear algebra solver."""
        return jnp.linalg.lstsq(basis_vals, points, rcond=None)[0]

    @staticmethod
    @jax.jit
    def _moving_frame_frenet(T, normal_approx):
        '''
        Docstring for _moving_frame_frenet
        
        :param T: normalized tangent vectors
        :param normal_approx: second derivative vectors (not necessarily normalized)
        
        '''
        # Binormal = T x N_approx
        binormals = jnp.cross(T, normal_approx)
        binormal_norms = jnp.linalg.norm(binormals, axis=-1, keepdims=True)
        
        binormal_normalized = binormals / binormal_norms
        
        normals = jnp.cross(binormal_normalized, T)
        
        frames = jnp.stack([T, normals, binormal_normalized], axis=1)
        return frames, binormal_norms

    @staticmethod
    @jax.jit
    def _reflect(vec, axis):
        """
        Reflects vector 'vec' across the hyperplane defined by normal vector 'axis'.
        R(x) = x - (2 / (axis . axis)) * (axis . x) * axis
        """
        c = jnp.dot(axis, axis)
        scale = jnp.where(c > 1e-16, 2.0/c, 0.0)
        return vec - scale * jnp.dot(axis, vec) * axis



    @staticmethod
    @partial(jax.jit, static_argnames=['closed'])
    def _moving_frame_bishop(points, T, initial_vector, t_sorted, M, closed):
        '''
        Computes Bishop frame using the Double Reflection method (RMF).
        If closed=True, applies topological correction assuming t_sorted covers [0, M].
        
        :param points: positions on the curve (N, 3)
        :param T: normalized tangent vectors (N, 3)
        :param initial_vector: Initial normal vector guess
        :param t_sorted: (N,) array of parameter values
        :param M: Spline domain limit
        :param closed: Boolean, whether the spline is closed
        :return: Frames (N, 3, 3) [T, N, B]
        '''
        # Ensure orthogonality of provided/calculated initial vector
        t0_T = T[0]
        initial_vector = initial_vector - t0_T * jnp.dot(t0_T, initial_vector)
        initial_vector_norm = jnp.linalg.norm(initial_vector)
        # Avoid division by zero
        initial_vector = initial_vector / (initial_vector_norm + 1e-12)

        # We only track Tangent and Normal
        init_frame = (t0_T, initial_vector) 
        
        # Carry: (prev_pos, prev_frame)
        init_carry = (points[0], init_frame)

        # Slice inputs to iterate from 1 to N
        points_slice = points[1:]
        T_slice = T[1:]

        def scan_body(carry, inputs):
            prev_pos, prev_frame = carry
            curr_pos, curr_T = inputs
            
            prev_t, prev_r = prev_frame # r=normal
            
            # Double Reflection Algorithm
            # Step 1: Reflect across bisector of chord
            v1 = curr_pos - prev_pos
            
            r_L = JaxSpline._reflect(prev_r, v1)
            t_L = JaxSpline._reflect(prev_t, v1)
            
            # Step 2: Reflect across bisector of tangents
            v2 = curr_T - t_L
            
            current_r = JaxSpline._reflect(r_L, v2)
            
            # Re-orthogonalize to allow for numerical drift if needed, 
            # but Double Reflection is generally orthogonal.
            # Make sure current_r is orthogonal to curr_T
            current_r = current_r - jnp.dot(curr_T, current_r) * curr_T
            current_r = current_r / (jnp.linalg.norm(current_r) + 1e-12)
            
            current_frame = (curr_T, current_r)
            new_carry = (curr_pos, current_frame)
            
            return new_carry, current_r

        _, normals_rest = jax.lax.scan(scan_body, init_carry, (points_slice, T_slice))
        
        # Prepend initial normal
        normals = jnp.concatenate([initial_vector[None, :], normals_rest], axis=0)
        
        if closed:
            # Correction logic for closed curves
            # We measure mismatch between f_start and f_end.
            
            v = normals[0] 
            u1 = normals[-1]
            # We need binormal at end to determine sign
            u2 = jnp.cross(T[-1], u1)
            
            c = jnp.dot(v, u1)
            s = jnp.dot(v, u2)
            alpha = jnp.arctan2(s, c)
            
            # Distribute alpha based on parameter t
            t_normalized = t_sorted / M
            correction_angles = alpha * t_normalized
            
            # Apply rotation to Normals around Tangent
            cos_theta = jnp.cos(correction_angles)[:, None]
            sin_theta = jnp.sin(correction_angles)[:, None]
            
            Bs_temp = jnp.cross(T, normals)
            normals = normals * cos_theta + Bs_temp * sin_theta
            
        # Compute Binormals
        binormals = jnp.cross(T, normals)
        frames = jnp.stack([T, normals, binormals], axis=1)
        
        return frames

    @staticmethod
    @jax.jit
    def _compute_pairwise_distances(points_a, points_b):
        '''
        Docstring for _compute_distances
        
        :param points_a: N_axdim array
        :param points_b: N_bxdim array
        :return: N_a x N_b array of distances
        '''
        return jnp.linalg.norm(points_a[:, None] - points_b[None], axis=-1)

    @staticmethod
    @jax.jit
    def _compute_closest_indices_between_points(curve_points, points):
        '''
        Docstring for _compute_closest_point_on_curve
        
        :param points: N_pxD array of points
        :param curve_points: N_cxD array of curve points
        :return: N_p array of indices of closest curve points
        '''
        dists = JaxSpline._compute_pairwise_distances(curve_points, points)
        closest_indices = jnp.argmin(dists, axis=0)
        return closest_indices

    @staticmethod
    @jax.jit
    def _compute_control_point_centroid(control_points):
        return jnp.mean(control_points, axis=0)

    @staticmethod
    @jax.jit
    def _compute_control_points_translated(control_points, vector):
        return control_points + vector

    @staticmethod
    @jax.jit
    def _compute_control_points_scaled(control_points, scaling_factor):
        centroid = JaxSpline._compute_control_point_centroid(control_points)
        centered_control_points = JaxSpline._compute_control_points_translated(control_points, -centroid)
        scaled_centered = centered_control_points * scaling_factor
        return JaxSpline._compute_control_points_translated(scaled_centered, centroid)

    @staticmethod
    @jax.jit
    def _compute_control_points_rotated(control_points, rotation_matrix):
        def rotate_point(rotation_matrix, control_point):
            rotated_point = jnp.matmul(rotation_matrix, control_point)
            return rotation_matrix, rotated_point
            
        _, rotated_control_points = jax.lax.scan(rotate_point, rotation_matrix, control_points)
        return rotated_control_points
    
    @staticmethod
    @jax.jit
    def _initial_vector_guess(t0_T, guess, guess_is_degen):
        # Estimate initial vector
        def alternate_guess(t0_T):
            guess_alt = jnp.zeros(3)
            max_axis = jnp.argmax(jnp.abs(t0_T))
            other_axis = (max_axis + 1) % 3
            guess_alt = guess_alt.at[max_axis].set(t0_T[other_axis])
            guess_alt = guess_alt.at[other_axis].set(-t0_T[max_axis])
            return guess_alt
        guess = lax.cond(
            guess_is_degen,
            lambda x: alternate_guess(x),
            lambda _: guess,
            t0_T
        )
        return guess / (jnp.linalg.norm(guess) + 1e-12)

    @staticmethod
    def _convert_to_array(t):
        is_single = False
        if isinstance(t, (int, float)):
            t = jnp.array([t])
            is_single = True
        elif isinstance(t, (list, tuple)):
            t = jnp.array(t)
        elif hasattr(t, 'shape') and t.shape == ():
            t = jnp.array([t])
            is_single = True
        elif not isinstance(t, jnp.ndarray):
            t = jnp.array(t)
        if t.ndim > 1:
            raise ValueError("t must be 1D array-like.")
        return t, is_single

    @staticmethod
    @partial(jax.jit, static_argnames=['single_val'])
    def _compute_spline_first_deriv(tval, control_points, single_val=False):
        basis_vals_d1 = JaxB3._derivative_1(tval)
        d1 = JaxSpline._eval_spline(basis_vals_d1, control_points)
        if single_val:
            d1 = d1[0]
        return d1

    @staticmethod
    @jax.jit
    def _tangent_vector_to_speed(d1):
        """Helper function to compute differential length element."""
        safe_d1 = jnp.nan_to_num(d1)
        speed = jnp.linalg.norm(safe_d1, axis=-1)
        return speed

    @staticmethod
    @partial(jax.jit, static_argnames=['M', 'half_support', 'closed', 'max_ninter'])
    def _compute_curve_length(
                                control_points, 
                                M, half_support, pad, closed,
                                start, stop, 
                                epsabs, epsrel, max_ninter
                                ):
        integral = quadgk(
            lambda t: JaxSpline._tangent_vector_to_speed(
                JaxSpline._compute_spline_first_deriv(
                    JaxSpline._get_tval(jnp.atleast_1d(t), M, half_support, pad, closed), control_points,
                )
            ),
            [start, stop],
            epsabs=epsabs,
            epsrel=epsrel,
            max_ninter=max_ninter,
        )
        return integral[0]

    @staticmethod
    @jax.jit
    def _curvilinear_integrand(speed, c):
        return (speed**2 - c)**2

    @staticmethod
    @partial(jax.jit, static_argnames=['M', 'half_support', 'closed', 'max_ninter'])
    def _compute_curvilinear_reparametrization_energy(
                                L, 
                                control_points, 
                                M, half_support, pad, closed,
                                start, stop, 
                                epsabs, epsrel, max_ninter
                                ):
        c = (L/M)**2
        integral = quadgk(
            lambda t: JaxSpline._curvilinear_integrand(
                JaxSpline._tangent_vector_to_speed(
                    JaxSpline._compute_spline_first_deriv(
                        JaxSpline._get_tval(jnp.atleast_1d(t), M, half_support, pad, closed), control_points, single_val=True,
                    )
                ), c),
            [start, stop],
            epsabs=epsabs,
            epsrel=epsrel,
            max_ninter=max_ninter,
        )
        return integral[0] / (L**4)

    @staticmethod
    @partial(jax.jit, static_argnames=['M', 'half_support', 'closed'])
    def _compute_spline_val_single_val(t, control_points, 
                                       M, half_support, pad, closed
                                       ):
        t = jnp.atleast_1d(t)
        tval = JaxSpline._get_tval(t, M, half_support, pad, closed)
        basis_vals = JaxB3._func(tval)
        return JaxSpline._eval_spline(basis_vals, control_points)

    @staticmethod
    @partial(jax.jit, static_argnames=['M', 'half_support', 'closed'])
    def _spline_distance_to_point(t, point, control_points,
                                M, half_support, pad, closed,
                                ):
        spline_point = JaxSpline._compute_spline_val_single_val(t, control_points, M, half_support, pad, closed)
        return jnp.sum((spline_point - point)**2)

    @staticmethod
    @partial(jax.jit, static_argnames=['M', 'half_support', 'closed'])
    @partial(jax.vmap, in_axes=(0, 0, None, None, None, None, None, None, None))
    def _projected_newton_step(t, point, 
                               control_points, min_t_bound, max_t_bound, M, half_support, pad, closed
                               ):
        grad_fn = jax.grad(JaxSpline._spline_distance_to_point, argnums=0) # only wrt t
        hess_fn = jax.hessian(JaxSpline._spline_distance_to_point, argnums=0) # only wrt t
        grad = grad_fn(t, point, control_points, M, half_support, pad, closed)
        hess = hess_fn(t, point, control_points, M, half_support, pad, closed)
        # Newton step
        step = -grad / (hess + 1e-12)
        new_tval = t + step
        # Projected to bounds
        new_tval = jnp.clip(new_tval, min_t_bound, max_t_bound)
        return new_tval

    @staticmethod
    @partial(jax.jit, static_argnames=['M', 'half_support', 'closed', 'max_iters'])
    def _find_closest_t_to_point(
        t_inits, points,
        control_points,
        M, half_support, pad, closed,
        min_t_bound, max_t_bound,
        max_iters
        ):
        def scan_body_fn(curr_ts, _):
            t = JaxSpline._projected_newton_step(curr_ts, points, control_points, min_t_bound, max_t_bound,
                                        M, half_support, pad, closed)
            return t, None
        t_final, _ = lax.scan(scan_body_fn, t_inits, None, length=max_iters)
        return t_final

    @staticmethod
    @partial(jax.jit, static_argnames=['closed'])
    def _generate_connectivity(closed, n_angles, n_t, n_points):
        raise NotImplementedError("Non-zero radius mesh not implemented yet.")
        """
        Generates mesh connectivity using broadcasting instead of loops.
        Replaces the Numba functions.
        """
        # Create grid of indices (i, j)
        # i goes from 0 to n_t-1 (or n_t if closed)
        # j goes from 0 to n_angles-1
        
        limit_t = n_t if closed else n_t - 1
        i_grid, j_grid = jnp.meshgrid(jnp.arange(limit_t), jnp.arange(n_angles), indexing='ij')
        
        # Flatten for processing
        i = i_grid.flatten()
        j = j_grid.flatten()
        
        # --- Surface Mesh (Triangles) ---
        # Triangle 1: (i, j), ((i+1), j), ((i+1), (j+1))
        p1 = i * n_angles + j
        p2 = ((i + 1) * n_angles + j) % n_points
        p3 = ((i + 1) * n_angles + (j + 1) % n_angles) % n_points
        
        # Triangle 2: (i, j), ((i+1), (j+1)), (i, (j+1))
        p4 = i * n_angles + j
        p5 = ((i + 1) * n_angles + (j + 1) % n_angles) % n_points
        p6 = (i * n_angles + (j + 1) % n_angles) % n_points
        
        t1 = jnp.stack([p1, p2, p3], axis=1)
        t2 = jnp.stack([p4, p5, p6], axis=1)
        
        # Interleave t1 and t2
        surface_conn = jnp.empty((t1.shape[0] * 2, 3), dtype=jnp.int32)
        surface_conn = surface_conn.at[0::2].set(t1)
        surface_conn = surface_conn.at[1::2].set(t2)
        
        # --- Volume Mesh (Tetrahedra) ---
        # Indices logic adapted from original code
        # i * (n_angles + 1) + j ...
        row_len = n_angles + 1
        
        # Recalculate p indices for volume (different stride)
        v_p1 = i * row_len + j + 1 # offset j by 1 because j starts 0 here but 1 in original loop?
        # Actually, let's stick to the logic: j goes 0..n_angles-1, mapped to 1..n_angles
        jj = j + 1
        
        curr_row = i * row_len
        next_row = (i + 1) * row_len
        
        # Tet 1
        tp1 = curr_row + jj
        tp2 = (next_row + jj) % n_points
        tp3 = (next_row + 1 + (jj % n_angles)) % n_points
        tp4 = (next_row) % n_points
        
        # Tet 2
        tp5 = curr_row + jj
        tp6 = (next_row + 1 + (jj % n_angles)) % n_points
        tp7 = (curr_row + 1 + (jj % n_angles)) % n_points
        tp8 = curr_row
        
        # Tet 3
        tp9 = curr_row + jj
        tp10 = (next_row + 1 + (jj % n_angles)) % n_points
        tp11 = (next_row) % n_points
        tp12 = curr_row
        
        # Stack
        tet1 = jnp.stack([tp1, tp2, tp3, tp4], axis=1)
        tet2 = jnp.stack([tp5, tp6, tp7, tp8], axis=1)
        tet3 = jnp.stack([tp9, tp10, tp11, tp12], axis=1)
        
        vol_conn = jnp.empty((tet1.shape[0] * 3, 4), dtype=jnp.int32)
        vol_conn = vol_conn.at[0::3].set(tet1)
        vol_conn = vol_conn.at[1::3].set(tet2)
        vol_conn = vol_conn.at[2::3].set(tet3)
        
        return surface_conn, vol_conn 

    @staticmethod
    @partial(jax.jit, static_argnames=['self_ndim', 'single_val'])
    def _curvature(d1, d2, self_ndim, single_val):
        # Handle 1D case (codomain dimension = 1)
        # If ndim=1, we treat it as a graph y=f(x) where x=t.
        # r(t) = [t, y(t)] -> r'(t) = [1, y'(t)], r''(t) = [0, y''(t)]
        if self_ndim == 1:
            # d1 is (N, 1) -> stack with ones -> (N, 2)
            d1 = jnp.hstack([jnp.ones_like(d1), d1])
            d2 = jnp.hstack([jnp.zeros_like(d2), d2])
            
        norm_d1 = jnp.linalg.norm(d1, axis=-1)
        norm_d2 = jnp.linalg.norm(d2, axis=-1)

        # Compute numerator
        # We need to distinguish 2D (signed) vs ND (unsigned)
        
        # Helper for 2D signed curvature: x'y'' - y'x''
        def signed_2d_numerator(d1, d2):
            return d1[:, 1] * d2[:, 0] - d1[:, 0] * d2[:, 1]

        # Helper for ND unsigned curvature
        def unsigned_nd_numerator(d1, d2, norm_d1, norm_d2):
            dot = jnp.sum(d1 * d2, axis=-1)
            val = norm_d1**2 * norm_d2**2 - dot**2
            return jnp.sqrt(jnp.maximum(val, 0.0))

        # Check dimension (using shape of d1 which handles the 1D->2D promotion)
        dim = d1.shape[-1]
        
        numerator = lax.cond(
            dim == 2,
            lambda _: signed_2d_numerator(d1, d2),
            lambda _: unsigned_nd_numerator(d1, d2, norm_d1, norm_d2),
            None
        )
        
        # Avoid division by zero if velocity is zero
        denom = norm_d1 ** 3
        k = jnp.where(denom > 1e-12, numerator / denom, 0.0)
        
        if single_val:
            return k[0]
        return k

    @staticmethod
    @jax.jit
    def _2d_normal_helper(d1):
        # Rotate 90 degrees: [x, y] -> [-y, x]
        normals = jnp.stack([-d1[:, 1], d1[:, 0]], axis=1)
        normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
        return normals

    # #####################################
    # Instance Methods
    # #####################################

    def tree_flatten(self):
        children = (self.control_points,) # arrays / dynamic values
        aux_data = dict( # static values
            M=self.M,
            basis_function_name=str(self.basis_function),
            closed=self.closed,
        )
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        basis_function = basis_function_from_name(aux_data['basis_function_name'])
        return cls(aux_data["M"], basis_function, aux_data["closed"], children[0])

    def _check_control_points(self):
        if self.control_points is None:
            raise RuntimeError("Control points not set.")
    
    def __str__(self):
        closed_str = "closed" if self.closed else "open"
        if self.control_points is None:
            return f"uninitialized {closed_str} {self.basis_function} spline with {self.M} knots"
        else:
            return f"{closed_str} {self.ndim}D {self.basis_function} spline with {self.M} knots"

    def __repr__(self):
        return f"splinebox.jax_spline_curves.JaxSpline(M={repr(self.M)}, basis_function={repr(self.basis_function)}, closed={repr(self.closed)}, control_points=jnp.{repr(self.control_points)})"

    def __eq__(self, other):
        return (
            self.M == other.M
            and self.basis_function == other.basis_function
            and self.closed == other.closed
            and jnp.all(self.control_points == other.control_points)
        )
    
    @property
    def control_points(self):
        return self._control_points

    @control_points.setter
    def control_points(self, values):
        if values is not None:
            if isinstance(values, (list, tuple, np.ndarray)):
                values = jnp.array(values)
            n = len(values)
            if self.closed and n != self.M:
                raise ValueError(f"Closed spline: expected {self.M} CP, got {n}.")
            padded_M = self.M + 2 * self.pad
            if not self.closed and n != padded_M:
                raise ValueError(f"Open spline: expected {padded_M} CP, got {n}.")
        self._control_points = values

    @property
    def knots(self):
        if self.padding_function is None and not self.closed:
            t = jnp.arange(-self.pad, self.M + self.pad)
        else:
            t = jnp.arange(self.M)
        return self(t)

    @knots.setter
    def knots(self, values):
        knots = jnp.array(values)
        n = len(knots)
        
        if self.closed:
            if n != self.M:
                raise ValueError(f"Expected {self.M} knots, got {n}.")
            # Assume basis_function has a JIT-able filter_periodic
            self.control_points = self.basis_function.filter_periodic(knots)
        else:
            raise NotImplementedError("Open splines are not implemented yet in JaxSpline.")
            padded_M = self.M + 2 * self.pad
            if self.padding_function is None:
                if n != padded_M:
                    raise ValueError(f"Expected {padded_M} padded knots.")
            else:
                if n != self.M:
                    raise ValueError(f"Expected {self.M} knots but got {n}.")
                knots = self.padding_function(knots, self.pad)
                if len(knots) != padded_M:
                    raise ValueError(f"Padding function returned wrong size. Expected {padded_M}, got {len(knots)}.")
                
            knots = jnp.squeeze(knots)
            # Assume basis_function has a JIT-able filter_symmetric
            self.control_points = self.basis_function.filter_symmetric(knots)

    @property
    def basis_function(self):
        r"""
        The basis function :math:`\Phi` of the spline :ref:`(1) <theory:eq:1>`.
        Should be an object of a specific implementation of the
        abstract base class :class:`splinebox.basis_functions.BasisFunction`.

        Raises
        ------
        ValueError
            If the basis_function is meant for a Hermite spline.
        """
        return self._basis_function

    @basis_function.setter
    def basis_function(self, value):
        if value.multigenerator:
            raise ValueError(
                "You are trying to construct a Hermite spline using the ordinary `Spline` class. Use the `HermiteSpline` class instead."
            )
        self._basis_function = value

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, M):
        if hasattr(self, "control_points") and self.control_points is not None and M != self.M:
            # The has attribute is necessary because M is assigned before control_points in the constructor
            raise RuntimeError(
                "M cannot be changed after the control points were set. Create a new spline or set the control_points to None first."
            )
        self._M = M

    @property
    def closed(self):
        return self._closed

    @closed.setter
    def closed(self, closed):
        if hasattr(self, "control_points") and self.control_points is not None and closed != self.closed:
            # The has attribute is necessary because closed is assigned before control_points in the constructor
            raise RuntimeError(
                "closed cannot be changed after the control points were set. Create a new spline or set the control_points to None first."
            )
        self._closed = closed

    @property
    def half_support(self):
        return self._half_support

    @half_support.setter
    def half_support(self, _):
        raise RuntimeError("The half support is determined by the basis function and cannot be set by the user.")

    @property
    def pad(self):
        return self._pad

    @pad.setter
    def pad(self, _):
        raise RuntimeError(
            "The amount of necessary padding is automatically calculated based on the support of the basis function and cannot be changed."
        )
    
    @property
    def ndim(self):
        if self.control_points is None:
            raise RuntimeError("Spline not initialized.")
        elif self.control_points.ndim == 1:
            return 1
        else:
            return self.control_points.shape[-1]

    def copy(self):
        """
        Returns a deep copy of this spline.
        """
        return copy.deepcopy(self)

    def _to_dict(self, version):
        """
        Helper function that creates a dictionary
        representing the spline that can be saved as a json.
        This is implemented separately from :meth:`splinebox.spline_curves.Spline.to_json`
        to allow the :class:`splinebox.spline_curves.HermiteSpline` to inherit this
        conversion only adding the addiontion tangents.

        Paramters
        ---------
        version : int
            The version of the convertion for future compatibility.

        Returns
        -------
        dictionary_representation : dictionary
            A dictionary representation of the spline.
        """
        dictionary_representation = {
            "version": version,
            "M": self.M,
            "basis_function": str(self.basis_function),
            "closed": self.closed,
            "control_points": self.control_points.tolist(),
        }
        return dictionary_representation

    def to_json(self, path, version=1):
        r"""
        Saves the spline as a json file.

        Parameters
        ----------
        path : str or pathlib.Path
            The path where the json file should be saved.
        version : int
            The version of the json file. Default is latest version.

        Examples
        --------

        >>> spline = splinebox.Spline(M=3, basis_function=splinebox.B1(), closed=True)
        >>> spline.knots = np.array([[0.8, 1.2], [0.7, 1.5], [1.1, 0.3]])

        Save the spline to file...

        >>> path = path_to_some_directory / "spline.json"
        >>> spline.to_json(path)

        Let's take a look at the file...

        >>> print(open(path, "r").read())
        {
          "version": 1,
          "M": 3,
          "basis_function": "B1",
          "closed": true,
          "control_points": [
            [
              0.8,
              1.2
            ],
            [
              0.7,
              1.5
            ],
            [
              1.1,
              0.3
            ]
          ]
        }
        """
        with open(path, "w") as f:
            json.dump(self._to_dict(version), f, indent=2)

    @classmethod
    def from_json(cls, path):
        """
        Constructs a spline from a json file that was saved using
        :meth:`splinebox.spline_curves.Spline.to_json`.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the json file.

        Examples
        --------

        >>> splinebox.Spline.from_json(path_to_single_spline_json)
        splinebox.spline_curves.Spline(M=3, basis_function=splinebox.basis_functions.B1(), closed=True, control_points=np.array([[0.8, 1.2],
               [0.7, 1.5],
               [1.1, 0.3]]))

        """
        with open(path) as f:
            data = json.load(f)
        data = _prepared_dict_for_constructor(data)
        return cls(**data)
    
    def draw(self, x, y):
        """
        Computes whether a point is inside or outside a closed
        spline on a regular grid of points.

        Parameters
        ----------
        x : numpy array
            A 1D array containing the x values of the grid of points.
        y : numpy array
            A 1D array containing the y values of the grid of points.

        Returns
        -------
        drawing : numpy array
            A 2D numpy array of float values.
            The values indicates the following:
            0.0 -> pixel centre lies outside the closed spline.
            0.5 -> pixel centre lies on the spline.
            1.0 -> pixel centre lies inside the spline.

        Examples
        --------

        >>> import splinebox
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Construct a spline

        >>> spline = splinebox.Spline(M=3, basis_function=splinebox.B1(), closed=True)
        >>> spline.knots = np.array([[1, 1], [1, 3], [3, 2]])

        Compute the points along the spline to plot it as a line.

        >>> t = np.linspace(0, 3, 200)
        >>> vals = spline(t)

        Draw the spline as an image.

        >>> pixel_size = 0.1
        >>> pixel_centres_x = np.arange(0.5, 3.5 + pixel_size, pixel_size)
        >>> pixel_centres_y = np.arange(0.5, 3.5 + pixel_size, pixel_size)
        >>> drawing = spline.draw(pixel_centres_x, pixel_centres_y)
        >>> print(drawing.shape, drawing.dtype, np.unique(drawing))
        (31, 31) float64 [0.  0.5 1. ]

        Plot the line and the drawing.

        >>> extent = (pixel_centres_x.min() - pixel_size / 2,
        ...           pixel_centres_x.max() + pixel_size / 2,
        ...           pixel_centres_y.min() - pixel_size / 2,
        ...           pixel_centres_y.max() + pixel_size / 2,)
        >>> plt.imshow(drawing, cmap="Greys_r", extent=extent)  # doctest: +SKIP
        >>> plt.plot(vals[:, 0], vals[:, 1])  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP
        """
        raise NotImplementedError(self._unimplemented_msg)
    
    def dtheta(self, t): # skip as dont need it
        r"""
        Helper function for calculating the winding number.

        `dtheta` is the derivative of the polar coordinate :math:`\theta(t)`

        .. math::
            \theta(t) = arctan \left( \frac{y(t)}{x(t)} \right)

        Differentiation yields:

        .. math::
            \frac{d \theta}{dt} = \frac{1}{r^2} \left( x\frac{dy}{dt} - y\frac{dx}{dt} \right) \text{, where } r^2 = x^2 + y^2
        """
        raise NotImplementedError(self._unimplemented_msg)
    
    def is_inside(self, x, y): # skip as dont need it
        r"""
        Determines if a point with coordinates `x`, `y` is inside the spline.
        Only works for closed 2D curves.

        To determine whether a point is inside or outside the spline, the winding number
        is used:

        .. math::
            wind(\gamma, 0) = \frac{1}{2\pi} \oint_\gamma d\theta = \frac{1}{2\pi} \oint_\gamma \left( \frac{x}{r^2}dy - \frac{y}{r^2}dx \right).

        For a description of :math:`d\theta` check :meth:`splinebox.spline_curves.Spline.dtheta`.

        Parameters
        ----------
        x : numpy.ndarray or float
            x coordinate(s) of point(s)
        y : numpy.ndarray or float
            y coordinate(s) of point(s)

        Returns
        -------
        val : float
            1 if the point is inside, 0.5 if its on the curve and 0 if it is outside the curve.

        Examples
        --------

        >>> import splinebox
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Construct a spline

        >>> spline = splinebox.Spline(M=3, basis_function=splinebox.B1(), closed=True)
        >>> spline.knots = np.array([[1, 1], [1, 3], [3, 2]])

        >>> points = np.array([[2.5, 1.0], [1.0, 2.0], [2.0, 2.0]])
        >>> spline.is_inside(points[:, 0], points[:, 1])
        array([0. , 0.5, 1. ])

        >>> t = np.linspace(0, 3, 200)
        >>> vals = spline(t)
        >>> plt.plot(vals[:, 0], vals[:, 1])  # doctest: +SKIP
        >>> plt.scatter(points[:, 0], points[:, 1], marker="x")  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP
        """
        raise NotImplementedError(self._unimplemented_msg)

    def fit(self, points, arc_length_parameterization=False):
        """Fits spline to points using JAX linear algebra.
        points: array-like, shape (N, ndim)
        """
        if not isinstance(points, jnp.ndarray):
            points = jnp.array(points)
        
        if len(points) < 2*(self.M + 2 * self.pad):
            raise RuntimeError(
                f"You provided too few points. For a unique solution you need to provide at least 2*({self.M}+{2 * self.pad}) points to match the number of control points and tangents (including padding). You provided {len(points)} points. Consider providing more points or reducing the number of knots M."
            )
        
        if len(points) < self.M:
            raise RuntimeError(
                "You provided fewer data points than your spline has knots. For the fit to have a unique solution you need to provide at least as many data points as your spline has knots. Consider adding more data or reducing the number of knots M."
            )
        
        if arc_length_parameterization:
            raise NotImplementedError("Arc length param not implemented yet.")
        
        limit = self.M if self.closed else self.M - 1
        t = jnp.linspace(0, limit, len(points), endpoint=not self.closed)
        
        tval = self._get_tval(t, self.M, self._half_support, self._pad, self.closed) 
        basis_vals = self.basis_function(tval, derivative=0) 
        
        self.control_points = self._fit_spline(basis_vals, points)

    @partial(jax.jit, static_argnames=['stop', 'start'])
    def arc_length(self, stop=None, start=0, epsabs:float=0.0, epsrel:float=1e-3):
        """
        Computes arc length quadax gauss-konrod "quadgk" integration.
        """
        self._check_control_points()
        if stop is None:
            stop = self.M if self.closed else self.M - 1

        if start == stop:
            return 0.0
    
        if start > stop:
            start, stop = stop, start
        
        integral = self._compute_curve_length(
            self.control_points,
            self.M, self._half_support, self._pad, self.closed,
            start, stop,
            epsabs, epsrel, 100,
        )
        return integral[0]

    def arc_length_to_parameter(self, s, atol=1e-4):
        """
        Convert the arc length `s` to the coresponding value in parameter space.

        Parameters
        ----------
        s : float or np.array
            Length on curve.
        atol : float
            The ablsolute error tolerance.

        Retruns
        -------
        parameter : float or numpy array of floats
            The parameter value whos arc length distance is :code:`s` from the
            start of the spline.

        Examples
        --------
        >>> spline = splinebox.Spline(M=4, basis_function=splinebox.B3(), closed=False)
        >>> spline.control_points=np.array([2, 3, 2, 6, 1, 2])

        >>> spline.arc_length_to_parameter(2.2)  # doctest: +NUMBER
        2.12

        >>> spline.arc_length(0, 2.12)  # doctest: +NUMBER
        2.2
        """
        raise NotImplementedError(self._unimplemented_msg)
    
    @jax.jit
    def curvilinear_reparametrization_energy(self, epsabs=1e-6, epsrel=1e-6):
        """
        Computes the energy used to enforce equal knot spacing.
        Implements eq. 25 from [Jacob2004]

        Parameters
        ----------
        n_samples : int
            Number of sample points for the integration grid. 
            Higher values = higher accuracy. Default 501.

        Returns
        -------
        energy : float
        """
        # 1. Compute Arc Length (reusing our JIT-able arc_length)
        L = self.arc_length(epsabs=epsabs, epsrel=epsrel)
        
        # Avoid division by zero if L is effectively 0
        L = jnp.where(L < 1e-12, 1.0, L)
        
        upper_limit = self.M if self.closed else self.M - 1

        return self._compute_curvilinear_reparametrization_energy(
            L, 
            self.control_points, 
            self.M, self._half_support, self._pad, self.closed,
            0, upper_limit,
            epsabs, epsrel, 100,
        )
        
    @jax.jit
    def curvature(self, t):
        """
        Compute JIT-compatible curvature.
        Returns signed curvature for 2D, unsigned for others.
        """
        self._check_control_points()
        t_arr, single_val = self._convert_to_array(t)
        
        # First and Second derivatives
        d1 = self(t_arr, derivative=1)
        d2 = self(t_arr, derivative=2)
        
        return self._curvature(d1, d2, self.ndim, single_val)
    
    
    def normal(self, t, frame="bishop", initial_vector=None):
        self._check_control_points()
        t_arr, single = self._convert_to_array(t)
        
        if self.ndim == 2:
            d1 = self(t_arr, derivative=1)
            
            normals = self._2d_normal_helper(d1)
            if single: return normals[0]
            return normals
            
        elif self.ndim == 3:
            frames = self.moving_frame(t_arr, method=frame, initial_vector=initial_vector)
            # Frame structure is [Tangent, Normal, Binormal]
            normals = frames[:, 1, :]
            if single: return normals[0]
            return normals
        
        else:
            raise RuntimeError("Normal only for 2D/3D.")

    def moving_frame(self, t, method="frenet", initial_vector=None):
        """JIT-accelerated Moving Frame.
        Compute a moving frame (local orthonormal coordinate system) along the spline.

        This method computes either the Frenet-Serret frame or the Bishop frame [#bishop]_ for
        the spline. A moving frame [#movingframe]_ consists of three orthonormal basis vectors at
        each point on the curve. The Frenet-Serret frame is derived from the curve's
        derivatives but may twist around the curve. The Bishop frame eliminates
        this twist, providing a zero-torsion alternative.

        Parameters
        ----------
        t : np.array or float
            A 1D array of parameter values or a single parameter value at which to evaluate the frame.
        method : str, optional
            The type of moving frame to compute. Options are:

            - "frenet": The classical Frenet-Serret frame, based on tangent, normal, and binormal vectors.
            - "bishop": A twist-free frame that requires an initial orientation.

            Default is "frenet".
        initial_vector : np.array or None, optional
            For the Bishop frame, an initial vector that is orthogonal to the tangent
            vector at `t[0]`. This vector determines the initial orientation of the
            basis, which is propagated along the curve without twisting. If None,
            the method computes a suitable initial vector automatically. This
            parameter is ignored when :code:`method="frenet"`.

        Returns
        -------
        frame : np.array
            A 3D numpy array with shape `(len(t), 3, 3)`. The dimensions are:

            - The first axis corresponds to the parameter values in `t`.
            - The second axis contains the three basis vectors at each `t`:
              [tangent, normal, binormal] for "frenet" or equivalent vectors for "bishop".
            - The third axis contains the components of each basis vector in 3D space.

        Raises
        ------
        RuntimeError
            If the spline is not defined in 3D or if the Frenet frame cannot be
            computed due to inflection points, straight segments, or undefined
            tangent/normal vectors.
        ValueError
            If the initial vector for the Bishop frame is not orthogonal to the
            tangent at `t[0]`, or if an invalid `method` is specified.

        Notes
        -----
        - The Frenet frame is not defined at points where the curve has zero curvature,
          such as straight segments or inflection points. In these cases, the Bishop
          frame is recommended.
        - For closed curves, check for discontinuities of the Bishop frame.

        References
        ----------
        .. [#movingframe] `Moving frame <https://en.wikipedia.org/wiki/Moving_frame>`_ on Wikipedia.
        .. [#bishop] Bishop, R. L. (1975). "There is More than One Way to Frame a Curve."
               American Mathematical Monthly, 82(3), 246-251.

        Examples
        --------

        We start by creating a 3D spline.

        >>> spline = splinebox.Spline(4, basis_function=splinebox.B3(), closed=True)
        >>> spline.knots = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])

        >>> spline.moving_frame(0)
        array([[ 0.,  1.,  0.],
               [-1.,  0.,  0.],
               [ 0., -0.,  1.]])

        >>> spline.moving_frame([0, 2, spline.M])
        array([[[ 0.,  1.,  0.],
                [-1.,  0.,  0.],
                [ 0., -0.,  1.]],
        <BLANKLINE>
               [[ 0., -1.,  0.],
                [ 1.,  0.,  0.],
                [-0.,  0.,  1.]],
        <BLANKLINE>
               [[ 0.,  1.,  0.],
                [-1.,  0.,  0.],
                [ 0., -0.,  1.]]])
        """
        self._check_control_points()
        if self.ndim != 3:
            raise RuntimeError("Moving frame only implemented for 3D splines.")
        
        t_arr, single_value = self._convert_to_array(t)
        
        # Sort t to ensure sequential scan works for Bishop
        sort_idx = jnp.argsort(t_arr)
        t_sorted = t_arr[sort_idx]
        
        d1 = self(t_sorted, derivative=1)
        d2 = self(t_sorted, derivative=2)
        d0 = self(t_sorted, derivative=0)
        
        # Normalize tangent safely
        T_norm = jnp.linalg.norm(d1, axis=-1, keepdims=True)
        T = d1 / (T_norm + 1e-12)
        
        if method == "frenet":
            # raise NotImplementedError("Frenet frame not implemented yet.")
            frames, binormal_norms = self._moving_frame_frenet(T, d2)
            if jnp.any(jnp.isclose(binormal_norms, 0.0)):
                if jnp.isclose(binormal_norms[0], 0.0) or jnp.isclose(binormal_norms[-1], 0.0):
                    raise RuntimeError(
                            "The Frenet frame cannot be computed at one or both ends of the spline. This is often due to edge padding of the knots. Try to skip t=0 and t=M-1 or change the padding."
                        )
                raise RuntimeError(
                    "The Frenet frame is not defined for splines with inflection points or straight segments, try the Bishop frame instead."
                )
        if method == "bishop":
            if initial_vector is None:
                guess = jnp.cross(jnp.cross(T[0], d2[0]), T[0])
                guess_norm = jnp.linalg.norm(guess)
                guess_is_degen = jnp.isclose(guess_norm, 0.0) or jnp.any(jnp.isnan(guess))
                initial_vector = self._initial_vector_guess(T[0], guess, guess_is_degen)
            
            # Bishop frame: Double Reflection (RMF) used.
            # _moving_frame_bishop now returns full frames (N, 3, 3).
            # It handles the topological correction internally if closed=True.
            frames = self._moving_frame_bishop(d0, T, initial_vector, t_sorted, self.M, self.closed)

        else:
            raise ValueError(f"Unknown moving frame method: {method}")
        
        if single_value: return frames[0]
        inv_sort_idx = jnp.argsort(sort_idx)
        frames = frames[inv_sort_idx]
        
        # Enforce coverage check for closed splines
        if self.closed and method == "bishop":
            # We require t to cover the start and end of the domain to ensure correct transport measure.
             # Ideally we check jnp.min(t) == 0 and jnp.max(t) == M.
            # But t_sorted is already sorted.
            
            # Check if start and end are within tolerance
            tol = 1e-3
            start_gap = jnp.abs(t_sorted[0])
            end_gap = jnp.abs(t_sorted[-1] - self.M)
            
            if start_gap > tol or end_gap > tol:
                raise ValueError(
                    f"For closed curves using Bishop frame, query points must cover the full domain [0, M] to compute topological correction. "
                    f"Gaps: start={start_gap}, end={end_gap}, Tolerance={tol}"
                )
            
        return frames

    def torsion(self, t):
        """
        Computes the torsion of the curve at t.
        tau = (r' x r'') . r''' / |r' x r''|^2
        """
        self._check_control_points()
        t_arr, single_val = self._convert_to_array(t)
        
        d1 = self(t_arr, derivative=1)
        d2 = self(t_arr, derivative=2)
        d3 = self(t_arr, derivative=3)
        
        val = self._torsion_jit(d1, d2, d3, self.ndim)
        
        if single_val:
            return val[0]
        return val

    @staticmethod
    @partial(jax.jit, static_argnames=['ndim'])
    def _torsion_jit(d1, d2, d3, ndim):
        if ndim == 2:
            return jnp.zeros(d1.shape[0])
        elif ndim == 3:
            cross_d1_d2 = jnp.cross(d1, d2)
            numerator = jnp.sum(cross_d1_d2 * d3, axis=-1)
            denominator = jnp.linalg.norm(cross_d1_d2, axis=-1)**2
            # Handle straight lines / inflection points where cross product is zero
            tau = jnp.where(denominator > 1e-12, numerator / denominator, 0.0)
            return tau
        else:
            # For N-dim, generalized torsion logic needed, or 0 if not 3D curve in specific sense
            return jnp.zeros(d1.shape[0])

    @jax.jit
    def total_torsion(self):
        """Computes integral of torsion over the spline domain."""
        limit = self.M if self.closed else self.M - 1
        # Use quadgk
        # Avoid lambda capture issues in JIT by defining integrand static or passing self params
        
        integral = quadgk(
            lambda t: self._torsion_jit(
                self._compute_spline_first_deriv(self._get_tval(jnp.atleast_1d(t), self.M, self._half_support, self._pad, self.closed), self.control_points, single_val=True), # d1
                JaxSpline._compute_spline_derivative(self._get_tval(jnp.atleast_1d(t), self.M, self._half_support, self._pad, self.closed), self.control_points, 2, self.basis_function), # d2
                JaxSpline._compute_spline_derivative(self._get_tval(jnp.atleast_1d(t), self.M, self._half_support, self._pad, self.closed), self.control_points, 3, self.basis_function), # d3
                self.ndim
            ),
            [0, limit],
            epsabs=1e-4, epsrel=1e-4,
            max_ninter=100
        )
        return integral[0]

    @staticmethod
    def _compute_spline_derivative(tval, control_points, order, basis_function):
        # We need a way to call arbitrary derivative from static context if possible, 
        # but basis_function object is available in self context. 
        # To make this fully static for quadgk lambda, we might need to pass basis_function or rely on closure.
        # Since 'total_torsion' is an instance method marked @jax.jit, 'self' is a valid pytree node.
        # So we can use self.basis_function inside the lambda? 
        # JAX jit on methods treats 'self' as pytree.
        # So we can access self.basis_function assuming it's also a Pytree node (which it is).
        
        basis_vals = basis_function(tval, derivative=order)
        res = jnp.matmul(basis_vals, control_points)
        return res[0] # Single val expected for quadgk scalar integrand? No, d1/d2/d3 are vectors. torsion returns scalar. ok.


    def __call__(self, t, derivative=0):
        """JIT-optimized evaluation."""
        self._check_control_points()
        t_arr, single_val = self._convert_to_array(t)
        
        # 1. Get t-vals (indices)
        tval = self._get_tval(t_arr, self.M, self._half_support, self._pad, self.closed)
        
        # 2. Evaluate basis functions (Assuming basis_function is JAX-compatible/JIT-ed)
        basis_vals = self.basis_function(tval, derivative=derivative)
        
        # 3. Matrix Multiplication
        value = self._eval_spline(basis_vals, self.control_points)
        
        if single_val:
            return value[0]
        return value

    def _control_points_centroid(self):
        self._check_control_points()
        return self._compute_control_point_centroid(self.control_points)
    
    def translate(self, vector):
        """
        Translates the spline by the given vector.
        """
        self._check_control_points()
        self.control_points = self._compute_control_points_translated(self.control_points, vector)

    def scale(self, scaling_factor):
        self._check_control_points()
        self.control_points = self._compute_control_points_scaled(self.control_points, scaling_factor)

    def rotate(self, rotation_matrix, centered=True):
        self._check_control_points()
        if centered:
            centroid = self._control_points_centroid()
            self.translate(-centroid)

        self.control_points = self._compute_control_points_rotated(self.control_points, rotation_matrix)

        if centered:
            self.translate(centroid)

    @partial(jax.jit, static_argnames=("return_t",))
    def distance(self, points, return_t=False):
        """
        Computes distance using Gradient Descent/Newton methods in JAX.
        """
        self._check_control_points()
        if self.ndim == 1:
            raise RuntimeError("cant compute distance for 1D splines.")
        
        single_point = False
        if points.ndim == 1:
            points = jnp.expand_dims(points, axis=0)
            single_point = True
        
        max_t = float(self.M if self.closed else self.M - 1)
        
        # 1. Coarse search to find initialization
        t_coarse = jnp.linspace(0.0, max_t, self.M * 10)
        pts_coarse = self(t_coarse) # (Grid, Dim)
        
        best_idx = self._compute_closest_indices_between_points(pts_coarse, points)
        t_init = t_coarse[best_idx]
        
        max_iter = 5

        final_t = self._find_closest_t_to_point(
            t_init, points,
            self.control_points,
            self.M, self._half_support, self._pad, self.closed,
            0.0, max_t,
            max_iter,
        )

        final_pos = self(final_t)
        min_dists = jnp.linalg.norm(final_pos - points, axis=1)
        
        if single_point:
            min_dists = min_dists[0]
            final_t = final_t[0]

        if return_t:
            return min_dists, final_t
        return min_dists
    
    def mesh(self, 
             radius=None, 
             step_t=0.1, 
             step_angle=5, 
             mesh_type="surface", 
             cap_ends=False, 
             frame="bishop", 
             initial_vector=None
        ):
        """
        Generates 3D mesh. Fully vectorized and JIT-optimized.
        """
        self._check_control_points()
        if self.ndim != 3: raise NotImplementedError("3D only.")
        
        limit = self.M if self.closed else self.M - 1 + step_t
        t = jnp.arange(0, limit, step_t)
        if radius is None:
            radius = 0.0

        centers = self(t)
        if radius == 0.0:
            connectivity = jnp.stack((jnp.arange(len(centers)), jnp.arange(1, len(centers)+1)), axis=-1)
            if self.closed:
                connectivity = connectivity.at[-1, 1].set(0)
            else:
                connectivity = connectivity[:-1]
            return centers, connectivity
        else:
            raise NotImplementedError("Non-zero radius mesh not implemented yet.")
            
            # # Calculate Frames and Centers
            frames = self.moving_frame(t, method=frame, initial_vector=initial_vector)
            normals = frames[:, 1, :]
            binormals = frames[:, 2, :] # Or derived from frame
            
            phi = jnp.arange(0, 360, step_angle)
            
            # Create Grid
            # T_grid: (nt, nphi), Phi_grid: (nt, nphi)
            T_grid, Phi_grid = jnp.meshgrid(t, phi, indexing='ij')
            
            # Calculate Radii
            if callable(radius):
                r_vals = jax.vmap(lambda t_v, p_v: radius(t_v, p_v))(T_grid.flatten(), Phi_grid.flatten())
                r_vals = r_vals.reshape(T_grid.shape)
            else:
                r_vals = radius
                
            # Calculate Offsets (Vectorized)
            # We rotate the normal vector by phi around the tangent
            # Normal (N) and Binormal (B) form the plane.
            # Offset = R * (cos(phi) * N + sin(phi) * B)
            # Note: Previous implementation assumed normal rotation logic, implementing standard tube logic here
            
            rad_phi = jnp.deg2rad(Phi_grid)[..., None] # (nt, nphi, 1)
            N_expanded = normals[:, None, :] # (nt, 1, 3)
            B_expanded = binormals[:, None, :] # (nt, 1, 3)
            
            # Shape: (nt, nphi, 3)
            offsets = (jnp.cos(rad_phi) * N_expanded + jnp.sin(rad_phi) * B_expanded)
            
            # Apply radius
            if jnp.ndim(r_vals) > 0:
                if r_vals.ndim == 2: r_vals = r_vals[..., None]
            offsets = offsets * r_vals
            
            # Points
            points = centers[:, None, :] + offsets
            points_flat = points.reshape(-1, 3)
            
            # Connectivity
            s_conn, v_conn = _generate_connectivity_jit(self.closed, len(phi), len(t), len(points_flat))
            
            if mesh_type == "surface":
                # (Simplification: skipping the cap_ends logic for brevity, but can be added similarly)
                return points_flat, s_conn
            else:
                # Volume logic requires center points added to array, handled in JIT function logic generally
                # For strict JIT compatibility, reshaping must be static. 
                # Returning surface mesh for now as primary example.
                return points_flat, v_conn
            
    
from splinebox.jax_basis_functions import basis_function_from_name
def _prepared_dict_for_constructor(data):
    """
    Helper function that processes the dictionaries loaded from
    json files. It ensure all of the values are valid and prepares
    a dictionary that can be passed to the constructor using `**`.

    Parameters
    ----------
    data : dict
        The dictionary from the json file.
    """
    if not isinstance(data["version"], int):
        raise ValueError("version has to be an integer.")

    if not isinstance(data["M"], int):
        raise ValueError("M has to be an integer.")

    data["basis_function"] = basis_function_from_name(data["basis_function"], M=data["M"])

    data["closed"] = str(data["closed"]).lower()
    true_strings = ["true", "1", "t", "y", "yes"]
    false_strings = ["false", "0", "f", "n", "no"]
    if data["closed"] not in true_strings and data["closed"] not in false_strings:
        raise ValueError(f"closed should be a string that can be interpreted as a boolean not {data['closed']}.")
    data["closed"] = data["closed"] in true_strings

    if "control_points" in data:
        data["control_points"] = jnp.array(data["control_points"])
    if "tangents" in data:
        data["tangents"] = jnp.array(data["tangents"])

    # The version of the json file is only required for parsing
    del data["version"]

    return data


def splines_to_json(path, splines, version=1):
    """
    Saves multiple splines in a single json file.

    Parameters
    ----------
    path : str or pathlib.Path
        The path where the json should be saved.
    splines : iterable
        For instance a list of :class:`splinebox.spline_curves.Spline`
        and :class:`splinebox.spline_curves.HermiteSpline` objects.
    version : int
        The version used to produce the json file.

    Examples
    --------
    Create two random splines...

    >>> spline1 = splinebox.Spline(M=5, basis_function=splinebox.B3(), closed=False)
    >>> spline1.control_points = np.random.rand(7, 3)
    >>> spline2 = splinebox.Spline(M=6, basis_function=splinebox.CatmullRom(), closed=True)
    >>> spline2.control_points = np.random.rand(6, 2)

    Next, we save them as a json file.

    >>> splinebox.splines_to_json("splines.json", [spline1, spline2])

    Then we can load them back into python.

    >>> loaded_splines = splinebox.splines_from_json("splines.json")
    >>> loaded_splines[0] == spline1
    True
    >>> loaded_splines[1] == spline2
    True
    """
    dicts = []

    for spline in splines:
        dicts.append(spline._to_dict(version))

    with open(path, "w") as f:
        json.dump(dicts, f, indent=2)


def splines_from_json(path):
    """
    Loads multiple splines from a json file generated using
    :func:`splinebox.spline_curves.splines_to_json`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the json file.

    Returns
    -------
    splines : list
        A list of :class:`splinebox.spline_curves.Spline` and
        :class:`splinebox.spline_curves.HermiteSpline` objects.

    Examples
    --------
    Create two random splines...

    >>> spline1 = splinebox.Spline(M=5, basis_function=splinebox.B3(), closed=False)
    >>> spline1.control_points = np.random.rand(7, 3)
    >>> spline2 = splinebox.Spline(M=6, basis_function=splinebox.CatmullRom(), closed=True)
    >>> spline2.control_points = np.random.rand(6, 2)

    Next, we save them as a json file.

    >>> splinebox.splines_to_json("splines.json", [spline1, spline2])

    Then we can load them back into python.

    >>> loaded_splines = splinebox.splines_from_json("splines.json")
    >>> loaded_splines[0] == spline1
    True
    >>> loaded_splines[1] == spline2
    True
    """
    splines = []
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        # this is a json file of a single spline
        data = [data]

    for spline_data in data:
        spline_data = _prepared_dict_for_constructor(spline_data)

        if spline_data["basis_function"].multigenerator:
            raise NotImplementedError("Loading Hermite splines from json is not implemented yet.")
            # This is a basis function for a Hermite spline
            # splines.append(HermiteSpline(**spline_data))
        else:
            splines.append(JaxSpline(**spline_data))

    return splines
