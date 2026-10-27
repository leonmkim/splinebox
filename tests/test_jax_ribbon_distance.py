"""Ribbon geometry must not depend on the batch of queried parameters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from splinebox import JaxB3, JaxSpline


@pytest.fixture(scope="module")
def ribbon():
    angle = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    controls = np.column_stack(
        (0.13 * np.cos(angle), 0.08 * np.sin(angle), 0.03 * np.sin(2 * angle), 0.4 * np.cos(angle))
    )
    return JaxSpline(9, JaxB3(), control_points=controls, is_ribbon=True, ribbon_width=0.025)


def global_surface(spline, t, u):
    """Evaluate with the dense global frames used by the closest-point search."""
    direction = JaxSpline._ribbon_width_dir(t, spline.theta(t), spline.frame_cache, closed=True, M=spline.M)
    return spline.centerline(t) + 0.5 * spline.ribbon_width * jnp.asarray(u)[..., None] * direction


def test_surface_uses_global_frames_for_clustered_and_single_queries(ribbon):
    # Partial coverage and duplicates cannot define the closed ribbon's frame.
    t = jnp.repeat(jnp.linspace(2.0, 2.3, 12), 8)
    u = jnp.linspace(-0.9, 0.9, len(t))
    expected = global_surface(ribbon, t, u)
    np.testing.assert_allclose(ribbon.ribbon_surface(t, u), expected, atol=2e-7)
    np.testing.assert_allclose(ribbon.ribbon_surface(t[17], u[17]), expected[17], atol=2e-7)

    perm = np.random.default_rng(42).permutation(len(t))
    evaluate = jax.jit(lambda spline, ts, us: spline.ribbon_surface(ts, us))
    np.testing.assert_allclose(evaluate(ribbon, t[perm], u[perm]), expected[perm], atol=2e-7)


def test_distance_and_gradient_match_global_surface_after_pytree_reconstruction(ribbon):
    t = jnp.repeat(jnp.linspace(2.0, 2.3, 12), 8)
    points = global_surface(ribbon, t, jnp.linspace(-0.8, 0.8, len(t)))
    # Nonzero residuals exercise differentiability as well as on-surface accuracy.
    points = points + jnp.array([0.001, -0.002, 0.003])
    aux = ribbon.tree_flatten()[1]

    @jax.jit
    def evaluate(controls):
        spline = JaxSpline.tree_unflatten(aux, (controls,))
        return spline.distance(points, return_arg=True)

    distance, (closest_t, closest_u) = evaluate(ribbon.control_points)
    expected = JaxSpline._safe_2norm(global_surface(ribbon, closest_t, closest_u) - points)
    np.testing.assert_allclose(distance, expected, atol=2e-7)
    np.testing.assert_allclose(ribbon.distance(points), distance, atol=2e-7)
    single_distance, single_arg = ribbon.distance(points[17], return_arg=True)
    np.testing.assert_allclose(single_distance, distance[17], atol=2e-7)
    np.testing.assert_allclose(single_arg, (closest_t[17], closest_u[17]), atol=2e-6)

    # The existing envelope gradient freezes t/u, but must differentiate the
    # same global surface (including its frame) as the forward evaluation.
    def reference_loss(controls):
        spline = JaxSpline.tree_unflatten(aux, (controls,))
        residual = global_surface(spline, closest_t, closest_u) - points
        return jnp.mean(JaxSpline._safe_2norm(residual))

    actual_gradient = jax.jit(jax.grad(lambda cp: jnp.mean(evaluate(cp)[0])))(ribbon.control_points)
    expected_gradient = jax.jit(jax.grad(reference_loss))(ribbon.control_points)
    assert np.isfinite(actual_gradient).all()
    assert np.linalg.norm(np.asarray(actual_gradient)[:, 3]) > 1e-6
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-4, atol=2e-6)
