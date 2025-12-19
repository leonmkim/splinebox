#%%
import jax
import jax.numpy as jnp
from jax import lax
from jax import jit
from jax import vmap
from jax.numpy import vectorize
from jax.tree_util import register_pytree_node_class
import numpy as np

import inspect
import sys
import warnings

#%%

@register_pytree_node_class
class JaxBasisFunction:
    """
    JAX-compatible base class for basis functions.
    """
    _unimplemented_message = "This function is not implemented."

    def __init__(self, multigenerator, support):
        self.multigenerator = multigenerator
        self.support = support

    # #####################################
    # Make the class a custom pytree so we can jit class methods. See https://docs.jax.dev/en/latest/faq.html#strategy-3-making-customclass-a-pytree
    # #####################################
    def tree_flatten(self):
        children = None  # arrays / dynamic values
        aux_data = dict( # static values
            multigenerator=self.multigenerator,
            support=self.support,
        )  
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)

    def __str__(self):
        return "BasisFunction"

    def __repr__(self):
        return f"splinebox.basis_functions.BasisFunction(multigenerator={repr(self.multigenerator)}, support={repr(self.support)})"

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and other.multigenerator == self.multigenerator
            and other.support == self.support
        )
    
    # We mark the derivative argument as static so JAX re-compiles 
    # specific versions for derivative=0, 1, or 2.
    def __call__(self, t, derivative=0):
        if derivative == 0:
            return self._func(t)
        elif derivative == 1:
            return self._derivative_1(t)
        elif derivative == 2:
            return self._derivative_2(t)
        else:
            raise ValueError(f"derivative has to be 0, 1, or 2 not {derivative}")

    def _func(self, t):
        raise NotImplementedError(JaxBasisFunction._unimplemented_message)

    def _derivative_1(self, t):
        raise NotImplementedError(JaxBasisFunction._unimplemented_message)

    def _derivative_2(self, t):
        raise NotImplementedError(JaxBasisFunction._unimplemented_message)

    def filter_symmetric(self, s):
        raise NotImplementedError(JaxBasisFunction._unimplemented_message)

    def filter_periodic(self, s):
        raise NotImplementedError(JaxBasisFunction._unimplemented_message)

    def refinement_mask(self):
        raise NotImplementedError(JaxBasisFunction._unimplemented_message)

class JaxB3(JaxBasisFunction):
    """
    JAX implementation of the cubic B-spline basis function.
    """
    def __init__(self):
        super().__init__(False, 4)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

    def __str__(self):
        return "JaxB3"

    def __repr__(self):
        return "splinebox.jax_basis_functions.JaxB3()"

    @staticmethod
    @jit
    @vectorize
    def _func(t: float) -> float:
        # Branch 1: 0 <= |t| < 1
        # val1 = 2 / 3 - (abst ** 2) + (abst ** 3) / 2
        # Branch 2: 1 <= |t| <= 2
        # val2 = ((2 - abst) ** 3) / 6
        abst = jnp.abs(t)
        branch_idx = jnp.floor(abst).astype(int)
        return lax.switch(branch_idx, # this will naturally clamp to valid indices in the branches so handles anything outside [0,2]
                            [
                                lambda x: 2 / 3 - (x ** 2) + (x ** 3) / 2,
                                lambda x: ((2 - x) ** 3) / 6,
                                lambda x: 0.0,
                            ],
                            abst
                        )

    @staticmethod
    @jit
    @vectorize
    def _derivative_1(t: float):
        return jax.grad(JaxB3._func)(t)
        # # t = jnp.asarray(t)
        
        # # 0 <= t < 1
        # v1 = -2 * t + 1.5 * t * t
        # # -1 < t < 0
        # v2 = -2 * t - 1.5 * t * t
        # # 1 <= t <= 2
        # v3 = -0.5 * ((2 - t) ** 2)
        # # -2 <= t <= -1
        # v4 = 0.5 * ((2 + t) ** 2)

        # cond1 = (t >= 0) & (t < 1)
        # cond2 = (t > -1) & (t < 0)
        # cond3 = (t >= 1) & (t <= 2)
        # cond4 = (t >= -2) & (t <= -1)

        # return jnp.select([cond1, cond2, cond3, cond4], [v1, v2, v3, v4], default=0.0)

    @staticmethod
    @jit
    @vectorize
    def _derivative_2(t: float):
        return jax.grad(JaxB3._derivative_1)(t)
        # # t = jnp.asarray(t)

        # # 0 <= t < 1
        # v1 = -2 + 3 * t
        # # -1 < t < 0
        # v2 = -2 - 3 * t
        # # 1 <= t <= 2
        # v3 = 2 - t
        # # -2 <= t <= -1
        # v4 = 2 + t

        # cond1 = (t >= 0) & (t < 1)
        # cond2 = (t > -1) & (t < 0)
        # cond3 = (t >= 1) & (t <= 2)
        # cond4 = (t >= -2) & (t <= -1)

        # return jnp.select([cond1, cond2, cond3, cond4], [v1, v2, v3, v4], default=0.0)

    def filter_symmetric(self, s):
        # Cast to float to ensure precision
        return self._filter_symmetric(s.astype(jnp.float32))

    @staticmethod
    @jit
    def _filter_symmetric(s):
        """
        JAX reimplementation of a recursive digital filter to convert knot points 
        to cubic B-spline control points.

        cp stands for c+ and cm is c- in the original paper.
        """
        # Ensure s is at least 2D for consistent processing (M, ndim)
        # s_in = s if s.ndim > 1 else s[:, None]
        # if is not allowed in JAX jit, so we handle outside the jit function.
        s_in = s
        M, ndim = s_in.shape
        
        pole = -2.0 + jnp.sqrt(3.0)
        eps = 1e-8

        # --- Step 1: Initialization of cp[0] ---
        # The original code loops k from 0 to k0 to accumulate the initial value.
        # k0 is small and bounded, so we can compute this analytically or via a loop.
        # Given the mathematical structure (summing a geometric series over a mirrored signal),
        # we can implement the loop using lax.fori_loop for JIT compatibility.
        
        k0_limit = 2 * M - 2
        # Determine iteration count based on tolerance
        k_eps = jnp.ceil(jnp.log(eps) / jnp.log(jnp.abs(pole))).astype(int)
        k0 = jnp.minimum(k0_limit, k_eps)

        def init_body_fun(k, val_acc):
            # Logic: m = k % (2M - 2)
            # if m >= M: idx = 2M - 2 - m; else: idx = m
            # logic: s_val = s[idx]
            
            m = k % (2 * M - 2)
            
            # Branchless index selection for JIT efficiency
            # if m >= M, we are in the mirrored part
            is_mirror = m >= M
            # idx = jnp.where(is_mirror, 2 * M - 2 - m, m)
            idx = lax.cond(is_mirror, lambda x: 2*M - 2 - x, lambda x: x, m)
            
            val = s_in[idx]
            return val_acc + val * (pole ** m)
        
        cp0_accum = lax.fori_loop(0, k0, init_body_fun, jnp.zeros((ndim,), dtype=s_in.dtype))
        
        # Final scaling for cp[0]
        scaling_factor = 1.0 / (1.0 - (pole ** (2 * M - 2)))
        cp0 = cp0_accum * scaling_factor

        # --- Step 2: Causal Recursive Filter (Forward Pass) ---
        # cp[k] = s[k] + pole * cp[k - 1]
        # We use scan to carry the previous cp value forward.
        
        def forward_scan(prev_cp, s_k):
            # The loop in original code starts at k=1, but scan naturally iterates over all.
            # We manually handle the first element injection.
            current_cp = s_k + pole * prev_cp
            return current_cp, current_cp

        # We provide cp0 as the "previous" value for the first iteration (k=1) effectively.
        # However, the original loop sets cp[0] explicitly, then loops k=1..M.
        # To use scan efficiently over s[1:], we carry cp0.
        
        _, cp_rest = lax.scan(forward_scan, cp0, s_in[1:])
        
        # Concatenate cp0 and the rest of the result
        cp = jnp.concatenate([cp0[None, :], cp_rest], axis=0)

        # --- Step 3: Anticausal Initialization (cm[M-1]) ---
        cm_last = (cp[M - 1] + pole * cp[M - 2]) * (pole / (pole**2 - 1.0))

        # --- Step 4: Anticausal Recursive Filter (Backward Pass) ---
        # cm[k] = pole * (cm[k+1] - cp[k])
        # Iterates backwards from M-2 to 0.
        
        # We need to scan backwards over cp. 
        # The input to the scan will be cp values from M-2 down to 0.
        # The carry will be the 'next' cm value (initially cm[M-1]).
        
        def backward_scan(next_cm, cp_k):
            current_cm = pole * (next_cm - cp_k)
            return current_cm, current_cm

        # Inputs for scan: cp[0 ... M-2] reversed, or read simply backwards
        # We slice cp up to M-2, then reverse it for the scan
        cp_slice = cp[:-1] # indices 0 to M-2
        
        _, cm_rest = lax.scan(backward_scan, cm_last, cp_slice, reverse=True)
        
        # The result of scan is ordered M-2 down to 0. We don't need to reverse it back
        # if we concatenate correctly, but usually scan outputs in order of processing.
        # If reverse=True in scan, JAX processes last-to-first, but the output array 
        # is stacked in the order processed (so it corresponds to indices M-2, M-3... 0).
        # To get [0, 1... M-2], we flip the result.
        # cm_rest = cm_rest_reversed[::-1]

        # Concatenate the rest and the last element
        cm = jnp.concatenate([cm_rest, cm_last[None, :]], axis=0)

        # --- Step 5: Final Scaling and Thresholding ---
        c = cm * 6.0

        # Apply soft thresholding (where(abs < eps, 0, c))
        c = jnp.where(jnp.abs(c) < eps, 0.0, c)

        # Reshape back to original dimensions if input was 1D
        # if is not allowed in JAX jit, so we handle outside the jit function.
        # if s.ndim == 1:
        #     c = c.squeeze(axis=1)

        return c
    
    def filter_periodic(self, s):
        return self._filter_periodic(s.astype(jnp.float32))

    @staticmethod
    @jit
    def _filter_periodic(s):
        """
        JAX reimplementation of a recursive digital filter to convert knot to control points for a periodic closed spline. 

        Original Logic:
        1. Initialize cp[0] via circular convolution (geometric series).
        2. Causal forward filter for cp[1...M-1].
        3. Initialize cm[M-1] via circular convolution of cp.
        4. Anticausal backward filter for cm[M-2...0].
        """
        # Force usage as is; caller must ensure shape compatibility
        s_in = s
        M, ndim = s_in.shape
        
        pole = -2.0 + jnp.sqrt(3.0)
        eps = 1e-8
        init_val = jnp.zeros((ndim,), dtype=s_in.dtype)

        # --- Step 1: Initialization of cp[0] ---
        # Original: sum_{k=0}^{M-1} s[(M-k)%M] * pole^k
        # We use fori_loop to handle the summation efficiently in XLA
        def init_cp0_body(k, val_acc):
            # Index logic: (M - k) % M
            # When k=0, idx=0. When k=1, idx=M-1.
            idx = (M - k) % M
            val = s_in[idx]
            return val_acc + val * (pole ** k)

        cp0_accum = lax.fori_loop(0, M, init_cp0_body, init_val)
        
        cp0 = cp0_accum * (1.0 / (1.0 - (pole ** M)))

        # --- Step 2: Causal Recursive Filter (Forward) ---
        # cp[k] = s[k] + pole * cp[k - 1]
        # Identical structure to symmetric case
        
        def forward_scan(prev_cp, s_k):
            current_cp = s_k + pole * prev_cp
            return current_cp, current_cp

        _, cp_rest = lax.scan(forward_scan, cp0, s_in[1:])
        cp = jnp.concatenate([cp0[None, :], cp_rest], axis=0)

        # --- Step 3: Anticausal Initialization (cm[M-1]) ---
        # Original logic:
        #   term1 = sum_{k=0}^{M-1} (pole^k) * cp[k]
        #   term1 *= pole / (1 - pole^M)
        #   cm[M-1] = -pole * (term1 + cp[M-1])
        
        def init_cm_last_body(k, val_acc):
            val = cp[k]
            return val_acc + val * (pole ** k)

        cm_last_accum = lax.fori_loop(0, M, init_cm_last_body, init_val)
        
        term1 = cm_last_accum * (pole / (1.0 - (pole ** M)))
        cm_last = -pole * (term1 + cp[M - 1])

        # --- Step 4: Anticausal Recursive Filter (Backward) ---
        # cm[k] = pole * (cm[k+1] - cp[k])
        # Iterates backwards from M-2 to 0.
        
        def backward_scan(next_cm, cp_k):
            current_cm = pole * (next_cm - cp_k)
            return current_cm, current_cm

        # Input for scan is cp[0 ... M-2] (indices 0 to M-2 inclusive)
        cp_slice = cp[:-1] 
        
        # reverse=True scans right-to-left. 
        # Output matches input alignment (0 to M-2), so no manual flip needed.
        _, cm_rest = lax.scan(backward_scan, cm_last, cp_slice, reverse=True)

        # Concatenate: [cm_rest (0..M-2), cm_last (M-1)]
        cm = jnp.concatenate([cm_rest, cm_last[None, :]], axis=0)

        # --- Step 5: Final Scaling ---
        c = cm * 6.0
        c = jnp.where(jnp.abs(c) < eps, 0.0, c)

        return c

    # def refinement_mask(self):
    #     # NOTE: The original code utilized a helper `_multinomial` 
    #     # which was not provided in the snippet. 
    #     # Implementing the mask for B3 (cubic B-spline) directly:
    #     # The mask for B3 is [1/8, 4/8, 6/8, 4/8, 1/8] * some factor?
    #     # Actually standard cubic b-spline subdivision mask is [1, 4, 6, 4, 1] / 8.
    #     # Returning the standard mask for now.
    #     return jnp.array([1, 4, 6, 4, 1], dtype=float) / 8.0


def inventory():
    """
    This function returns a dictionary with all
    implemented basis function.
    The keys are the names of the basis function and the
    values are the classes.

    Examples
    --------

    >>> splinebox.basis_functions.inventory()
    {'B1': <class 'splinebox.basis_functions.B1'>, 'B2': <class 'splinebox.basis_functions.B2'>, 'B3': <class 'splinebox.basis_functions.B3'>, 'CatmullRom': <class 'splinebox.basis_functions.CatmullRom'>, 'CubicHermite': <class 'splinebox.basis_functions.CubicHermite'>, 'Exponential': <class 'splinebox.basis_functions.Exponential'>, 'ExponentialHermite': <class 'splinebox.basis_functions.ExponentialHermite'>}
    """
    _inventory = inspect.getmembers(sys.modules[__name__])
    # Filter out everything that is not a class
    _inventory = list(filter(lambda pair: inspect.isclass(pair[1]), _inventory))
    # Remove the base class from the inventory
    _inventory = dict(filter(lambda pair: pair[0] != "JaxBasisFunction", _inventory))
    return _inventory

#%%
def basis_function_from_name(name, **kwargs):
    """
    Returns a basis function object based on the name of the basis function.

    Parameters
    ----------
    name : str
        The name of the basis function (e.g. 'B1', 'ExponentialHermite', etc.)
    kwargs
        Additional keyword arguments that may be required to instantiate
        certain basis functions. For example, :code:`M`(the number of
        knots/control points) is required for exponential basis functions.

    Returns
    -------
    basis_function : Object of one of the subcases of :class:`splinebox.basis_functions.BasisFunction`.
        An instance of a subclass of splinebox.basis_functions.BasisFunction,
        initialized according to the provided name and keyword arguments.

    Examples
    --------

    >>> splinebox.basis_functions.basis_function_from_name("B3")
    splinebox.basis_functions.B3()
    """
    basis_function_inventory = inventory()
    if name not in basis_function_inventory:
        raise ValueError(
            f"Unknown basis function '{name}'. Available basis functions are {list(basis_function_inventory.keys())}."
        )
    basis_function_class = basis_function_inventory[name]
    signature = inspect.signature(basis_function_class)
    basis_functions_kwargs = {}
    for param in signature.parameters.values():
        if param.name in kwargs:
            basis_functions_kwargs[param.name] = kwargs[param.name]
        elif param.default is not param.empty:
            basis_functions_kwargs[param.name] = param.default
        else:
            raise RuntimeError(
                f"{name} requires a keyword argument {param.name}, please specify it when calling `basis_function_from_name`"
            )
    return basis_function_class(**basis_functions_kwargs)

# # %%
# @jit
# @vmap
# def _func(t: float) -> float:
#     # Branch 1: 0 <= |t| < 1
#     # val1 = 2 / 3 - (abst ** 2) + (abst ** 3) / 2
#     # Branch 2: 1 <= |t| <= 2
#     # val2 = ((2 - abst) ** 3) / 6
#     abst = jnp.abs(t)
#     branch_idx = jnp.floor(abst).astype(int)
#     return lax.switch(branch_idx, # this will naturally clamp to valid indices in the branches so handles anything outside [0,2]
#                         [
#                             lambda x: 2 / 3 - (x ** 2) + (x ** 3) / 2,
#                             lambda x: ((2 - x) ** 3) / 6,
#                             lambda x: 0.0,
#                         ],
#                         abst
#                     )
#%%
# test_vals = jnp.zeros((10,50,9), dtype=jnp.float32)

# b3 = JaxB3()
# output = b3(test_vals, derivative=0)
# print(b3(test_vals, derivative=2))
# %%

# %%
