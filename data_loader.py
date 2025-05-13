"""dataloader.py – loads <pde>.mat and builds training batches"""
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from scipy.io import loadmat

# ───────────────────────────────────────────────────────────────────────────────

def load_mat_data(pde: str):
    mat_path = Path(f"{pde}.mat")
    if not mat_path.exists():
        raise FileNotFoundError(f"{mat_path} not found – place solution file in working dir")
    data = loadmat(mat_path)
    return data['x'].squeeze(), data['t'].squeeze(), data['usol']  # x, t, solution grid


def build_dataset(key, pde: str, N_f=10000, N_i=1000, N_b=1000):
    x, t, usol = load_mat_data(pde)
    k1, k2, k3 = jax.random.split(key, 3)

    # Collocation points
    x_f = jax.random.uniform(k1, (N_f,), minval=float(x[0]), maxval=float(x[-1]))
    t_f = jax.random.uniform(k2, (N_f,), minval=float(t[0]), maxval=float(t[-1]))

    # Initial condition (earliest time slice)
    x_i = jnp.linspace(x[0], x[-1], N_i)
    t_i = jnp.full_like(x_i, t[0])
    u_i = jnp.interp(x_i, x, usol[0])

    # Boundary conditions (left & right)
    t_b = jax.random.uniform(k3, (N_b,), minval=float(t[0]), maxval=float(t[-1]))
    x_lb = jnp.full_like(t_b, x[0]); x_rb = jnp.full_like(t_b, x[-1])
    u_lb = jnp.interp(t_b, t, usol[:, 0]); u_rb = jnp.interp(t_b, t, usol[:, -1])

    return (x_f, t_f), (x_i, t_i, u_i), (x_lb, x_rb, t_b, jnp.stack([u_lb, u_rb], 0))
