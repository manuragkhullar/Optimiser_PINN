"""residual.py – PDE residual functions"""
import jax
import jax.numpy as jnp

# ───────────────────────────── Burgers residual ────────────────────────────────

def _burgers_residual(params, x, t, model, nu=0.01/jnp.pi):
    u_fn = lambda xx, tt: model.apply(params, xx, tt)
    u    = u_fn(x, t)
    u_x  = jax.grad(u_fn, argnums=0)(x, t)
    u_t  = jax.grad(u_fn, argnums=1)(x, t)
    u_xx = jax.grad(lambda xx, tt: jax.grad(u_fn, 0)(xx, tt), 0)(x, t)
    return u_t + u * u_x - nu * u_xx

# ──────────────────────────── Allen‑Cahn residual ─────────────────────────────

def _allen_cahn_residual(params, x, t, model):
    u_fn = lambda xx, tt: model.apply(params, xx, tt)
    u    = u_fn(x, t)
    u_t  = jax.grad(u_fn, argnums=1)(x, t)
    u_xx = jax.grad(lambda xx, tt: jax.grad(u_fn, 0)(xx, tt), 0)(x, t)
    return u_t - (5*u + 0.0001*u_xx - 5*u**3)

# ─────────────────────────────────── Dispatcher ────────────────────────────────

def get_residual(pde: str):
    if pde == 'burgers':
        return _burgers_residual
    elif pde == 'allen_cahn':
        return _allen_cahn_residual
    else:
        raise ValueError(f"Unknown PDE type: {pde}")
