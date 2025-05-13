"""optimiser.py – Adam, SOAP‑PDE, SOAP‑Lib (JAX) trainers"""
import optax
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import tree_util
from typing import NamedTuple, Any

try:
    from soap_jax import soap as SoapLib  # <-- JAX-native SOAP
except ImportError:
    SoapLib = None

# ─────────────────────────────────────── Loss Function ─────────────────────────

def loss_fn(params, batch, model, residual_fn):
    (x_f, t_f), (x_i, t_i, u_i), (x_lb, x_rb, t_b, u_both) = batch
    u_left, u_right = u_both[0], u_both[1]
    ic = jnp.mean((model.apply(params, x_i, t_i) - u_i) ** 2)
    bc = jnp.mean((model.apply(params, x_lb, t_b) - u_left) ** 2 +
                  (model.apply(params, x_rb, t_b) - u_right) ** 2)
    res = jax.vmap(lambda xx, tt: residual_fn(params, xx, tt, model))(x_f, t_f)
    phys = jnp.mean(res ** 2)
    return ic + bc + phys

# ───────────────────────────────────────── Adam ─────────────────────────────────

def make_adam_trainer(model, residual_fn, lr=1e-3):
    opt = optax.adam(lr)

    @jax.jit
    def step(p, s, batch):
        l, g = jax.value_and_grad(lambda pp: loss_fn(pp, batch, model, residual_fn))(p)
        updates, s = opt.update(g, s, p)
        p = optax.apply_updates(p, updates)
        return p, s, l

    return opt.init, step

# ──────────────────────────────────────── SOAP‑PDE ─────────────────────────────
class _SoapState(NamedTuple):
    count: jnp.ndarray
    m: Any
    v: Any

def _init_soap_state(params):
    zeros = lambda p: jnp.zeros_like(p)
    return _SoapState(count=jnp.zeros([], jnp.int32),
                      m=tree_util.tree_map(zeros, params),
                      v=tree_util.tree_map(zeros, params))

def make_soap_pde_trainer(model, residual_fn, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    @jax.jit
    def step(p, s, batch):
        l, g_tot = jax.value_and_grad(lambda pp: loss_fn(pp, batch, model, residual_fn))(p)
        (x_f, t_f) = batch[0]
        g_res = jax.vmap(lambda xx, tt: jax.grad(lambda pp: residual_fn(pp, xx, tt, model))(p))(x_f, t_f)
        v = tree_util.tree_map(lambda v_old, g: b2 * v_old + (1 - b2) * jnp.mean(g ** 2, axis=0), s.v, g_res)
        m = tree_util.tree_map(lambda m_old, g: b1 * m_old + (1 - b1) * g, s.m, g_tot)
        m_hat = tree_util.tree_map(lambda m_, v_: m_ / (jnp.sqrt(v_) + eps), m, v)
        p = tree_util.tree_map(lambda w, mh: w - lr * mh, p, m_hat)
        return p, _SoapState(s.count + 1, m, v), l

    return _init_soap_state, step

# ─────────────────────────────────────── SOAP‑Lib (JAX) ──────────────────────────────

def make_soap_lib_trainer(model, residual_fn, lr=3e-3, b1=0.95, b2=0.95, weight_decay=0.01, precond=10):
    if SoapLib is None:
        print("[!] SOAP-JAX not installed. Skipping SOAP-Lib.")
        return lambda _: None, lambda *a: (None, None, float('inf'))

    opt = SoapLib(
        learning_rate=lr,
        b1=b1,
        b2=b2,
        weight_decay=weight_decay,
        precondition_frequency=precond
    )

    @jax.jit
    def step(p, s, batch):
        l, g = jax.value_and_grad(lambda pp: loss_fn(pp, batch, model, residual_fn))(p)
        updates, s = opt.update(g, s, p)
        p = optax.apply_updates(p, updates)
        return p, s, l

    return opt.init, step

# ───────────────────────── Trainer‑factory collection helper ───────────────────

def get_optim_trainer_factories(model, residual_fn):
    factories = {
        'Adam': make_adam_trainer(model, residual_fn),
        'SOAP-PDE': make_soap_pde_trainer(model, residual_fn)
    }
    if SoapLib is not None:
        factories['SOAP-Lib'] = make_soap_lib_trainer(model, residual_fn)
    return factories

# ───────────────────────── Display all figures (for Colab) ────────────────────

def show_all_figures():
    """Manually display all active matplotlib figures (e.g., in Colab)."""
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.show()
