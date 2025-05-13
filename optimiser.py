"""optimiser.py – Adam, SOAP‑PDE, SOAP‑Lib trainers"""
import optax
import jax
import jax.numpy as jnp
from jax import tree_util
from typing import NamedTuple
try:
    from soap import SOAP as SoapLib
except ImportError:
    SoapLib = None

# ───────────────────────────────────────── Adam ─────────────────────────────────

def make_adam_trainer(model, residual_fn, lr=1e-3):
    opt = optax.adam(lr)
    @jax.jit
    def step(p, s, batch):
        l, g = jax.value_and_grad(lambda pp: batch['loss'](pp))(p)
        upd, s = opt.update(g, s, p); p = optax.apply_updates(p, upd)
        return p, s, l
    return opt.init, step

# ──────────────────────────────────────── SOAP‑PDE ─────────────────────────────
class _SoapState(NamedTuple):
    count: jnp.ndarray; m: any; v: any

def _init_soap_state(params):
    zeros = lambda p: jnp.zeros_like(p)
    return _SoapState(count=jnp.zeros([], jnp.int32),
                      m=tree_util.tree_map(zeros, params),
                      v=tree_util.tree_map(zeros, params))

def make_soap_pde_trainer(model, residual_fn, lr=1e-3, b1=.9, b2=.999, eps=1e-8):
    @jax.jit
    def step(p, s, batch):
        l, g_tot = jax.value_and_grad(lambda pp: batch['loss'](pp))(p)
        x_f, t_f = batch['x_f'], batch['t_f']
        g_res = jax.vmap(lambda xx, tt: jax.grad(lambda pp: residual_fn(pp, xx, tt, model))(p))(x_f, t_f)
        v = tree_util.tree_map(lambda v_old, g: b2*v_old + (1-b2)*jnp.mean(g**2,0), s.v, g_res)
        m = tree_util.tree_map(lambda m_old, g: b1*m_old + (1-b1)*g, s.m, g_tot)
        m_hat = tree_util.tree_map(lambda m_, v_: m_/(jnp.sqrt(v_)+eps), m, v)
        p = tree_util.tree_map(lambda w, mh: w - lr*mh, p, m_hat)
        return p, _SoapState(s.count+1, m, v), l
    return _init_soap_state, step

# ─────────────────────────────────────── SOAP‑Lib ──────────────────────────────

def make_soap_lib_trainer(model, residual_fn, lr=3e-3, betas=(.95,.95), weight_decay=.01, precond=10):
    if SoapLib is None:
        raise ImportError("pip install soap-optimizer to use SOAP‑Lib")
    opt = SoapLib(lr=lr, betas=betas, weight_decay=weight_decay, precondition_frequency=precond)
    @jax.jit
    def step(p, s, batch):
        l, g = jax.value_and_grad(lambda pp: batch['loss'](pp))(p)
        upd, s = opt.update(g, s, p); p = optax.apply_updates(p, upd)
        return p, s, l
    return opt.init, step

# ───────────────────────── Trainer‑factory collection helper ───────────────────

def get_optim_trainer_factories(model, residual_fn):
    # Wrapper to adapt our batch dict in main.py
    factories = {}
    factories['Adam']      = make_adam_trainer(model, residual_fn)
    factories['SOAP-PDE']  = make_soap_pde_trainer(model, residual_fn)
    if SoapLib is not None:
        factories['SOAP-Lib'] = make_soap_lib_trainer(model, residual_fn)
    return factories
