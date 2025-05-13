# main.py
"""Top‑level runner for PINN benchmarks.

Usage examples
--------------
python main.py --pde burgers --model basic
python main.py --pde allen_cahn --model basic

Required companion modules
--------------------------
model.py        – PINN architectures
optimiser.py    – Adam / SOAP‑PDE / SOAP‑Lib trainers
residual.py     – PDE residual functions
dataloader.py   – Loads <pde>.mat and builds training batches

Place the corresponding <pde>.mat file (e.g. burgers.mat, allen_cahn.mat) in your
working directory before running.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from dataloader import build_dataset, load_mat_data
from residual import get_residual
from model import get_model_class
from optimiser import get_optim_trainer_factories

# ────────────────────────────────────────────────────────────────────────────────
#  Training loop (optimizer‑agnostic)
# ────────────────────────────────────────────────────────────────────────────────

def train(rng, init_fn, step_fn, dataset, model, epochs=100, batch_size=1024):
    params = model.init(rng, jnp.zeros((1,)), jnp.zeros((1,)))
    state  = init_fn(params)
    (x_f, _), *_ = dataset
    n_batches = x_f.shape[0] // batch_size
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(x_f.shape[0])
        epoch_loss = 0.0
        for i in range(n_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            batch = ((dataset[0][0][idx], dataset[0][1][idx]), dataset[1], dataset[2])
            params, state, loss = step_fn(params, state, batch)
            epoch_loss += float(loss)
        if ep == 1 or ep % 200 == 0:
            print(f"Epoch {ep:4d} | Loss {epoch_loss / n_batches:.3e}")
    return params

# ────────────────────────────────────────────────────────────────────────────────
#  Evaluation helpers
# ────────────────────────────────────────────────────────────────────────────────

def evaluate(model, params, X, T):
    XT = jnp.stack([X.flatten(), T.flatten()], -1)
    return np.array(model.apply(params, XT[:, 0], XT[:, 1])).reshape(X.shape)


def compare_error_plots(pde: str, model, trained_params: dict):
    x, t, fdm = load_mat_data(pde)
    X, T = np.meshgrid(x, t)
    for name, params in trained_params.items():
        U_pred = evaluate(model, params, X, T)
        err = np.abs(fdm - U_pred)
        plt.figure(figsize=(6, 4))
        plt.imshow(err, extent=[x[0], x[-1], t[0], t[-1]], origin='lower', aspect='auto', cmap='plasma')
        plt.colorbar(); plt.xlabel('x'); plt.ylabel('t')
        plt.title(f"|FDM - {name}| error")
        plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='PINN benchmark runner')
    parser.add_argument('--pde',   required=True, choices=['burgers', 'allen_cahn'], help='PDE type')
    parser.add_argument('--model', default='basic', choices=['basic'],              help='Model architecture')
    args = parser.parse_args()

    rng_key = jax.random.PRNGKey(0)

    # Dataset & residual
    dataset      = build_dataset(rng_key, args.pde)
    residual_fn  = get_residual(args.pde)

    # Model instance
    ModelClass = get_model_class(args.model)
    model      = ModelClass()

    # Optimizer trainers
    trainer_factories = get_optim_trainer_factories(model, residual_fn)

    # Train with each optimizer
    trained_params = {}
    for name, (init_fn, step_fn) in trainer_factories.items():
        if init_fn is None or step_fn is None:
            print(f"[!] Skipping {name} (optimizer unavailable)")
            continue

    print(f"\n🚀 Training with {name}…")
    params = train(rng_key, init_fn, step_fn, dataset, model)
    trained_params[name] = params

    # Compare predictions
    compare_error_plots(args.pde, model, trained_params)
    from optimiser import show_all_figures
    show_all_figures()


if __name__ == '__main__':
    main()
