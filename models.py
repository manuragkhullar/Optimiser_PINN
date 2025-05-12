"""model.py – PINN architectures"""
import jax.numpy as jnp
from flax import linen as nn

class BasicPINN(nn.Module):
    hidden_sizes: tuple = (64, 64, 64)

    @nn.compact
    def __call__(self, x, t):
        h = jnp.stack([x, t], -1)
        for w in self.hidden_sizes:
            h = nn.tanh(nn.Dense(w)(h))
        return nn.Dense(1)(h).squeeze(-1)


def get_model_class(name: str):
    if name == 'basic':
        return BasicPINN
    else:
        raise ValueError(f"Unknown model architecture: {name}")
