import jax.numpy as jnp
from jax import random
from jax.lax import scan
from jaxsnn.functional.lif import LIFState, lif_step


def LIF(out_dim, scale_in=0.7, scale_rec=0.2):
    """Layer constructor function for a lif (leaky-integrated-fire) layer."""

    def init_fn(rng, input_shape):
        i_key, r_key = random.split(rng)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        recurrent_weights = scale_rec * random.normal(r_key, (out_dim, out_dim))
        return out_dim, (input_weights, recurrent_weights)

    def apply_fn(params, inputs, **kwargs):
        batch = inputs.shape[1]
        shape = (batch, out_dim)
        state = LIFState(jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        (state, _), spikes = scan(lif_step, (state, params), inputs)

        return spikes

    return init_fn, apply_fn


def LIFStep(out_dim, scale_in=0.7, scale_rec=0.2):
    """Layer constructor function for a lif (leaky-integrated-fire) layer."""

    def init_fn(rng, input_shape):
        i_key, r_key = random.split(rng)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        recurrent_weights = scale_rec * random.normal(r_key, (out_dim, out_dim))
        return out_dim, (input_weights, recurrent_weights)

    def state_fn(batch_size):
        shape = (batch_size, out_dim)
        state = LIFState(jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        return state

    def apply_fn(state, params, inputs, **kwargs):
        return lif_step((state, params), inputs)

    return init_fn, apply_fn, state_fn
