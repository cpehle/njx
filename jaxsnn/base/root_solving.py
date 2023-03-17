# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

import jax
import jax.numpy as jnp
from jaxsnn.base.types import ArrayLike
import tree_math
import dataclasses


def linear_interpolation(f_a, f_b, a, b, x):
    return (x - a) / (b - a) * f_b + (b - x) / (b - a) * f_a


def linear_interpolated_root(f_a, f_b, a, b):
    return (a * f_b - b * f_a) / f_b - f_a


@dataclasses.dataclass
@tree_math.struct
class NewtonState:
    it: int
    x: ArrayLike


def newton_1d(f, x0, tol):
    initial_state = NewtonState(it=0, x=x0)

    def cond(state):
        return jnp.logical_and((jnp.abs(f(state.x)) > tol), state.it < 1e6)

    def body(state):
        it, x = state.it, state.x
        fx, dfx = f(x), jax.grad(f)(x)
        step = fx / (dfx + 0.0001)
        new_state = NewtonState(it + 1, x - step)
        return new_state

    return jax.lax.while_loop(
        cond,
        body,
        initial_state,
    ).x


def newton_nd(f, x0):
    initial_state = (0, x0)

    def cond(state):
        it, x = state
        return it < 10

    def body(state):
        it, x = state
        fx, dfx = f(x), jax.grad(f)(x)
        step = jax.numpy.linalg.solve(dfx, -fx)

        new_state = it + 1, x + step
        return new_state

    return jax.lax.while_loop(
        cond,
        body,
        initial_state,
    )[1]


def bisection(f, x_min, x_max, tol):
    """Bisection root finding method

    Based on the intermediate value theorem, which
    guarantees for a continuous function that there
    is a zero in the interval [x_min, x_max] as long
    as sign(f(x_min)) != sign(f(x_max)).

    NOTE: We do not check the precondition sign(f(x_min)) != sign(f(x_max)) here

    f: function for which we want to find a root in the interval [x_min,x_max].
    x_min: lower bound
    x_max: upper bound
    eps: resolution or tolerance of the root finding method

    """
    initial_state = (0, x_min, x_max)  # (iteration, x)

    def cond(state):
        it, x_min, x_max = state
        return jnp.abs(f(x_min)) > tol  # it > 10

    def body(state):
        it, x_min, x_max = state
        x = (x_min + x_max) / 2

        sfxm = jnp.sign(f(x_min))
        sfx = jnp.sign(f(x))

        x_min = jnp.where(sfx == sfxm, x, x_min)
        x_max = jnp.where(sfx == sfxm, x_max, x)

        new_state = (it + 1, x_min, x_max)
        return new_state

    return jax.lax.while_loop(
        cond,
        body,
        initial_state,
    )[1]


@dataclasses.dataclass
@tree_math.struct
class State:
    a: ArrayLike
    b: ArrayLike
    fa: ArrayLike
    fb: ArrayLike


def illinois_method(f, a, b, eps):
    """Illinois root finding method

    This is a modified version of the secant method. Some version of this is
    used in the SUNDIALS suite.

    Reference: Kathie L. Hiebert and Lawrence F. Shampine, Implicitly Defined Output Points for Solutions of ODEs, Sandia National Laboratory Report SAND80-0180, February 1980.
    TODO: We need to fully check edge cases and introduce a suitable set of benchmark problems.

    f: function for which we want to find a root in the interval [a,b].
    a: lower bound
    b: upper bound
    eps: resolution or tolerance of the root finding method

    """
    fa = f(a)
    fb = f(b)
    (a, fa, b, fb) = jax.lax.cond(
        jnp.abs(fa) > jnp.abs(fb), lambda: (b, fb, a, fa), lambda: (a, fa, b, fb)
    )
    init = State(a=a, b=b, fa=fa, fb=fb)

    def cond(state: State):
        return jnp.abs(state.b - state.a) > eps

    def body_fun(state: State):
        a = state.a
        b = state.b
        fa = state.fa
        fb = state.fb

        c = a - (fa * (b - a)) / (fb - fa)
        fc = f(c)
        b, fb = jax.lax.cond(fa * fc <= 0, lambda: (a, fa), lambda: (b, 0.5 * fb))

        state = State(a=c, b=b, fa=fc, fb=fb)
        return state

    return jax.lax.while_loop(cond, body_fun, init).a
