# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

from jaxsnn.base.tree_solver import hines_solver, tree_to_matrix, tree_matmul
import jaxsnn.morph.generate_morphologies as gm

import jax.numpy as jnp
import numpy as onp
import arbor

from absl.testing import absltest


def test_tree_to_matrix():
    N = 3
    d = 2 * onp.ones(N)
    u = onp.ones(N - 1)
    p = onp.arange(-1, N, 1)

    expected = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
    actual = tree_to_matrix(d, u, p)
    onp.testing.assert_allclose(expected, actual)


def test_tree_to_matrix_2():
    morph = gm.y_geometry()
    tm = gm.compute_tree_matrix(morph)

    expected = [[-1, 1, 0, 0], [1, -3, 1, 1], [0, 1, -1, 0], [0, 1, 0, -1]]

    actual = tree_to_matrix(tm.d, tm.u, tm.p)
    onp.testing.assert_allclose(expected, actual)


def test_hines_solver():
    N = 10
    d = 2 * onp.random.randn(N)
    u = onp.random.randn(N - 1)
    b = onp.random.randn(N)
    p = onp.arange(-1, N, 1)
    a = onp.diag(d, 0) + onp.diag(u, 1) + onp.diag(u, -1)

    x = hines_solver(jnp.array(d), jnp.array(u), jnp.array(p), jnp.array(b))
    x_ = jnp.linalg.solve(a, b)

    # TODO: This is a rather liberal error tolerance...
    onp.testing.assert_allclose(x, x_, rtol=1e-4)


def test_hines_solver_2():
    morph = gm.y_geometry()
    tm = gm.compute_tree_matrix(morph)
    b = onp.random.randn(4)
    a = tree_to_matrix(tm.d + 0.01, tm.u, tm.p)

    x = hines_solver(tm.d + 0.01, tm.u, tm.p, jnp.array(b))
    x_ = jnp.linalg.solve(a, jnp.array(b))

    # TODO: This is a rather liberal error tolerance...
    onp.testing.assert_allclose(x, x_, rtol=1e-4)


def test_hines_solver_3():
    morph = gm.y_geometry()
    tm = gm.compute_tree_matrix(morph)
    b = onp.random.randn(4)

    # TODO: why is the 0.01 needed!?
    a = tree_to_matrix(tm.d + 0.01, tm.u, tm.p)
    x = hines_solver(tm.d + 0.01, tm.u, tm.p, jnp.array(b))
    x_ = jnp.linalg.solve(a, jnp.array(b))

    # TODO: This is a rather liberal error tolerance...
    onp.testing.assert_allclose(x, x_, rtol=1e-4)


def test_hines_solver_4():
    morph = gm.branched_geometry()
    tm = gm.compute_tree_matrix(morph)
    b = onp.random.randn(tm.d.shape[0])

    # TODO: why is the 0.01 needed!?
    a = tree_to_matrix(tm.d + 0.01, tm.u, tm.p)
    x = hines_solver(tm.d + 0.01, tm.u, tm.p, jnp.array(b))
    x_ = jnp.linalg.solve(a, jnp.array(b))

    # TODO: This is a rather liberal error tolerance...
    onp.testing.assert_allclose(x, x_, rtol=1e-2)


def test_hines_solver_5():
    morph = gm.swc_geometry(
        "data/morphologies/allen/Cux2-CreERT2_Ai14-211772.05.02.01_674408996_m.swc"
    )
    tm = gm.compute_tree_matrix(morph, policy=arbor.cv_policy_fixed_per_branch(1))
    b = onp.random.randn(tm.d.shape[0])

    # TODO: why is the 0.01 needed!?
    a = tree_to_matrix(tm.d + 0.01, tm.u, tm.p)
    x = hines_solver(tm.d + 0.01, tm.u, tm.p, jnp.array(b))
    x_ = jnp.linalg.solve(a, jnp.array(b))

    # TODO: This is a rather liberal error tolerance...
    onp.testing.assert_allclose(x, x_, rtol=1e-1)


def test_hines_solver_6():
    morph = gm.swc_geometry(
        "data/morphologies/allen/Cux2-CreERT2_Ai14-211772.05.02.01_674408996_m.swc"
    )
    tm = gm.compute_tree_matrix(morph, policy=arbor.cv_policy_fixed_per_branch(3))
    b = onp.random.randn(tm.d.shape[0])

    # TODO: why is the 0.01 needed!?
    a = tree_to_matrix(tm.d + 0.01, tm.u, tm.p)
    x = hines_solver(tm.d + 0.01, tm.u, tm.p, jnp.array(b))
    x_ = jnp.linalg.solve(a, jnp.array(b))

    # TODO: This is a rather liberal error tolerance...
    onp.testing.assert_allclose(x, x_, rtol=1e-2)


def test_tree_matmul():
    N = 3
    d = 2 * onp.ones(N)
    u = onp.ones(N - 1)
    p = onp.arange(-1, N, 1)

    mat = tree_to_matrix(d, u, p)
    a = jnp.array(onp.random.randn(3))
    expected = jnp.dot(mat, a)
    actual = tree_matmul(d, u, p, a)

    onp.testing.assert_allclose(expected, actual)


def test_tree_matmul_2():
    morph = gm.y_geometry()
    tm = gm.compute_tree_matrix(morph)
    a = onp.random.randn(4)
    mat = tree_to_matrix(tm.d, tm.u, tm.p)

    expected = jnp.dot(mat, a)
    actual = tree_matmul(tm.d, tm.u, tm.p, a)

    onp.testing.assert_allclose(expected, actual, rtol=0.1)


if __name__ == "__main__":
    absltest.main()
