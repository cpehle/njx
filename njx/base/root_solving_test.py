# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import jax.numpy as np
import jax
import math

import njx.base.root_solving as root_solving


ALL_TEST_PROBLEMS = [
    dict(
        testcase_name="root of unity",
        f=lambda x: x**20 - 1,
        df=lambda x: 20 * x**19,
        bounds=[(0.0, 4.0)],
        roots=[-1.0, 1.0],
    ),
    dict(
        testcase_name="sign",
        f=lambda x: np.sqrt(np.abs(x)) * np.sign(x),
        df=lambda x: 1 / np.sqrt(np.abs(x)),
        bounds=[(-1.0, 1.0)],
        roots=[0.0],
    ),
    dict(
        testcase_name="squareroot",
        f=lambda x: x**2 - 2,
        df=lambda x: 2 * x,
        bounds=[(0.0, 2.0)],
        roots=[-math.sqrt(2), math.sqrt(2)],
    ),
    dict(
        testcase_name="closeroot",
        f=lambda x: x**2 - 1e-8,
        df=lambda x: 2 * x,
        bounds=[(0.0, 2.0)],
        roots=[-math.sqrt(1e-8), math.sqrt(1e-8)],
    ),
    dict(
        testcase_name="xexp",
        f=lambda x: x * np.exp(-x),
        df=lambda x: np.exp(-x) - x * np.exp(-x),
        bounds=[(-1.0, 1.0)],
        roots=[0.0],
    ),
    # TODO: Doesn't work!
    # dict(
    #     testcase_name="xpow7",
    #     f=lambda x: (x - 1)**7,
    #     df=lambda x: 7*(x-1)**6,
    #     bounds = [(0.0,2.0)],
    #     roots=[1.0]
    # )
]

ALL_BOUNDED_ROOT_METHODS = [root_solving.bisection, root_solving.illinois_method]

ALL_GRADIENT_BASED_METHODS = [root_solving.newton_1d]


class RootSolvingTest(parameterized.TestCase):
    @parameterized.named_parameters(ALL_TEST_PROBLEMS)
    def test_root_solving(self, f, df, bounds, roots):
        bound = bounds[0]
        expected = roots[-1]
        tol = 0.01
        for solver in ALL_BOUNDED_ROOT_METHODS:
            actual = solver(f, bound[0], bound[1], tol)
            onp.testing.assert_allclose(expected, actual, atol=tol)

    @parameterized.named_parameters(ALL_TEST_PROBLEMS)
    def test_gradient_root_solving(self, f, df, bounds, roots):
        bound = bounds[0]
        expected = roots[-1]
        tol = 0.01
        for solver in ALL_GRADIENT_BASED_METHODS:
            actual = solver(f, bound[0], tol=tol)
            onp.testing.assert_allclose(expected, actual, atol=tol)


def test_newton_1d():
    expected = math.sqrt(2)
    def f(x):
        return x**2 - 2
    tol = 0.1
    actual = root_solving.newton_1d(f, 1.0, tol=tol)
    onp.testing.assert_allclose(actual, expected, atol=tol)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    absltest.main()
