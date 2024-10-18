# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

import math
import tree_math

def channel_time_constant(alpha, beta, v_shift=0.0, q=1.0):
    def tau(v):
        v = v + v_shift
        return 1 / q * 1 / (alpha(v) + beta(v))

    return tau


def channel_equilibrium(alpha, beta, v_shift=0.0):
    def x_inf(v):
        v = v + v_shift
        return alpha(v) / (alpha(v) + beta(v))

    return x_inf


def q_conversion(target_deg_c, original_deg_c, q10):
    return math.exp(q10, (target_deg_c - original_deg_c) / 10)


def channel_dynamics(alpha, beta):
    def dynamics(x, voltage):
        return alpha(voltage) * (1 - x) - beta(voltage) * x
    return dynamics