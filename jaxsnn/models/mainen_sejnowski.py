from jaxsnn.channels.common import channel_dynamics
import jaxsnn.base.implicit as implicit
import jaxsnn.base.funcutils as funcutils
import jaxsnn.base.tree_solver as tree_solver

import jax.numpy as jp
import tree_math
import dataclasses
import jax
from functools import partial

@tree_math.struct
class Point:
    x: jp.ndarray # (in um)
    y: jp.ndarray # (in um)
    z: jp.ndarray # (in um)
    radius: jp.ndarray # (in um)

@tree_math.struct
class Segment:
    prox: Point
    dist: Point
    identity: jp.ndarray


@tree_math.struct
class ChannelState:
    m : jp.ndarray
    h : jp.ndarray

@dataclasses.dataclass
@tree_math.struct
class MembraneChannels:
    na : ChannelState
    ca : ChannelState
    kv : ChannelState
    km : ChannelState
    kca : ChannelState

@dataclasses.dataclass
@tree_math.struct
class ChannelParameter:
    E: float
    g: jp.ndarray

@dataclasses.dataclass
@tree_math.struct
class BoltzmannParameter:
    v_half: float
    k: float
    a: float

@tree_math.struct
class NeuronParameters:
    Ra : jp.ndarray
    Cm : jp.ndarray
    na : ChannelParameter
    ca : ChannelParameter
    kv : ChannelParameter
    km : ChannelParameter
    kca : ChannelParameter
    leak : ChannelParameter
    ca_infty : float
    tau_ca : float

def channel_alpha(p: ChannelParameter, v):
    return p.a * (v - p.v_half) / (1 - jp.exp((p.v_half - v) / p.k))

def channel_beta(p: ChannelParameter, v):
    return -p.a * (v - p.v_half) / (1 - jp.exp((p.v_half - v) / p.k))

@tree_math.struct
class NeuronState:
    v : jp.ndarray
    ca_i : jp.ndarray
    channels: MembraneChannels


def x0(alpha, beta):
    def x0(v):
        return alpha(v) / (alpha(v) + beta(v))
    return x0

def tau(alpha, beta):
    def tau(v):
        return 1 / (alpha(v) + beta(v))

    return tau

def channel_dynamics_from_equilibrium_and_timeconstant(x_infty, tau):
    def dynamics(x, voltage):
        return (x_infty(voltage) - x) / tau(voltage)
    return dynamics

# Sodium channel dynamics
# alpha_na_activation = partial(alpha, BoltzmannParameter(v_half=-25, k=9.0, a=0.182))
# beta_na_activation = partial(beta, BoltzmannParameter(v_half=-25, k=9.0, a=-0.124))

def alpha_na_activation(v):
    return 0.182 * (v + 25) / (1 - jp.exp(-(v + 25) / 9))

def beta_na_activation(v):
    return -0.124 * (v + 25) / (1 - jp.exp((v + 25) / 9))

def alpha_na_inactivation(v):
    return 0.024 * (v + 40) / (1 - jp.exp(-(v + 40) / 5))

def beta_na_inactivation(v):
    return -0.0091 * (v + 65) / (1 - jp.exp((v + 65) / 5))

def b_na_infty_deactivation(v):
    return 1 / (1 + jp.exp((v + 55) / 6.2))

na_activation_dynamics = channel_dynamics(alpha_na_activation, beta_na_activation)
na_inactivation_dynamics = channel_dynamics_from_equilibrium_and_timeconstant(b_na_infty_deactivation, tau(alpha_na_inactivation, beta_na_inactivation))

def na_channel_dynamics(v, s):
    return ChannelState(
        m=na_activation_dynamics(s.m, v),  #alpha_na_activation(v) * (1 - s.m) - beta_na_activation(v) * s.m,
        h=na_inactivation_dynamics(s.h, v) # alpha_na_inactivation(v) * (1 - s.h) - beta_na_inactivation(v) * s.h
    )

def I_Na(v, s, p):
    return s.m**3 * s.h * p.g * (v - p.E)

# Calcium channel dynamics
def alpha_ca_activation(v):
    return 0.055 * (v + 27) / (1 - jp.exp(-(v + 27) / 3.8))

def beta_ca_activation(v):
    return 0.94 * jp.exp(-(v + 75) / 17)

def alpha_ca_deactivation(v):
    return 4.57e-4 * jp.exp(-(v + 13) / 50)

def beta_ca_deactivation(v):
    return 0.0065 / (1 + jp.exp(-(v + 15) / 28))

ca_activation_dynamics = channel_dynamics(alpha_ca_activation, beta_ca_activation)
ca_deactivation_dynamics = channel_dynamics(alpha_ca_deactivation, beta_ca_deactivation)

def ca_channel_dynamics(v, s):
    return ChannelState(
        m=alpha_ca_activation(v) * (1 - s.m) - beta_ca_activation(v) * s.m,
        h=alpha_ca_deactivation(v) * (1 - s.h) - beta_ca_deactivation(v) * s.h
    )

def I_Ca(v, s, p):
    return s.m**2 * s.h * p.g * (v - p.E)

# Potassium channel dynamics
def alpha_kv_activation(v):
    return 0.02 * (v - 25) / (1 - jp.exp(-(v - 25) / 9))

def beta_kv_activation(v):
    return -0.002 * (v - 25) / (1 - jp.exp((v - 25) / 9))

kv_activation_dynamics = channel_dynamics(alpha_kv_activation, beta_kv_activation)

def kv_channel_dynamics(v, s):
    return ChannelState(
        m=kv_activation_dynamics(s.m, v),
        h=jp.zeros_like(v)
    )

def I_Kv(v, s, p):
    return s.m * p.g * (v - p.E)

def alpha_km_activation(v):
    return 10**(-4) * (v + 30) / (1 - jp.exp(-(v + 30) / 9))

def beta_km_activation(v):
    return -1*(10**(-4)) * (v + 30) / (1 - jp.exp((v + 30) / 9))

def km_dynamics(v, s):
    return ChannelState(
        m=alpha_km_activation(v) * (1 - s.m) - beta_km_activation(v) * s.m,
        h=jp.zeros_like(v)
    )

def I_Km(v, s: ChannelState, p: ChannelParameter):
    return s.m * p.g * (v - p.E)

def alpha_kca_activation(v, ca_i):
    return 0.01 * ca_i

def beta_kca_activation(v, ca_i):
    return 0.02

def kca_dynamics(v, ca_i, s):
    return ChannelState(
        m=alpha_kca_activation(v, ca_i) * (1 - s.m) - beta_kca_activation(v, ca_i) * s.m,
        h=jp.zeros_like(v)
    )

def I_KCa(v, s: ChannelState, p: ChannelParameter):
    return s.m * p.g * (v - p.E)

def I_L(v, p: ChannelParameter):
    return p.g * (v - p.E)

def membrane_channel_dynamics(v, ca_i, s, p):
    return MembraneChannels(
            na=na_channel_dynamics(v, s.na),
            ca=ca_channel_dynamics(v, s.ca),
            kv=kv_channel_dynamics(v, s.kv),
            km=km_dynamics(v, s.km),
            kca=kca_dynamics(v, ca_i, s.kca)
    )

def I_total(v, s, p):
    return (
        I_Na(v, s.na, p.na) +
        I_Ca(v, s.ca, p.ca) +
        I_Kv(v, s.kv, p.kv) +
        I_Km(v, s.km, p.km) +
        I_KCa(v, s.kca, p.kca) + 
        I_L(v, p.leak)
    )



def na_equilibrium(v):
    return ChannelState(
        m=x0(alpha_na_activation, beta_na_activation)(v), 
        h=b_na_infty_deactivation(v))

def na_time_constant(v):
    return ChannelState(
        m=tau(alpha_na_activation, beta_na_activation)(v),
        h=tau(alpha_na_inactivation, beta_na_inactivation)(v)
    )

def ca_equilibrium(v):
    return ChannelState(
        m=x0(alpha_ca_activation, beta_ca_activation)(v),
        h=x0(alpha_ca_deactivation, beta_ca_deactivation)(v)
    )

def ca_time_constant(v):
    return ChannelState(
        m=tau(alpha_ca_activation, beta_ca_activation)(v),
        h=tau(alpha_ca_deactivation, beta_ca_deactivation)(v)
    )

def kv_equilibrium(v):
    return ChannelState(
        m=x0(alpha_kv_activation, beta_kv_activation)(v),
        h=jp.zeros_like(v)
    )

def km_equilibrium(v):
    return ChannelState(
        m=x0(alpha_km_activation, beta_km_activation)(v),
        h=jp.zeros_like(v)
    )

def km_equilibrium(v):
    return ChannelState(
        m=x0(alpha_km_activation, beta_km_activation)(v),
        h=jp.zeros_like(v)
    )

def kca_equilibrium(v, ca_i):
    alpha = alpha_kca_activation
    beta = beta_kca_activation

    return ChannelState(
        m=alpha(v, ca_i) / (alpha(v, ca_i) + beta(v, ca_i)),
        h=jp.zeros_like(v)
    )

# calcium dynamics
def ca_i_dynamics(v, s, p):
    return -1e5/2.0 * I_Ca(v, s.channels.ca, p.ca) + (p.ca_infty - s.ca_i) /  p.tau_ca


def conical_frustum_surface(segment : Segment):
    prox = jp.stack((segment.prox.x, segment.prox.y, segment.prox.z))
    dist = jp.stack((segment.dist.x, segment.dist.y, segment.dist.z))
    h = jp.linalg.norm(prox - dist, axis=0)
    s = jp.sqrt((segment.prox.radius - segment.dist.radius) ** 2 + h ** 2)
    area = (segment.prox.radius + segment.dist.radius) * s * jp.pi
    return area # in um^2

def conical_frustum_length(segment : Segment):
    prox = jp.stack((segment.prox.x, segment.prox.y, segment.prox.z))
    dist = jp.stack((segment.dist.x, segment.dist.y, segment.dist.z))
    h = jp.linalg.norm(prox - dist, axis=0)
    return h # in um

def number_of_segments(morph):
    num_segments = 0
    for i in range(morph.num_branches):
        num_segments += len(morph.branch_segments(i))
    return num_segments

def compute_parent_index_array_for_segments(morph):
    num_segments = number_of_segments(morph)
    parents = jp.arange(-1, num_segments)
    segment_index = 0
    for i in range(morph.num_branches):
        for child_index in morph.branch_children(i):
            parents = parents.at[child_index].set(segment_index)
        segment_index += len(morph.branch_segments(i))
    return parents

def morphology_to_segments(morph, filter_by_tag=None):
    nb = morph.num_branches
    result = []
    for i in range(nb):
        segments = morph.branch_segments(i)
        for seg in segments:
            if filter_by_tag and (seg.tag in filter_by_tag):
                result.append(Segment(
                    prox=Point(x=seg.prox.x, y=seg.prox.y, z=seg.prox.z, radius=seg.prox.radius),
                    dist=Point(x=seg.dist.x, y=seg.dist.y, z=seg.dist.z, radius=seg.dist.radius),
                    identity=seg.tag
                ))
            elif filter_by_tag is None:
                result.append(Segment(
                    prox=Point(x=seg.prox.x, y=seg.prox.y, z=seg.prox.z, radius=seg.prox.radius),
                    dist=Point(x=seg.dist.x, y=seg.dist.y, z=seg.dist.z, radius=seg.dist.radius),
                    identity=seg.tag
                ))
    return jax.tree_util.tree_map(lambda *v: jp.stack(v), *result)

def parent_to_directed_adjacency(parents):
    N = parents.shape[0]
    adj = jp.zeros((N, N))
    for i in range(1,N):
        adj = adj.at[parents[i], i].set(1)
    return adj

def parent_to_adjacency(parents):
    N = parents.shape[0]
    adj = jp.zeros((N, N))
    for i in range(1,N):
        adj = adj.at[parents[i], i].set(1)
        adj = adj.at[i, parents[i]].set(1)
    return adj

@dataclasses.dataclass
class NeuronModel(implicit.ImplicitExplicitODE):
    parameters: NeuronParameters
    
    conductance_matrix: jp.ndarray
    tm : tree_solver.TreeMatrix

    def explicit_terms(self, state):
        return NeuronState(
            v = -1/self.parameters.Cm * I_total(state.v, state.channels, self.parameters),
            ca_i = jp.zeros_like(state.ca_i), # ca_i_dynamics(state.v, state, self.parameters),
            channels = membrane_channel_dynamics(state.v, state.ca_i, state.channels, self.parameters),
        )

    def implicit_terms(self, state):
        return NeuronState(
            # v = 1/self.parameters.Cm * self.conductance_matrix @ state.v,
            v = tree_solver.tree_matmul(self.tm.d, self.tm.u, self.tm.p, state.v), 
            ca_i = jp.zeros_like(state.ca_i),
            channels = MembraneChannels(
                    na = ChannelState(m=jp.zeros_like(state.ca_i), h=jp.zeros_like(state.ca_i)),
                    ca = ChannelState(m=jp.zeros_like(state.ca_i), h=jp.zeros_like(state.ca_i)),
                    kv = ChannelState(m=jp.zeros_like(state.ca_i), h=jp.zeros_like(state.ca_i)),
                    km = ChannelState(m=jp.zeros_like(state.ca_i), h=jp.zeros_like(state.ca_i)),
                    kca = ChannelState(m=jp.zeros_like(state.ca_i), h=jp.zeros_like(state.ca_i))
                )
            )

    def implicit_solve(self, state, step_size):
        return NeuronState(
            # v = jp.linalg.solve(1 - step_size * 1/self.parameters.Cm * self.conductance_matrix, state.v),
            v = tree_solver.hines_solver(1 - step_size * 1/self.parameters.Cm * self.tm.d, - step_size * 1/self.parameters.Cm * self.tm.u, self.tm.p, state.v), 
            ca_i = state.ca_i,
            channels = state.channels
        )

@jax.jit
def compute_conductance_matrix(axial_conductances, parents) -> jp.array:
    m = tree_solver.tree_to_matrix(jp.zeros_like(axial_conductances), -axial_conductances[1:], parents)
    return m-jp.diag(jp.sum(m, axis=1))


@jax.jit
def compute_tree_conductance_matrix(axial_conductances, parents) -> jp.array:
    N = axial_conductances.shape[0]
    u = -axial_conductances
    diag = jp.zeros_like(u)
    
    def body_fun(i, diag):
        diag = diag.at[i].add(u[i - 1])
        diag = diag.at[parents[i]].add(u[i - 1])
        return diag

    diag = jax.lax.fori_loop(1, N, body_fun, diag)
    return tree_solver.TreeMatrix(d=diag, u=u, p=parents)




def test_morphology():
    import arbor
    import numpy as np

    morph = arbor.load_swc_arbor(
        "data/morphologies/allen/Cux2-CreERT2_Ai14-211772.05.02.01_674408996_m.swc"
    )

    segments = morphology_to_segments(morph)
    sorted_indices = np.argsort(segments.identity)
    parent = compute_parent_index_array_for_segments(morph)
    parent = parent[sorted_indices]
    segments = Segment(
        prox = Point(
            x=segments.prox.x[sorted_indices],
            y=segments.prox.y[sorted_indices],
            z=segments.prox.z[sorted_indices],
            radius=segments.prox.radius[sorted_indices]
        ),
        dist=Point(
            x=segments.dist.x[sorted_indices],
            y=segments.dist.y[sorted_indices],
            z=segments.dist.z[sorted_indices],
            radius=segments.prox.radius[sorted_indices]
        ),
        identity=segments.identity[sorted_indices]
    )

    soma_mask = segments.identity == 1
    dendritic_mask = (segments.identity == 3) | (segments.identity == 4)
    axon_mask = segments.identity == 2

    soma_segments = jax.tree_util.tree_map(lambda x: x[soma_mask], segments)
    axon_segments = jax.tree_util.tree_map(lambda x: x[axon_mask], segments)
    dendritic_segments = jax.tree_util.tree_map(lambda x: x[dendritic_mask], segments)


    # TODO: fix units

    mu_to_cm = 1e-4
    mu2_to_cm2 = 1e-8

    frustum_surface = conical_frustum_surface(soma_segments) * mu2_to_cm2
    soma_parameters = NeuronParameters(
        Cm = 10e3 * 0.75 * frustum_surface, # uF/cm^2 * cm^2 = uF
        Ra = 10e-3 * 150.0 / (mu_to_cm * conical_frustum_length(soma_segments)), # ohm.cm / cm * 10e-3 = kOhm
        na = ChannelParameter(E=50, g=20.0 * frustum_surface), # mS/cm^2 * cm^2 = mS
        ca = ChannelParameter(E=140, g=0.3 * frustum_surface), # mS/cm^2 * cm^2 = mS
        kv = ChannelParameter(E=-90, g=3.0 * frustum_surface), # mS/cm^2 * cm^2 = mS
        km = ChannelParameter(E=-90, g=200.0 * frustum_surface), # mS/cm^2 * cm^2 = mS
        kca = ChannelParameter(E=-90, g=3.0 * frustum_surface), # mS/cm^2 * cm^2 =  mS 
        leak = ChannelParameter(E=-70, g=0.3 * frustum_surface), # frustum_surface * 1/30000.0), # mS/cm^2 * cm^2 = mS
        ca_infty = 0.1, # uM
        tau_ca = 200.0, # ms
    )
    frustum_surface = conical_frustum_surface(dendritic_segments) * mu2_to_cm2
    dendritic_parameters = NeuronParameters(
        Cm = 10e3 * 0.75 * frustum_surface * mu2_to_cm2,
        Ra = 10e-3 * 150.0 / (mu_to_cm * conical_frustum_length(dendritic_segments)),
        na = ChannelParameter(E=50, g=20.0 * frustum_surface),
        kv = ChannelParameter(E=-90, g=3.0 * frustum_surface),
        km = ChannelParameter(E=-90, g=0 * frustum_surface),  
        ca = ChannelParameter(E=140, g=0.3 * frustum_surface),
        kca = ChannelParameter(E=-90, g=3 * frustum_surface), 
        leak = ChannelParameter(E=-70, g=0.3 * frustum_surface),  
        ca_infty = 0.1, # uM
        tau_ca = 200.0, # ms
    )
    frustum_surface = conical_frustum_surface(axon_segments) * mu2_to_cm2
    axon_hillock_parameters = NeuronParameters(
        Cm = 10e3 * 0.75 * frustum_surface * mu2_to_cm2, # uF/cm^2 * cm^2 = uF
        Ra = 10e-3 * 150.0 / (mu_to_cm * conical_frustum_length(axon_segments)), # ohm.cm / cm * 10e-3 = kOhm
        na = ChannelParameter(E=50, g=30000.0 * frustum_surface), # mS/cm^2 * cm^2 = mS
        kv = ChannelParameter(E=-90, g=2000.0 * frustum_surface), # mS/cm^2 * cm^2 = mS
        leak = ChannelParameter(E=-70, g=1/50.0* frustum_surface), # mS/cm^2 * cm^2 = mS
        km = ChannelParameter(E=-90, g=0 * frustum_surface), # mS/cm^2 * cm^2 = mS
        ca = ChannelParameter(E=140, g=0 * frustum_surface), # mS/cm^2 * cm^2 = mS
        kca = ChannelParameter(E=-90, g=0 * frustum_surface), # mS/cm^2 * cm^2 = mS
        ca_infty = 0.1, # uM
        tau_ca = 200.0, # ms
    )

    def concatenate_channel_parameters(c1: ChannelParameter, c2: ChannelParameter, c3: ChannelParameter):
        assert(c1.E == c2.E and c2.E == c3.E)
        return ChannelParameter(
            E = c1.E,
            g = jp.concatenate((c1.g, c2.g, c3.g))
        )

    # NOTE: The ordering does matter here !
    neuron_parameters = NeuronParameters(
        Cm = 0.75 * conical_frustum_surface(segments) * mu2_to_cm2, # uF/cm^2 * cm^2 = uF
        Ra = 150.0 / (mu_to_cm * conical_frustum_length(segments)), # ohm.cm / cm = ohm
        na = concatenate_channel_parameters(
            soma_parameters.na,
            axon_hillock_parameters.na,
            dendritic_parameters.na
        ),
        ca = concatenate_channel_parameters(
            soma_parameters.ca,
            axon_hillock_parameters.ca,
            dendritic_parameters.ca
        ),
        kv = concatenate_channel_parameters(
            soma_parameters.kv,
            axon_hillock_parameters.kv,
            dendritic_parameters.kv
        ),
        km = concatenate_channel_parameters(
            soma_parameters.km,
            axon_hillock_parameters.km,
            dendritic_parameters.km
        ),
        kca = concatenate_channel_parameters(
            soma_parameters.kca,
            axon_hillock_parameters.kca,
            dendritic_parameters.kca
        ),
        leak = concatenate_channel_parameters(
            soma_parameters.leak,
            axon_hillock_parameters.leak,
            dendritic_parameters.leak
        ),
        ca_infty = soma_parameters.ca_infty,
        tau_ca = soma_parameters.tau_ca
    )

    def initial_channel_state(v: jp.ndarray, ca_i: jp.ndarray):
        # return equilibrium 
        return MembraneChannels(
            na = na_equilibrium(v),
            ca = ca_equilibrium(v),
            kv = kv_equilibrium(v),
            km = km_equilibrium(v),
            kca = kca_equilibrium(v, ca_i)
        )

    conductance_matrix = compute_conductance_matrix(1.0/neuron_parameters.Ra, parent)
    model = NeuronModel(
        parameters=neuron_parameters,
        conductance_matrix=conductance_matrix,
        tm=compute_tree_conductance_matrix(1.0/neuron_parameters.Ra, parent)
    )
    initial_state = NeuronState(
        v = neuron_parameters.leak.E * jp.ones_like(neuron_parameters.Ra),
        ca_i = neuron_parameters.ca_infty * jp.ones_like(neuron_parameters.Ra),
        channels = initial_channel_state(neuron_parameters.leak.E * jp.ones_like(neuron_parameters.Ra), neuron_parameters.ca_infty * jp.ones_like(neuron_parameters.Ra))
    )

    
    dt = 0.001
    inner_steps = 1
    outer_steps = 10000
    time = dt * inner_steps * (1 + np.arange(outer_steps))

    from jax import config
    config.update("jax_debug_nans", True)
    config.update("jax_enable_x64", True)

    semi_implicit_step = implicit.imex_rk_sil3(model, dt)
    integrator = funcutils.trajectory(
        funcutils.repeated(semi_implicit_step, inner_steps), outer_steps
    )
    integrator = jax.jit(integrator)
    _, actual = integrator(initial_state)

    import matplotlib.pyplot as plt


    fig, ax = plt.subplots(1,3)
    ax[0].plot(time, actual.v)
    ax[1].plot(time, actual.ca_i)
    fig.savefig("test.png", dpi=400)




if __name__ == '__main__':
    test_morphology()
