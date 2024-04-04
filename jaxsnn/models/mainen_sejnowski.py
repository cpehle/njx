from jaxsnn.channels.common import channel_dynamics
import jaxsnn.base.implicit as implicit
import jaxsnn.base.tree_solver as tree_solver

import jax.numpy as jp
import tree_math
import dataclasses
import jax

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

@tree_math.struct
class DendriticMembraneChannels:
    na : ChannelState
    ca : ChannelState
    kv : ChannelState
    kca : ChannelState

@dataclasses.dataclass
@tree_math.struct
class DendriticState:
    ca_i : jp.ndarray
    channels : DendriticMembraneChannels

@tree_math.struct
class SomaticMembraneChannels:
    na : ChannelState
    ca : ChannelState
    kv : ChannelState
    km : ChannelState
    kca : ChannelState

@tree_math.struct
class SomaticState:
    ca_i : jp.ndarray
    channels : SomaticMembraneChannels

@tree_math.struct
class ChannelParameter:
    E: float
    g: jp.ndarray

@tree_math.struct
class SomaticParameters:
    na : ChannelParameter
    ca : ChannelParameter
    kv : ChannelParameter
    km : ChannelParameter
    kca : ChannelParameter
    leak : ChannelParameter
    ca_infty : float
    tau_ca : float

@tree_math.struct
class DendriticParameters:
    na : ChannelParameter
    ca : ChannelParameter
    kv : ChannelParameter
    kca : ChannelParameter
    leak : ChannelParameter
    ca_infty : float
    tau_ca : float

@tree_math.struct
class AxonhillockParameters:
    na : ChannelParameter
    kv : ChannelParameter
    leak : ChannelParameter

@tree_math.struct
class AxonhillockMembraneChannels:
    na : ChannelState
    kv : ChannelState

@tree_math.struct
class AxonhillockState:
    channels : AxonhillockMembraneChannels

@tree_math.struct
class CompartmentState:
    soma : jp.ndarray
    axon_hillock : jp.ndarray
    dendritic : jp.ndarray

@tree_math.struct
class CompartmentParameter:
    soma : jp.ndarray
    axon_hillock : jp.ndarray
    dendritic : jp.ndarray

@tree_math.struct
class NeuronParameters:
    Ra : CompartmentParameter
    Cm : CompartmentParameter
    soma : SomaticParameters
    dendritic : DendriticParameters
    axon_hillock : AxonhillockParameters

@tree_math.struct
class MainenSejnowskiState:
    v : CompartmentState
    dendritic : DendriticState
    somatic : SomaticState
    axon_hillock : AxonhillockState



# def soma_parameters():
#     return SomaticParameters(
#         na = ChannelParameter(E=50, g=20),
#         ca = ChannelParameter(E=140, g=0.3),
#         kv = ChannelParameter(E=-90, g=3),
#         km = ChannelParameter(E=-90, g=200),
#         kca = ChannelParameter(E=-90, g=3),
#         leak = ChannelParameter(E=-70, g=0.1),
#         ca_infty = 0.1, # uM
#         tau_ca = 200, # ms
#     )
# 
# def dendritic_parameters():
#     return DendriticParameters(
#         na = ChannelParameter(E=50, g=20),
#         ca = ChannelParameter(E=140, g=0.3),
#         kv = ChannelParameter(E=-90, g=3),
#         kca = ChannelParameter(E=-90, g=3),
#         leak = ChannelParameter(E=-70, g=0.1),
#         ca_infty = 0.1, # uM
#         tau_ca = 200, # ms
#     )
# 
# def axon_hillock_parameters():
#     return AxonhillockParameters(
#         na = ChannelParameter(E=50, g=30000),
#         kv = ChannelParameter(E=-90, g=2000)
#     )


# longitudinal resistance
# RL = rL * dx / (2 * pi * r^2)
# Rm = rm * / (2 * pi * r * dx)

# E_Leak = -70 # mV
# Ek = -90 # mV
# ENa = 50 # mV
# ECa = 140 # mV

# specific membrane capacitance
Cm = 0.75 # uF/cm^2
# myelinated axon segments
Cm = 0.02 # uF/cm^2  

# specific membrane resistance
Rm = 30000 # ohm.cm^2
# axon segments
Rm = 50 # ohm.cm^2
# specific axial resistance
Ra = 150 # ohm.cm
# Conductance densities
# Dendrites
gNa = 20  # mS/cm^2
gCa = 0.3 # mS/cm^2
gKa = 3   # mS/cm^2
gKv = 0.1 # mS/cm^2
# Soma
gKm = 200 # mS/cm^2
# Axon hillock and initial segment
gNa = 30000 # mS/cm^2
gKv = 2000 # mS/cm^2
# Nodes of Ranvier
gNa = 30000 # mS/cm^2


# Sodium channel dynamics
def alpha_na_activation(v):
    return 0.182 * (v + 25) / (1 - jp.exp(-(v + 25) / 9))

def beta_na_activation(v):
    return -0.124 * (v + 25) / (1 - jp.exp(-(v + 25) / 9))

def alpha_na_inactivation(v):
    return 0.024 * (v + 40) / (1 - jp.exp(-(v + 40) / 5))

def beta_na_inactivation(v):
    return -0.0091 * (v + 65) / (1 - jp.exp((v + 65) / 5))

def b_na_infty_deactivation(v):
    return 1 / (1 + jp.exp(-(v + 55) / 6.2))

na_activation_dynamics = channel_dynamics(alpha_na_activation, beta_na_activation)
na_inactivation_dynamics = channel_dynamics(alpha_na_inactivation, beta_na_inactivation)

def na_channel_dynamics(v, s):
    return ChannelState(
        m=na_activation_dynamics(v, s.m),
        h=na_inactivation_dynamics(v, s.n)
    )

def I_Na(v, s, p):
    return jp.pow(s.m,3) * s.h * p.gNa * (v - p.ENa)

# Calcium channel dynamics
def alpha_ca_activation(v):
    return 0.055 * (v + 27) / (1 - jp.exp(-(v + 27) / 3.8))

def beta_ca_activation(v):
    return 0.94 * jp.exp(-(v + 75) / 17)

def alpha_ca_deactivation(v):
    return 4.57 * 10e-4 * jp.exp(-(v + 13) / 50)

def beta_ca_deactivation(v):
    return 0.0065 / (1 + jp.exp(-(v + 15) / 28))

ca_activation_dynamics = channel_dynamics(alpha_ca_activation, beta_ca_activation)
ca_deactivation_dynamics = channel_dynamics(alpha_ca_deactivation, beta_ca_deactivation)

def ca_channel_dynamics(v, s):
    return ChannelState(
        m=ca_activation_dynamics(v, s.m),
        h=ca_deactivation_dynamics(v, s.h)
    )

def I_Ca(v, s, p):
    return jp.pow(s.m,2) * s.h * p.gCa * (v - p.ECa)

# Potassium channel dynamics
def alpha_kv_activation(v):
    return 0.02 * (v - 25) / (1 - jp.exp(-(v + 25) / 9))

def beta_kv_activation(v):
    return -0.002 * (v - 25) / (1 - jp.exp(-(v + 25) / 9))

kv_activation_dynamics = channel_dynamics(alpha_kv_activation, beta_kv_activation)

def kv_channel_dynamics(v, s):
    return ChannelState(
        m=kv_activation_dynamics(v, s.m),
        h=0.0
    )

def I_Kv(v, s, p):
    return s.m * p.gKv * (v - p.Ek)

def alpha_km_activation(v):
    return 1e-4 * (v + 30) / (1 - jp.exp(-(v + 30) / 9))

def beta_km_activation(v):
    return -1.1e-4 * (v + 30) / (1 - jp.exp(-(v + 30) / 9))

def km_dynamics(v, s):
    return ChannelState(
        m=alpha_km_activation(v) * (1 - s.m) - beta_km_activation(v) * s.m,
        h=0.0
    )

def I_Km(v, s: ChannelState, p: ChannelParameter):
    return s.m * p.g * (v - p.E)

def alpha_kca_activation(v, ca_i):
    return 0.01 * ca_i

def beta_kca_activation(v, ca_i):
    return 0.02

def kca_dynamics(v, s):
    return ChannelState(
        m=alpha_kca_activation(v, s.ca_i) * (1 - s.m) - beta_kca_activation(v, s.ca_i) * s.m,
        h=0.0
    )

def I_KCa(v, s: ChannelState, p: ChannelParameter):
    return s.m * p.g * (v - p.E)

def I_L(v, p: ChannelParameter):
    return p.g * (v - p.E)

def dendritic_channel_dynamics(v, s, p):
    return DendriticMembraneChannels(
            na=na_channel_dynamics(v, s.channels.na),
            ca=ca_channel_dynamics(v, s.channels.ca),
            kv=kv_channel_dynamics(v, s.channels.kv),
            kca=kca_dynamics(v, s.channels.kca)
    )

def somatic_channel_dynamics(v, s, p):
    return SomaticMembraneChannels(
            na=na_channel_dynamics(v, s.channels.na),
            ca=ca_channel_dynamics(v, s.channels.ca),
            kv=kv_channel_dynamics(v, s.channels.kv),
            km=km_dynamics(v, s.channels.km),
            kca=kca_dynamics(v, s.channels.kca)
    )


def axonhillock_channel_dynamics(v, s, p):
    return AxonhillockMembraneChannels(
        na = na_channel_dynamics(v, s.channels.na),
        kv = kv_channel_dynamics(v, s.channels.kv)
    )

def dendritic_current(v, s, p):
    return (
        I_Na(v, s.channels.na, p.na) +
        I_Ca(v, s.channels.ca, p.ca) +
        I_Kv(v, s.channels.kv, p.kv) +
        I_KCa(v, s.channels.kca, p.kca) +
        I_L(v, p.leak)
    )

def somatic_current(v, s, p):
    return (
        I_Na(v, s.channels.na, p.na) +
        I_Ca(v, s.channels.ca, p.ca) +
        I_Kv(v, s.channels.kv, p.kv) +
        I_Km(v, s.channels.km, p.km) +
        I_KCa(v, s.channels.kca, p.kca) + 
        I_L(v, p.leak)
    )

def axon_hillock_current(v, s, p):
    return (
        I_Na(v, s.channels.na, p.na) +
        I_Kv(v, s.channels.kv, p.kv) + 
        I_L(v, p.leak)
    )

# calcium dynamics
def ca_i_dynamics(v, s, p):
    return -1e5/2 * I_Ca(v, s.channels.ca, p.ca) - (s.ca_i - p.ca_infty) /  p.tau_ca


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

    def explicit_terms(self, state):
        return MainenSejnowskiState(
            v = CompartmentState(
                soma = 1/self.parameters.Cm.soma * somatic_current(state.v.somatic, state.somatic.channels, self.parameters.soma),
                dendritic = 1/self.parameters.Cm.dendritic * dendritic_current(state.v.dendritic, state.dendritic.channels, self.parameters.dendritic),
                axon_hillock = 1/self.parameters.Cm.axon_hillock * axon_hillock_current(state.v.axon_hillock, state.axon_hillock.channels, self.parameters.axon_hillock)                                            
            ),
            dendritic = DendriticState(
                ca_i = ca_i_dynamics(state.dendritic.v, state.dendritic, self.parameters.dendritic),
                channels = dendritic_channel_dynamics(state.dendritic.v, state.dendritic.channels, self.parameters.dendritic),
            ),
            somatic = SomaticState(
                ca_i = ca_i_dynamics(state.somatic.v, state.somatic, self.parameters.somatic),
                channels = somatic_channel_dynamics(state.somatic.v, state.somatic.channels, self.parameters.soma),
            ),
            axon_hillock = AxonhillockState(
                channels = axonhillock_channel_dynamics(state.v.axon_hillock, state.axon_hillock.channels, self.parameters.axon_hillock)
            )
        )

    def implicit_terms(self, state):
        @tree_math.wrap
        def multiply(voltage_state):
            return self.conductance_matrix @ voltage_state

        return MainenSejnowskiState(
            v = multiply(state.v),
            dendritic = DendriticState(
                ca_i = 0,
                channels = DendriticMembraneChannels(
                    na = ChannelState(m=0, h=0),
                    ca = ChannelState(m=0, h=0),
                    kv = ChannelState(m=0, h=0),
                    kca = ChannelState(m=0, h=0)
                )
            ),
            somatic = SomaticState(
                ca_i = 0,
                channels = SomaticMembraneChannels(
                    na = ChannelState(m=0, h=0),
                    ca = ChannelState(m=0, h=0),
                    kv = ChannelState(m=0, h=0),
                    km = ChannelState(m=0, h=0),
                    kca = ChannelState(m=0, h=0)
                )
            ),
            axon_hillock = AxonhillockState(
                channels = AxonhillockMembraneChannels(
                    na = ChannelState(m=0, h=0),
                    kv = ChannelState(m=0, h=0)
                )
            )
        )


    def implicit_solve(self, state, step_size):
        @tree_math.wrap
        def solve(voltage_state):
            return jax.linalg.solve(1 - step_size * self.conductance_matrix, voltage_state)

        return MainenSejnowskiState(
            v = solve(state.v),
            dendritic = state.dendritic,
            somatic = state.somatic,
            axon_hillock = state.axon_hillock
        )

@jax.jit
def compute_conductance_matrix(axial_conductances, parents) -> jp.array:
    m = tree_solver.tree_to_matrix(jp.zeros_like(axial_conductances), -axial_conductances[1:], parents)
    return m-jp.diag(jp.sum(m, axis=1))


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

    print(dendritic_segments.shape)
    print(axon_segments.shape)


    # TODO: fix units
    frustum_surface = conical_frustum_surface(soma_segments)    
    soma_parameters = SomaticParameters(
        na = ChannelParameter(E=50, g=20 * frustum_surface), # mS/cm^2 * um^2 = 10^4 mS
        ca = ChannelParameter(E=140, g=0.3 * frustum_surface), # mS/cm^2 * um^2 = 10^4 mS
        kv = ChannelParameter(E=-90, g=3 * frustum_surface), # mS/cm^2 * um^2 = 10^4 mS
        km = ChannelParameter(E=-90, g=200 * frustum_surface), # mS/cm^2 * um^2 = 10^4 mS
        kca = ChannelParameter(E=-90, g=3 * frustum_surface), # mS/cm^2 * um^2 = 10^4 mS 
        leak = ChannelParameter(E=-70, g=1/30000.0 *frustum_surface), # mS/cm^2 * um^2 = 10^4 mS = 10 ohm^{-1}
        ca_infty = 0.1, # uM
        tau_ca = 200, # ms
    )
    frustum_surface = conical_frustum_surface(dendritic_segments)
    dendritic_parameters = DendriticParameters(
        na = ChannelParameter(E=50, g=20 * frustum_surface), # mS/cm^2 * um^2 = mS
        ca = ChannelParameter(E=140, g=0.3 * frustum_surface), # mS/cm^2 * um^2 = mS
        kv = ChannelParameter(E=-90, g=3 * frustum_surface), # mS/cm^2 * um^2 = mS
        kca = ChannelParameter(E=-90, g=3 * frustum_surface), # mS/cm^2 * um^2 = mS
        leak = ChannelParameter(E=-70, g=1/30000.0 * frustum_surface), # mS/cm^2 * um^2 = mS    
        ca_infty = 0.1, # uM
        tau_ca = 200, # ms
    )
    frustum_surface = conical_frustum_surface(axon_segments)
    axon_hillock_parameters = AxonhillockParameters(
        na = ChannelParameter(E=50, g=30000 * frustum_surface), # mS/cm^2 * um^2 = mS
        kv = ChannelParameter(E=-90, g=2000 * frustum_surface), # mS/cm^2 * um^2 = mS
        leak = ChannelParameter(E=-70, g=1/50.0* frustum_surface) # mS/cm^2 * um^2 = mS
    )
    neuron_parameters = NeuronParameters(
        Cm = CompartmentParameter(
            soma = 0.75 * conical_frustum_surface(soma_segments), # uF/cm^2 * um^2 = 10^(-4) uF
            dendritic = 0.75 * conical_frustum_surface(dendritic_segments), # uF/cm^2 * um^2 = 10^(-4) uF
            axon_hillock = 0.75 * conical_frustum_surface(axon_segments), # uF/cm^2 * um^2 = 10^(-4) uF
        ),
        Ra = CompartmentParameter(
            soma = 150 / conical_frustum_length(soma_segments), # ohm.cm / um = 10^4 ohm
            dendritic = 150 / conical_frustum_length(dendritic_segments), # ohm.cm / um = 10^4 ohm
            axon_hillock = 150 / conical_frustum_length(axon_segments), # ohm.cm / um = 10^4 ohm
        ),
        soma = soma_parameters,
        dendritic = dendritic_parameters,
        axon_hillock = axon_hillock_parameters
    )

    print(neuron_parameters.Ra.soma.shape)
    print(neuron_parameters.Ra.dendritic.shape)

    # Cm = jp.concatenate([neuron_parameters.Cm.soma, neuron_parameters.Cm.dendritic, neuron_parameters.Cm.axon_hillock])
    Ra = jp.concatenate([neuron_parameters.Ra.soma, neuron_parameters.Ra.dendritic, neuron_parameters.Ra.axon_hillock])                        
    conductance_matrix = compute_conductance_matrix(1/Ra, parent)

    model = NeuronModel(
        parameters=neuron_parameters,
        conductance_matrix=conductance_matrix
    )

    



if __name__ == '__main__':
    test_morphology()
