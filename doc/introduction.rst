Introduction
============

Rationale
~~~~~~~~~

Computational Neuroscience has relied on computer simulation to investigate diverse hypothesis on neural computation, with many established packages providing convenient, well tested and easy to use simulation tools. So why even attempt to build a new simulation library?

We believe a confluence of several factors warrants such an attempt: 

- Existing simulation tools do not offer convenient methods of gradient-based optimisation.
- JAX is now enabling high-performance and functional implementations of *differentiable* simulation algorithms in Python, which can then run on a wide variety of hardware accellerators. Existing attempts at GPU accelerated neuron simulators relied on hard to write and maintain custom code.  
- Using our approach we can write the differential equations governing neuron and synapse dynamics in a high-level and composable fashion, without compromising on speed of execution. The :doc:`notebooks/hodgekin_huxley` notebook demonstrates this approach.
- The emerging field of NeuroAI and increasing interest in machine learning with bio-plausible components, suggests to align methods with those in common use in the machine learning community.

As it stands the existing components are expressive enough to simulate models of morphologically detailed neuron models, largely due to the supported integration methods and domain specific solvers. We are not in any way attempting to match the 
scope and breadth of existing simulation tools like NEURON or Brian 2. 
