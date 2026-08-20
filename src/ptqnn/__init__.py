"""ptqnn — prototype quantum neural networks with realistic noise.

Reproduction of Rosenhahn & Hirche, *Prototype Learning for Quantum Neural Networks* (PTQNN)
and its prerequisite Rosenhahn, Osborne & Hirche, *Stochastic Neural Networks for Quantum
Devices* (SNN, arXiv:2602.22241), on top of a CP-map-first circuit IR.

Sub-packages: ``circuits`` (IR, neurons), ``backends`` (numpy oracle, orkan, trajectories),
``noise`` (Kraus channels, ``NoiseModel``, vendor presets), ``train``, ``tasks``, ``hardware``.
"""

__version__ = "0.1.0"
