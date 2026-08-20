"""Iris classification ``[PTQNN §III.C, Fig 7–9]`` on the ``[SNN §IV.B]`` binary encoding.

Submodules are imported explicitly (``from ptqnn.tasks.iris import data, model, train``) so that
``model`` and ``train`` can import ``data`` without a circular initialisation.
"""
