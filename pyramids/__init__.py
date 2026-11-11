"""Collapsing pyramids: a dynamical model of what AI does to the base of professions."""

from .model import SERIES_DOC, Trajectory, simulate
from .parameters import PROFESSIONS, REFERENCE, WEIGHTS, Params
from .scenarios import FULL_POLICY, KNOWLEDGE, PLANET, SOCIETIES, classify, society

__all__ = [
    "simulate",
    "Trajectory",
    "SERIES_DOC",
    "Params",
    "REFERENCE",
    "PROFESSIONS",
    "WEIGHTS",
    "SOCIETIES",
    "KNOWLEDGE",
    "PLANET",
    "FULL_POLICY",
    "society",
    "classify",
]
