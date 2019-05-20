
# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from mpmath import *

from .lseries import lerch_series
from .integration import lerch_integration


def lerchphi(z, s, a, **kwargs):
    """Lerch transcendent implementation."""
    verbose = kwargs.get('verbose')
    parallel = kwargs.get('parallel')

    # special cases
    if z == -1 and a == 1:
        return altzeta(s)
    if z == -1:
        return (2*mp.one)**(-s) * (zeta(s, a / 2) - zeta(s, (a + mp.one) / 2))
    if z == 0:
        return a**(-s) 
    if z == 1:
        return zeta(s,a)
    if s == 0:
        return mp.one / (mp.one - z)
    if s == 1 and a == 1:
        return -log(mp.one - z) / z

    # general cases
    if abs(z) < 1.0:
        if verbose:
            print("lerch: L-series")
        return lerch_series(z, s, a, verbose, parallel)
    else:
        if verbose:
            print("lerch: Numerical integration")
        return lerch_integration(z, s, a, verbose)