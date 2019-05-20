
# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from mpmath import *

from .lseries import _lerch_lseries_trunc


def lerch_integration(z, s, a, verbose):
    """Lerch numerical integration as in mpmath."""
    if re(a) < 1:
        M = int(ceil(mp.one - re(a)))
        if verbose:
            print("lerch: L-series terms = ", M)
        lsum = _lerch_lseries_trunc(z, s, a, M)
        a = a + M
        return z**M * lerch_integration(z, s, a, verbose) + lsum

    g = log(z)
    v = mp.one/(2*a**s) + gammainc(1-s, -a*g) * (-g)**(s-1) / z**a
    h = s / 2
    r = 2*pi
    f = lambda t: sin(s*atan(t/a)-t*g) / ((a**2+t**2)**h * expm1(r*t))
    ff = 2*quad(f, [0, inf])
    if verbose:
        print("v = ", v)
        print("ff = ", ff)
    v += ff
    if not im(z) and not im(s) and not im(a) and re(z) < 1:
        v = re(v)
    return v