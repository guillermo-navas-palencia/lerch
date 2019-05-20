
# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import multiprocessing

from mpmath import *
from time import perf_counter


def _lerch_lseries_terms(z, s, a, verbose):
    """Estimate number of terms for L-series."""
    prec = mp.prec
    try:
        mp.prec = 53
        one = mp.one 
        two = mpf("2.0")

        z = abs(z)
        logz = log(z)
        s = re(s)
        a = abs(a)
        if s > 0:
            s += 1
            q = -(two**(-prec) * a**(-s) * z**a)**(-one/s) * logz / s
            N = int(abs((s * lambertw(q) + a*logz) / logz))
            if verbose:
                print("lerch: LambertW(0): {0} > {1}".format(q, -1.0/e))
        else:
            q = -(two**(-prec) *  a**(-s) * z**a)**(-one/s) * logz / s
            N = int(abs((s * lambertw(q, -1) + a*logz) / logz))
            if verbose:
                print("lerch: LambertW(-1): {0} < {1} < 0".format(-1.0/e, q))
        return N
    finally:
        mp.prec = prec


def _lerch_lseries_trunc(z, s, a, N):
    """Truncated L-series."""
    if N == 0: return 0

    lsum = a**(-s)
    u = mp.one
    for k in range(1, N):
        u *= z
        lsum += u/(k+a)**s
    return lsum


def _lerch_lseries_trunc_parallel(z, s, a, N):
    """Truncated L-series, parallel implementation."""
    if N == 0: return 0

    threads = multiprocessing.cpu_count()
    block_size, last_block = divmod(N, threads)
        
    queue = multiprocessing.Queue()    
    processes = [multiprocessing.Process(target=_lerch_lseries_trunc_block, 
                args=(z, s, a, block_size*thread, block_size*(thread+1), 
                    queue)) for thread in range(threads)]
        
    for p in processes: p.start()
    for p in processes: p.join()
    
    series_sum = sum([queue.get() for p in processes]) 
    
    if last_block:
        _lerch_lseries_trunc_block(z, s, a, block_size*(threads), 
            block_size*(threads) + last_block, queue)
        series_sum += queue.get()
        
    return series_sum

    
def _lerch_lseries_trunc_block(z, s, a, n1, n2, queue):
    """Truncated L-series block for parallel implementation."""
    u = z**n1
    lsum = u / (n1 + a)**s
    for k in range(n1+1, n2):
        u *= z
        lsum += u / (k + a)**s
    queue.put(lsum)


def _lerch_lseries(z, s, a, verbose, parallel):
    """Lerch L-series heuristics."""
    prec = mp.prec
    try:
        N = _lerch_lseries_terms(z, s, a, verbose)
        if verbose:
            print("lerch: number of terms L-series = ", N)
        if re(z) < 0:
            extraprec = 20
        else:
            extraprec = 10
        if re(s) < 0:
            extraprec += mp.prec // 3 + int(-re(s))
        if verbose:
            print("lerch: extraprec: ", extraprec)
        mp.prec += extraprec

        if parallel and (mp.prec >= 1024 or N > 1024):
            if verbose:
                print("lerch: L-series parallel mode")
            return _lerch_lseries_trunc_parallel(z, s, a, N)
        else:
            if verbose:
                print("lerch: L-series serial mode")
            return _lerch_lseries_trunc(z, s, a, N)
    finally:
        mp.prec = prec


def _lerch_lseries_alter(z, s, a, verbose):
    """Series acceleration for alternating L-series."""
    prec = mp.prec
    try:
        N = int(1.31*mp.dps)
        N += 5
        extraprec = 10
        mp.prec += extraprec

        if verbose:
            print("lerch: number of terms L-series alt. = ", N)

        d = (mpf("3") + sqrt(mpf("8")))**N
        d = (d + 1 / d) / 2
        b = -mp.one
        c = -d
        lsum = 0
        u = mp.one
        for k in range(N):
            t = u / (k + a)**s
            c = b - c
            if k % 2 == 0:
                lsum = lsum + c * t
            else:
                lsum = lsum - c * t
            b *= 2 * (k + N) * (k - N) / ((2 * k + mp.one) * (k + mp.one))
            u *= z
        return lsum / d
    finally:
        mp.prec = prec


def lerch_series(z, s, a, verbose, parallel):
    """
    Lerch L-series main algorithm. Choose between alternating series or
    L-series depending on sign(z) and number of terms.
    """
    if re(z) < 0 and im(s) == 0 and re(s) > 1:
        # choose series acceleration of L-series when possible
        if _lerch_lseries_terms(z, s, a, verbose) > 1.2 * int(1.31*mp.dps):
            return _lerch_lseries_alter(z, s, a, verbose)
    
    return _lerch_lseries(z, s, a, verbose, parallel)