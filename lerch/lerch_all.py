from mpmath import *
from time import perf_counter, process_time
import multiprocessing


def _lerch_lseries_terms(z, s, a, verbose):
    # estimate number of terms L-series
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
    # truncated L-series 
    if N == 0: return 0

    lsum = a**(-s)
    u = mp.one
    for k in range(1, N):
        u *= z
        lsum += u/(k+a)**s
    return lsum


def _lerch_lseries_trunc_parallel(z, s, a, N):
    # truncated L-series, parallel implementation
    if N == 0: return 0

    threads = multiprocessing.cpu_count()
    block_size = N // threads
    last_block = N % threads
        
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
    # truncated L-series block for parallel implementation
    u = z**n1
    lsum = u / (n1 + a)**s
    for k in range(n1+1, n2):
        u *= z
        lsum += u / (k + a)**s
    queue.put(lsum)


def _lerch_lseries(z, s, a, verbose, parallel):
    # Lerch L-series heuristics
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
    # series acceleration for alternating L-series
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


def _lerch_series(z, s, a, verbose, parallel):
    # Lerch L-series main algorithm
    if re(z) < 0 and im(s) == 0 and re(s) > 1:
        # choose series acceleration of L-series when possible
        if _lerch_lseries_terms(z, s, a, verbose) > 1.2 * int(1.31*mp.dps):
            return _lerch_lseries_alter(z, s, a, verbose)
    
    return _lerch_lseries(z, s, a, verbose, parallel)


def _lerch_em_sum(s, logz, p, q, M):
    # Lerch Euler-Maclaruin Tailsum
    logz2 = logz * logz
    m = 2
    n = logz
    t1 = mp.one
    t2 = mp.one + s * q

    u = t2
    u *= n * bernoulli(2) / m
    tailsum = u
    for k in range(2, M+1):
        r = 2*k - 3
        p1 = p + r
        p2 = -r
        t1 = (t2 * p1 + t1 * p2) * q
        t2 = (t1 * p1 + t2 * p2 + t1 - t2) * q
        u = t2
        m *= (2*k-1)*(2*k)
        n *= logz2
        u *= n * bernoulli(2*k) / m
        tailsum += u
    return tailsum


def _lerch_em_N_M(z, s, a, verbose):
    # Lerch Euler-Maclaurin determine terms M and N. Heuristics and 
    # asymptotic analysis.
    dps = mp.dps
    prec = mp.prec
    try:
        mp.prec = 53
        N = dps // 3
        if (re(a) > abs(re(s)) + abs(re(z)) + dps):
            N = 0
        if verbose:
            print("lerch: E-M L-series, N terms = ", N)
        if prec < 500:
            M = N + prec // 3
            if verbose:
                print("lerch: E-M, M terms (heuristic) = ", M)
        else:
            # asymptotic estimate
            log2pi = 1.8378770664093453
            logz = log(log(z))
            M = ((-prec - 1) * ln2 + logz) / (log2pi - logz)
            M = int(abs(M) / 2) 
            if verbose:
                print("lerch: E-M, M terms (asymptotic) = ", M)
        return N, M
    finally:
        mp.dps = dps
        mp.prec = prec


def _lerch_em_alg(z, s, a, verbose, parallel):
    if verbose:
        print("lerch: precision: ", mp.dps)
    
    N, M = _lerch_em_N_M(z, s, a, verbose)

    if N:
        if parallel and mp.prec >= 1024:
            if verbose:
                print("lerch: L-series parallel mode")
            l_series_sum = _lerch_lseries_trunc_parallel(z, s, a, N)
        else:
            if verbose:
                print("lerch: L-series serial mode")
            l_series_sum = _lerch_lseries_trunc(z, s, a, N)
    else:
        l_series_sum = 0

    a += N
    logz = -log(z)
    r = a * logz
    q = mp.one / r
    p = s + r
    tailsum = _lerch_em_sum(s, logz, p, q, M)
    lterm = mp.one/(2*a**s) + logz**(s-mp.one) / z**a * gammainc(mp.one-s, r)
    return l_series_sum, z**N * (lterm + tailsum * a**(-s))


def lerch_em(z, s, a, verbose, parallel):
    prec = mp.prec 
    # TODO: estimate better possible cancelation to avoid recomputation
    try:
        extraprec = mp.prec // 10
        if re(s) < 0:
            prec_neg_s = 1.53 * log(prec) - 0.55
            extraprec +=  int(abs(re(a) + re(s)) * mp.dps / prec_neg_s)
        if abs(z) > abs(a):
            extraprec += prec # not very good -> switch to integration
        mp.prec += extraprec

        while 1:
            mp.prec = prec + extraprec
            T1, T2 = _lerch_em_alg(z, s, a, verbose, parallel)
            if T1 != 0:
                cancellation = abs(mag(T1) - mag(T1+T2))
            else:
                cancellation = 0
            if verbose:
                print("Term 1:", T1)
                print("Term 2:", T2)
                print("M1 = ", mag(T1))
                print("M2 = ", mag(T1+T2))
                print("Cancellation: ", cancellation, "bits")
            if cancellation < extraprec:
                return T1 + T2
            else:
                extraprec = max(2*extraprec, min(cancellation + 5, 100*prec))
                if verbose:
                    print("lerch: extraprec: ", extraprec)
    finally:
        mp.prec = prec


def _lerch_integration(z, s, a, verbose):
    if re(a) < 1:
        M = int(ceil(mp.one - re(a)))
        if verbose:
            print("lerch: L-series terms = ", M)
        lsum = _lerch_lseries_trunc(z, s, a, M)
        a = a + M
        return z**M * _lerch_integration(z, s, a, verbose) + lsum

    g = log(z)
    v = mp.one/(2*a**s) + gammainc(1-s, -a*g) * (-g)**(s-1) / z**a
    h = s / 2
    r = 2*pi
    f = lambda t: sin(s*atan(t/a)-t*g) / ((a**2+t**2)**h * expm1(r*t))
    ff = 2*quad(f, [0, inf])
    print("v = ", v)
    print("ff = ", ff)
    v += ff
    if not im(z) and not im(s) and not im(a) and re(z) < 1:
        v = re(v)
    return v


def _generate_peak_numbers(n):
    # Generate peak numbers to compute peak polynomials
    V = [[1], [2]]
    for k in range(2, n):
        v = []
        w = int(ceil(k/2))
        q = int(floor(k/2))
        for j in range(q+1):
            t1 = 2*(j+1) * V[k-1][j] if j < w else 0
            t2 = (k+1-2*j) * V[k-1][j-1] if j > 0 else 0
            v.append(t1 + t2)
        V.append(v)
    return V


def _lerch_asymp_series(z, s, a, N):
    # Lerch asymptotic expansion for positive a, s and z. Large a, z and fixed
    # s.
    logz = log(z)
    logz2 = logz / 2
    ilogz = 1j*logz
    _2pi = mp.one /(2*pi)
    q = _2pi
    y = mp.one + ilogz *_2pi
    ia = mp.one/a * q
    n = mp.one * q
    
    V = _generate_peak_numbers(N)
    u = logz2 / pi
    csch2 = csch(logz2)**2
    sech2 = sech(logz2)**2
    cothk = coth(logz2)
    iu = mp.one / u
    t = iu
    d = csch2 * pi / cothk
    
    c = (2/logz - coth(logz2)) / 2
    
    _sum = mp.zero
    for k in range(1, N):
        n *= ia * (s + k - 1)
        t *= iu 
        d *= pi * cothk / k
        pN = int(ceil(k/2))
        ps = mp.zero
        sc = mp.one
        for i in range(pN):
            ps += V[k-1][i] * sc
            sc *= sech2
        hh = t - d * ps
        v = n * hh
        _sum += v
    return (c + _sum) / a**(s)


def _lerch_asymp_z_terms(z, s, a, verbose):
    # Heuristic to estimate number of terms. TODO: improve this.
    prec = mp.prec
    try:
        mp.prec = 53
        t = -log(2)*prec -log(1+s)
        k = t / lambertw(t / (a*2*pi*log(abs(z))),-1)
        N = int(abs(k)*1.2)
        if verbose:
            print("lerch: asympotic series terms N =", N)
        return N
    finally:
        mp.prec = prec


def _lerch_asymp_z(z, s, a, verbose):
    prec = mp.prec
    try:
        mp.prec += 10
        N = _lerch_asymp_z_terms(z, s, a, verbose)
        g = log(z)
        v = mp.one/(mpf("2")*a**s) 
        v+= gammainc(mp.one-s, -a*g) * (-g)**(s-mp.one) / z**a
        s = _lerch_asymp_series(z, s, a, N)
        v += s
        return v
    finally:
        mp.prec = prec


def lerch(z, s, a, **kwargs):
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
        return _lerch_series(z, s, a, verbose, parallel)
    if log(abs(z)) < 2*pi:
        if verbose:
            print("lerch: Euler-Maclaurin algorithm")
        return lerch_em(z, s, a, verbose, parallel)
    elif abs(re(z)) > 1:
        if verbose:
            print("lerch: Asymptotic expansion")
        return _lerch_asymp_z(z, s, a, verbose)
    else:
        if verbose:
            print("lerch: Numerical integration")
        return _lerch_integration(z, s, a, verbose)


if __name__ == "__main__":
    mp.prec = 10000
    z = mpc("10000", "0")
    s = mpc("10/4","0")
    a = mpc("2000","0")

    t_start = perf_counter()
    #result = lerch(z, s, a, verbose=True, parallel=True)
    result = lerchphi(z, s, a)
    t_end1 = perf_counter() - t_start
    print(result)

    t_start = perf_counter()
    result = lerch(z, s, a, verbose=True, parallel=True)
    #result = lerchphi(z, s, a)
    t_end2 = perf_counter() - t_start
    print("process time = ", t_end1)
    print("process time = ", t_end2)
    print(result)