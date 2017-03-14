#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''A port to Python 3 of finidiff from Octave.

:author: G. D. McBain <gdmcbain@protonmail.com>

:created: 2017-03-13

'''

import numpy as np
from scipy.special import gamma


def postpad(x, l):
    '''return array of length l, being x followed by zeros

    For compatibility with GNU Octave.

    '''

    return np.concatenate([x, np.zeros(max(0, l - len(x)))])
    

def finidiff(x, s, q, b=0, z=None):
    '''return the stencil

    :param x: sequence, float, abscissae

    :param s: float, the point at which the derivative of order q is
    desired to be approximated

    :param q: int, order of derivative; an interpolation stencil is
    returned for q = 0

    :param b: int, the formula should be exact for the polynomials (x
    - z)^p for p up to b + len(x)

    :param z: irrelevant if b is zero

    '''

    z = s if z is None else z

    n = len(x)
    p = n + b
    i = np.arange(1, p - q + 1)

    return np.linalg.solve(np.vander(postpad(np.asarray(x) - z, p)).T[:n, :n],
                           postpad(gamma(p - i + 1) / gamma(p - i - q + 1) *
                                   (s - z) ** (p - i - q), n))
