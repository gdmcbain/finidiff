#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''A port to Python 3 of finidiff from Octave.

:author: G. D. McBain <gdmcbain@protonmail.com>

:created: 2017-03-13

'''

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.special import gamma
from scipy.sparse import coo_matrix


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

def finidiff_matrix(x, q):
    '''Return a sparse centred finite difference matrix
    
    for the `q`-th derivative on the abscissæ `x`, omitting the rows
    for the derivatives at peripheral points---i.e., those with
    stencils requiring points outside the `x`.  The stencils reach
    `ceil(q/2)` on either side of the evaluation point, which is the
    second element of the pair returned.

    :param x: sequence of floats, abscissae

    :param q: int > 0, order of derivative

    :rtype: pair, (scipy.sparse.coo_matrix, uint), the second being
    the stencil's reach

    '''
    n = len(x)
    reach = (q + 1) // 2            # the stencil's reach
    nzr = 2 * reach + 1             # entries per row
    nz = (n - 2 * reach) * nzr      # total number of entries
    ri, ci = np.zeros((2, nz), dtype=np.uint)
    d = np.zeros(nz)

    for row in np.arange(reach, n - reach):
        entries = (row - (1 + reach)) * nzr + np.arange(nzr)
        ri[entries] = row
        ci[entries] = np.arange(row - reach, row + reach + 1, dtype=np.uint)
        d[entries] = finidiff(np.asarray(x)[ci[entries]], x[row], q)

    return coo_matrix((d, (ri, ci)), (n, n)), reach

def finidiff_matrix2_mod(D, x, s, p, q):
    '''Return the differentiation matrix with a boundary condition imposed

    :param D: coo_matrix

    :param x: grid, sequence of floats

    :param s: float, coordinate of boundary

    :param p: int or float: for homogeneous Dirichlet or Neumann
    condition at a boundary point, 0 or 1, resp., and q isn't
    needed. For an interfacial condition, p and q are the thermal
    conductivities on the left and right sides of s.

    :param q: float, optional (only needed if min(x) < s < max(x)),
    only needed for interfacial condition in which case it is the
    thermal conductivity on the right.

    '''
    
    
    n = len(x)

    i = sum(x < s)              # x[i-1] < s < x[i]
    if i == 0:                  # s == x[0], left boundary
        new_stencil = [0, 1]
        old_stencil = np.concatenate([np.array([s]), x[new_stencil]])
        b = finidiff(old_stencil, s, p)
        k = finidiff(old_stencil, x[0], 2)
        D[0, new_stencil] = k[[1, 2]] - k[0] * b[[2, 3]] / b[0]

    elif i < n:                 # internal interface
        raise NotImplementedError
    else:                       # s = x[-1], right boundary
        raise NotImplementedError
