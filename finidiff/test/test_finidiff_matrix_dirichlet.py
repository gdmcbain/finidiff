#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from scipy.misc import comb
from scipy.sparse.linalg import spsolve
from scipy.special import gamma

from ..finidiff import finidiff_matrix_dirichlet


class TestFinidiffMatrixDirichlet(unittest.TestCase):

    def test_low_odd_orders_singular(self):

        x = np.linspace(-3, 3, 7)

        for q in 1 + np.arange(3):
            self.assertTrue(
                np.linalg.cond(
                    finidiff_matrix_dirichlet(x, q, -4, 4).toarray()) > 1e9)

    def test_poisson(self, n=2**6):
        '''Solve u" + 2 = 0 subject to u(0) = u(1) = 0

        which has exact solution x * (1 - x)

        '''

        x = np.sort(np.random.rand(n))
        np.testing.assert_array_almost_equal(
            spsolve(finidiff_matrix_dirichlet(x, 2, 0, 1),
                    -2 * np.ones_like(x)),
            x * (1 - x))
