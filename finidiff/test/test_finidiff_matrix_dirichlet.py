#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from scipy.misc import comb
from scipy.special import gamma

from ..finidiff import finidiff_matrix_dirichlet


class TestFinidiffMatrixDirichlet(unittest.TestCase):

    def test_low_odd_orders_singular(self):

        x = np.linspace(-3, 3, 7)

        for q in 1 + np.arange(3):
            self.assertTrue(
                np.linalg.cond(
                    finidiff_matrix_dirichlet(x, q, -4, 4).toarray()) > 1e9)
