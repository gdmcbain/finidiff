#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from scipy.misc import comb
from scipy.special import gamma

from ..finidiff import finidiff_matrix


class TestFinidiff(unittest.TestCase):

    def test_low_even_orders_on_a_uniform_grid(self):
        'first to fourth-order on an even grid of seven points'

        n = 7
        for q in 1 + np.arange(3):
            np.testing.assert_array_almost_equal(
                finidiff_matrix(np.linspace(-(n // 2), n // 2, n),
                                2 * q)[0].diagonal(),
                np.concatenate([np.zeros(q),
                                (-1)**q * comb(2 * q, q) * np.ones(n - 2 * q),
                                np.zeros(q)]))

    def test_low_odd_orders_on_a_uniform_grid(self):
        'lowest odd orders on a uniform grid of seven points'

        n = 7
        for q in [1, 3, 5]:
            np.testing.assert_array_almost_equal(
                finidiff_matrix(np.linspace(-(n // 2), n // 2, n),
                                q)[0].diagonal(),
                np.zeros(n))
