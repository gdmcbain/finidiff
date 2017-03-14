#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from scipy.special import gamma

from ..finidiff import finidiff, postpad

class TestFinidiff(unittest.TestCase):

    def test_second_derivative(self):
        'central three-point 2nd derivative on evenly spaced grid'
        np.testing.assert_array_almost_equal(
            finidiff(np.linspace(-1., 1., 3), 0, 2),
            [1, -2, 1])

    def test_fourth_derivative(self):
        'central five-point 4th derivative on evenly spaced grid'
        np.testing.assert_array_almost_equal(
            finidiff(np.linspace(-2., 2., 5), 0, 4),
            [1, -4, 6, -4, 1])

    def test_differentiating_monomials(self, n=9):
        'differentiate x**(0:8) 8 times on 9 random points'

        
        x = np.random.rand(n + 1)
        d = finidiff(x, 0.5, n)
        for i in range(n + 1):
            np.testing.assert_almost_equal(d.dot(x ** i),
                                           0 if i < n else gamma(n + 1),
                                           2)


