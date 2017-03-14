#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from scipy.special import gamma

from ..finidiff import finidiff_matrix


class TestFinidiff(unittest.TestCase):

    def test_low_orders_on_an_even_grid(self):
        'first to fourth-order on an even grid of seven points'

        for q in 1 + np.arange(6):
            D, _ = finidiff_matrix(np.linspace(-3., 3, 7), q)
            print(q, D.T)
