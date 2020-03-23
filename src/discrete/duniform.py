# ==================================================================== #
# Relationalyze                                                        #
# ==================================================================== #
# Copyright (C) 2018  Markus Semmler                                   #
#                                                                      #
# This program is free software: you can redistribute it and/or modify #
# it under the terms of the GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or    #
# (at your option) any later version.                                  #
#                                                                      #
# This program is distributed in the hope that it will be useful,      #
# but WITHOUT ANY WARRANTY; without even the implied warranty of       #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        #
# GNU General Public License for more details.                         #
#                                                                      #
# You should have received a copy of the GNU General Public License    #
# along with this program. If not, see <http://www.gnu.org/licenses/>. #
# ==================================================================== #

import math as m
import numpy as np

from src.probs.distributions.marginals.prob_distribution import ProbDist
from src.probs.distributions.continuous_uv.uniform import UniformDist
from src.probs.spaces.spaces1d_leafs import DiscreteSpace


class DUniformDist(ProbDist):
    """Sample U(K) distribution."""

    def __init__(self, a = 1, b = 3):
        """Create U(K) distribution.

        :param rate The rate parameter.
        """

        # save params
        assert b >= a
        self.a = a
        self.b = b

        # create distribution for sampling
        self.UG = UniformDist()
        super().__init__(DiscreteSpace(a, b+ 1))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        a = self.a
        b = self.b
        return (a + b) / 2

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        a = self.a
        b = self.b

        return ((b - a) * (b - a + 2)) / 12

    def sample(self, num_samples = 1):
        """Generate random numbers from Poi(rate).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Poi(rate).
        """

        U = self.UG.sample(num_samples)
        X = np.floor()
        return

    def __density(self, x):
        """This method calculates the mass Poi(rate).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        z = np.power(self.rate, x) / m.factorial(x)
        return z * np.exp(-self.rate)
