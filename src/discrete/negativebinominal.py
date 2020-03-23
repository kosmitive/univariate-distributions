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
from src.probs.distributions.continuous_uv.gamma import GammaDist
from src.probs.distributions.discrete_uv.poisson import PoissonDist
from src.probs.spaces.spaces1d_leafs import DiscreteSpace


class NegBinDist(ProbDist):
    """Simple ngeative binomial distribution."""

    def __init__(self, r = 10, p = 0.5):
        """Create NegBin(r, p) distribution.

        :param r Which should be summed
        :param p Probability that random var is true
        """

        # save params
        assert 0 <= p <= 1
        assert 0 <= r
        self.r = r
        self.p = p

        # create distribution for sampling
        self.GG = GammaDist(r, p / (1 - p))
        self.PG = PoissonDist(0.1)
        super().__init__(DiscreteSpace(0, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        r = self.r
        p = self.p
        return r * (1 - p) / p

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        r = self.r
        p = self.p
        return r * (1 - p) / p ** 2

    def sample(self, num_samples = 1):
        """Generate random numbers from NegBin(r, p).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ NegBin(r, p).
        """

        # a gamma generator and the result vector
        L = self.GG.sample(num_samples)
        X = np.empty(num_samples)

        # fill it up
        for k in range(len(X)):
            self.PG.rate = L[k]
            X[k] = self.PG.sample()

    def __density(self, x):
        """This method calculates the mass NegBin(r, p).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        r = self.r
        p = self.p
        prev = m.gamma(r + x) / (m.gamma(r) * m.factorial(x))
        return prev * p ** r * (1 - p) ** x
