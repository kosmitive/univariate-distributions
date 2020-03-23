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
from src.probs.distributions.continuous_uv.normal import NormalDist
from src.probs.spaces.spaces1d_leafs import ContinuousSpace


class LogNormalDist(ProbDist):
    """Simple log-normal distribution."""

    def __init__(self, mean=0, var=1):
        """Create LogN(mean, var) distribution.

        :param mean Center of the gaussian
        :param var Variance around the center.
        """
        self.mean = mean
        self.var = var

        # create random generators
        self.NG = NormalDist(mean, var)
        super().__init__(ContinuousSpace(0, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return m.exp(self.mean + self.var / 2)

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return m.exp(2 * self.mean + self.var) * (m.exp(self.var) - 1)

    def sample(self, num_samples=1):
        """Generate random numbers from LogN(mean, var).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers from LogN(mean, var).
        """

        # generate some samples
        Y = self.NG.sample(num_samples)
        return np.exp(Y)

    def c_pdf(self, x):
        """This method calculates the density LogN(x|mean, var).

        :param x What values should be evaluated.
        :returns The probability this element occurs.
        """

        var = self.var
        mean = self.mean
        return 1 / (np.multiply(x, np.sqrt(2 * var * np.pi))) \
               * np.exp(-0.5 * (np.subtract(np.log(x), mean) ** 2) / var)
