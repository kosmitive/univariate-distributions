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
from src.probs.spaces.spaces1d_leafs import ContinuousSpace


class WeibullDist(ProbDist):
    """Simple Weibull distribution."""

    def __init__(self, shape = 1, loc = 1, scale = 1):
        """Creates Weib(shape,loc,scale) distribution.

        :param shape Shape of the distribution.
        :param loc Location of the distribution.
        :param scale Scale of the distribution
        """

        # save params
        self.shape = shape
        self.loc = loc
        self.scale = scale

        # create generator
        self.UG = UniformDist()
        super().__init__(ContinuousSpace(0, np.inf, open_brackets=False))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return (self.scale ** -1) * m.gamma(1 + self.shape ** -1) + self.loc

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return self.scale ** -2 \
            * (m.gamma(1 + 2 * self.shape ** -1) - m.gamma(1 + self.shape ** -1) ** 2)

    def sample(self, num_samples = 1):
        """Generate random numbers from Weib(shape,loc,scale).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Weib(shape,loc,scale).
        """

        # shortcut
        shape = self.shape
        loc = self.loc
        scale = self.scale

        # some sampling
        U = self.UG.sample(num_samples)
        X = 1 / scale * (-np.log(U)) ** (1 / shape)
        return scale * X + loc

    def c_pdf(self, x):
        """This method calculates the density Weib(shape,loc,scale).

        :param x Which value should be evaluated.
        :returns The probability that this element occurs.
        """

        assert x > 0

        # shortcut
        shape = self.shape
        loc = self.loc
        scale = self.scale
        xn = np.subtract(x, loc) / scale

        # update x
        ft = shape * xn ** (shape - 1) * np.exp(-xn ** shape)
        return ft / scale
