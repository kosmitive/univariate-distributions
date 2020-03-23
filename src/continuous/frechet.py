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


class FrechetDist(ProbDist):
    """Simple Fréchet distribution."""

    def __init__(self, shape = 1, loc = 0, scale = 1):
        """Create Fréchet(shape) distribution.

        :param shape Define the shape of density
        """

        # save params
        self.shape = shape
        self.loc = loc
        self.scale = scale
        self.UG = UniformDist()
        super().__init__(ContinuousSpace(0, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        loc = self.loc
        shape = self.shape
        scale = self.scale
        return scale * m.gamma(1 - shape ** -1) + loc

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        scale = self.scale
        return scale ** 2 * (m.gamma(1 - 2 * self.shape ** -1) - m.gamma(1 - self.shape ** -1) ** 2)

    def sample(self, num_samples = 1):
        """Generate random numbers from Fréchet(shape) by using a beta generator.

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Fréchet(shape).
        """

        # generate samples
        loc = self.loc
        shape = self.shape
        scale = self.scale

        # sample data
        U = self.UG.sample(num_samples)
        X = (-np.log(U)) ** (-1/shape)

        # transform
        return scale * X + loc

    def c_pdf(self, x):
        """This method calculates the density Fréchet(shape).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        # only support for positive x
        assert x > 0

        # shortcuts
        loc = self.loc
        shape = self.shape
        scale = self.scale

        # gen pdf
        xn = (np.subtract(x, loc) / scale)
        et = np.exp(-np.power(xn, -shape))
        xt = np.power(xn, -shape - 1)
        return shape / scale * et * xt
