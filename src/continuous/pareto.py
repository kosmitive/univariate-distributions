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

import numpy as np

from src.probs.distributions.marginals.prob_distribution import ProbDist
from src.probs.distributions.continuous_uv.uniform import UniformDist
from src.probs.spaces.spaces1d_leafs import ContinuousSpace


class ParetoDist(ProbDist):
    """Simple Pareto distribution."""

    def __init__(self, shape = 1, scale = 1):
        """Creates Pareto(shape,scale) distribution.

        :param shape Shape of the distribution.
        :param scale Scale of the distribution
        """

        # save params
        self.shape = shape
        self.scale = scale

        # create generator
        self.UG = UniformDist()
        super().__init__(ContinuousSpace(0, np.inf, open_brackets=False))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return 1 / (self.scale * (self.shape - 1))

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        shape = self.shape
        scale = self.scale
        return shape / (scale ** 2 * (shape - 1) ** 2 * (shape - 2))

    def sample(self, num_samples = 1):
        """Generate random numbers from Pareto(shape,scale).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Pareto(shape,scale).
        """

        # shortcut
        shape = self.shape
        scale = self.scale

        # sample data
        U = self.UG.sample(num_samples)
        X = U ** (-1/shape) - 1
        return scale * X

    def c_pdf(self, x):
        """This method calculates the density Pareto(shape,scale).

        :param x Which value should be evaluated.
        :returns The probability that this element occurs.
        """

        # shortcut
        shape = self.shape
        scale = self.scale

        # update x
        return shape * scale * (1 + np.multiply(scale, x)) ** (-(shape + 1))
