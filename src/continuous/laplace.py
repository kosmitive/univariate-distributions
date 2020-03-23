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
from src.probs.distributions.continuous_uv.exponential import ExpDist
from src.probs.distributions.continuous_uv.normal import NormalDist
from src.probs.spaces.spaces1d_leafs import ContinuousSpace


class LaplaceDist(ProbDist):
    """Simple Laplace distribution."""

    def __init__(self, loc = 0, scale = 1):
        """Creates Laplace(loc,scale) distribution.

        :param loc Location of the distribution.
        :param scale Scale of the distribution
        """

        # save params
        self.loc = loc
        self.scale = scale

        # create generator
        self.NG = NormalDist()
        self.EG = ExpDist()
        super().__init__(ContinuousSpace(-np.inf, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.loc

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return 2 * (self.scale ** 2)

    def sample(self, num_samples = 1):
        """Generate random numbers from Laplace(loc, scale) by using acceptance rejection distributions.

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Laplace(loc, scale).
        """

        # shortcut
        loc = self.loc
        scale = self.scale

        # sample data
        E = self.EG.sample(num_samples)
        Y = self.NG.sample(num_samples)
        X = Y * np.sqrt(2 * E)
        return scale * X + loc

    def c_pdf(self, x):
        """This method calculates the density Laplace(loc, scale).

        :param x Which value should be evaluated.
        :returns The probability that this element occurs.
        """

        # shortcut
        loc = self.loc
        scale = self.scale

        # update x
        xn = np.subtract(x, loc) / scale
        f = 0.5 * np.exp(-np.abs(xn))
        return f / scale
