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
from src.probs.distributions.continuous_uv.normal import NormalDist
from src.probs.distributions.continuous_uv.uniform import UniformDist
from src.probs.spaces.spaces1d_leafs import ContinuousSpace


class WaldDist(ProbDist):
    """Simple Wald distribution."""

    def __init__(self, loc = 0, scale = 1):
        """Creates Wald(loc,scale) distribution.

        :param loc Location of the distribution.
        :param scale Scale of the distribution
        """

        # save params
        self.loc = loc
        self.scale = scale

        # create generator
        self.NG = NormalDist()
        self.UG = UniformDist()
        super().__init__(ContinuousSpace(0, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.loc

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return self.loc ** 3 / self.scale

    def sample(self, num_samples = 1):
        """Generate random numbers from Wald(loc,scale).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Wald(loc,scale).
        """

        # shortcut
        loc = self.loc
        scale = self.scale

        # some sampling
        W = self.NG.sample(num_samples)
        Y = W ** 2
        Z = loc + (loc ** 2 * Y / 2 * scale) + (loc / 2 * scale) * np.sqrt(4 * loc * scale * Y + loc ** 2 * Y ** 2)
        X = Z

        # sample uniforms
        bound = loc / (loc + Z)
        B = self.UG.sample(num_samples)

        # iterate
        for k in range(num_samples):
            if B[k] > bound: X[k] = loc ** 2 / Z[k]

        return X

    def c_pdf(self, x):
        """This method calculates the density Wald(loc,scale).

        :param x Which value should be evaluated.
        :returns The probability that this element occurs.
        """

        assert x > 0

        # shortcut
        loc = self.loc
        scale = self.scale

        # update x
        z = np.sqrt(scale / (2 * np.pi * np.power(x, 3)))
        b = np.exp(-0.5 * (scale / loc ** 2) * (np.subtract(x, loc) ** 2) / x)
        return z * b
