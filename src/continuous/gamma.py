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
from src.probs.distributions.continuous_uv.normal import NormalDist
from src.probs.spaces.spaces1d_leafs import ContinuousSpace


class GammaDist(ProbDist):
    """Simple gamma distribution."""

    def __init__(self, shape = 1, scale = 1):
        """Create Ga(shape,scale) distribution.

        :param shape Shape of Ga(shape,scale)
        :param scale Scale of Ga(shape,scale)
        """
        self.shape = shape
        self.scale = scale

        self.UG = UniformDist()
        if shape >= 1: self.NG = NormalDist()
        super().__init__(ContinuousSpace(0, np.inf, open_brackets=False))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.shape / self.scale

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return self.shape / self.scale ** 2

    def sample(self, num_samples = 1):
        """Generate random numbers from Ga(shape,scale) by using acceptance rejection distributions.

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Ga(shape,scale).
        """

        # extract vars
        shape = self.shape

        # when the shape is bigger than 1
        elements = self.rand_shp_gt_1(num_samples) \
            if shape >= 1 else \
            self.rand_shp_st_1(num_samples)

        return elements / self.scale

    def rand_shp_gt_1(self, num_samples):
        """Creates random variables, if gamma has shape bigger or equal to one.
        Marsaglia and Tsang's method from [1] implemented underneath.

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Ga(shape,scale).

        Refs: [1] https://dl.acm.org/citation.cfm?id=358414.
        """

        # some pre settings
        elements = np.empty(num_samples)
        shape = self.shape
        d = shape - 1 / 3
        c = 1 / m.sqrt(9 * d)

        # create all samples
        for k in range(num_samples):

            # generate normal and unif
            z = self.NG.sample()
            u = self.UG.sample()
            v = (1 + c * z) ** 3

            # first check
            while z <= -(1 / c) or m.log(u) > 0.5 * z ** 2 + d - d * v + d * m.log(v):
                z = self.NG.sample()
                u = self.UG.sample()
                v = (1 + c * z) ** 3

            elements[k] = d * v

        return elements

    def rand_shp_st_1(self, num_samples):
        """Creates random variables, if gamma has shape smaller than one.
        Best's method from [1] implemented underneath.

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Ga(shape,scale).

        Refs: [1] https://link.springer.com/article/10.1007/BF02280789.
        """

        # some pre settings
        elements = np.empty(num_samples)
        shape = self.shape

        # some pre settings
        d = 0.07 + 0.75 * m.sqrt(1 - shape)
        b = 1 + m.exp(-d) * (shape / d)

        # create all samples
        for k in range(num_samples):

            found = False

            # repeat till found
            while not found:

                # two uniform ones
                u1 = self.UG.sample()
                u2 = self.UG.sample()
                v = b * u1

                if v <= 1:

                    # shorthand
                    x = d * v ** (1 / shape)

                    # acceptance check
                    if u2 <= (2 - x) / (2 + x) or u2 <= m.exp(-x):
                        elements[k] = x
                        found = True
                else:

                    # shorthand
                    x = -m.log(d * (b - v) / shape)
                    y = x / d

                    # acceptance check
                    if u2 * (shape + y * (1 - shape)) <= 1 or u2 < y ** (shape - 1):
                        elements[k] = x
                        found = True

        return elements

    def c_pdf(self, x):
        """This method calculates the density Ga(x|shape,scale).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        shape = self.shape
        scale = self.scale
        z = scale ** shape / m.gamma(shape)
        xa = np.power(x, (shape - 1))
        ex = np.exp(np.multiply(-scale, x))
        return z * xa * ex
