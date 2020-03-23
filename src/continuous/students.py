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


class StudentsTDist(ProbDist):
    """Simple student-t distribution."""

    def __init__(self, v = 1, loc = 0, scale = 1):
        """Create t(v,loc,scale) distribution.

        :param v Degrees of Freedom
        """

        # save params
        self.v = v
        self.loc = loc
        self.scale = scale

        # create generator
        self.UG = UniformDist()
        super().__init__(ContinuousSpace(-np.inf, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.loc

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return self.scale ** 2 * (self.v / (self.v - 2))

    def sample(self, num_samples=1):
        """Generate random numbers from t(v,loc,scale).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers from t(v,loc,scale).
        """

        # shortcut
        v = self.v
        loc = self.loc
        scale = self.scale
        elements = np.empty(num_samples)

        # create samples
        for k in range(num_samples):

            # iterate till found
            found = False
            while not found:

                # generate some samples
                u1 = self.UG.sample()
                u2 = self.UG.sample()

                # set X and V
                if u1 < 0.5:
                    X = 1 / (4 * u1 - 1)
                    V = u2 / X ** 2
                else:
                    X = 4 * u1 - 3
                    V = u2

                # acceptance check
                if V < 1 - abs(X) / 2 or V < (1 + (X ** 2) / v) ** (-(v+1) / 2):
                    elements[k] = X
                    found = True

        return scale * elements + loc

    def c_pdf(self, x):
        """This method calculates the density t(v,loc,scale).

        :param x What values should be evaluated.
        :returns The probability this element occurs.
        """

        # shortcut
        v = self.v
        loc = self.loc
        scale = self.scale

        # update x
        xn = np.subtract(x, loc) / scale
        z = m.gamma((v + 1) / 2) / (np.sqrt(v * m.pi) * m.gamma(v / 2))
        X = (1 + np.square(xn) / v) ** (-(v+1) / 2)
        return (z * X) / scale
