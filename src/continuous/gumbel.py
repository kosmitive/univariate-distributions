import math as m
import numpy as np

from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import ContinuousSpace


class GumbelDist(ProbDist):
    """Simple Gumbel distribution."""

    def __init__(self, loc = 1, scale = 1):
        """Creates Gumbel(loc,scale) distribution.

        :param loc Location of the distribution.
        :param scale Scale of the distribution
        """

        # save params
        self.loc = loc
        self.scale = scale

        # create generator
        self.UG = UniformDist()
        super().__init__(ContinuousSpace(-np.inf, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.scale * 0.577216 + self.loc

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return (self.scale ** 2) * (m.pi ** 2 / 6)

    def sample(self, num_samples = 1):
        """Generate random numbers from Gumbel(loc, scale) by using acceptance rejection distributions.

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Gumbel(loc, scale).
        """

        # shortcut
        loc = self.loc
        scale = self.scale

        # sample data
        U = self.UG.sample(num_samples)
        X = -np.log(-np.log(U))
        return scale * X + loc

    def c_pdf(self, x):
        """This method calculates the density Dist(x).

        :param x Which value should be evaluated.
        :returns The probability that this element occurs.
        """

        # shortcut
        loc = self.loc
        scale = self.scale

        # update x
        xn = np.subtract(x, loc) / scale
        f = np.exp(-xn - np.exp(-xn)) / scale
        return f
