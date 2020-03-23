import math as m
import numpy as np

from src.continuous.normal import NormalDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import ContinuousSpace


class CauchyDist(ProbDist):
    """Simple cauchy distribution."""

    def __init__(self, loc = 1, scale = 1):
        """Creates Cauchy(loc, scale) distribution.

        :param loc Location of the distribution.
        :param scale Scale of the distribution
        """

        # save params
        self.loc = loc
        self.scale = scale

        # create generator
        self.NG = NormalDist()
        super().__init__(ContinuousSpace(-np.inf, np.inf))

    def sample(self, num_samples = 1):
        """Generate random numbers from Cauchy(mean,scale) by using ratio of normals.

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Cauchy(mean,scale).
        """

        # create gamma distributed vars
        y1 = self.NG.sample(num_samples)
        y2 = self.NG.sample(num_samples)

        # transform
        return self.scale * (y1 / y2) + self.loc

    def c_pdf(self, x):
        """This method calculates the density Cauchy(mean,scale).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        loc = self.loc
        scale = self.scale
        v = m.pi * self.scale * (1 + np.square((x - loc) / scale))
        return 1 / v
