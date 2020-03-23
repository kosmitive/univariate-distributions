import math as m
import numpy as np

from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import ContinuousSpace


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
