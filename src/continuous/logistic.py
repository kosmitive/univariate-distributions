import math as m
import numpy as np

from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import ContinuousSpace


class LogisticDist(ProbDist):
    """Simple Logistic distribution."""

    def __init__(self, loc = 0, scale = 1):
        """Creates Logistic(loc,scale) distribution.

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

        return self.loc

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return (m.pi ** 2 / 3) * (self.scale ** 2)

    def sample(self, num_samples = 1):
        """Generate random numbers from Logistic(loc, scale) by using acceptance rejection distributions.

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Logistic(loc, scale).
        """

        # shortcut
        loc = self.loc
        scale = self.scale

        # sample data
        U = self.UG.sample(num_samples)
        X = np.log(U / (1 - U))
        return scale * X + loc

    def c_pdf(self, x):
        """This method calculates the density Logistic(loc, scale).

        :param x Which value should be evaluated.
        :returns The probability that this element occurs.
        """

        # shortcut
        loc = self.loc
        scale = self.scale

        # update x
        xn = np.subtract(x, loc) / scale
        t = np.exp(-xn)
        b = (1 + t) ** 2
        f = t / b
        return f / scale
