import math as m
import numpy as np

from src.continuous.normal import NormalDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import ContinuousSpace


class LogNormalDist(ProbDist):
    """Simple log-normal distribution."""

    def __init__(self, mean=0, var=1):
        """Create LogN(mean, var) distribution.

        :param mean Center of the gaussian
        :param var Variance around the center.
        """
        self.mean = mean
        self.var = var

        # create random generators
        self.NG = NormalDist(mean, var)
        super().__init__(ContinuousSpace(0, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return m.exp(self.mean + self.var / 2)

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return m.exp(2 * self.mean + self.var) * (m.exp(self.var) - 1)

    def sample(self, num_samples=1):
        """Generate random numbers from LogN(mean, var).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers from LogN(mean, var).
        """

        # generate some samples
        Y = self.NG.sample(num_samples)
        return np.exp(Y)

    def c_pdf(self, x):
        """This method calculates the density LogN(x|mean, var).

        :param x What values should be evaluated.
        :returns The probability this element occurs.
        """

        var = self.var
        mean = self.mean
        return 1 / (np.multiply(x, np.sqrt(2 * var * np.pi))) \
               * np.exp(-0.5 * (np.subtract(np.log(x), mean) ** 2) / var)
