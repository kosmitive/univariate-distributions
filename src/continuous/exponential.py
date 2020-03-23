import math as m
import numpy as np

from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import ContinuousSpace


class ExpDist(ProbDist):
    """Simple exponential distribution."""

    def __init__(self, rate = 1):
        """Create Exp(rate) distribution.

        :param rate First shape parameter of exp
        """

        assert rate > 0

        self.rate = rate
        self.UG = UniformDist()
        super().__init__(ContinuousSpace(0, np.inf, open_brackets=False))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return 1 / self.rate

    def moment(self, k):
        """Calculates the k-th moment for that distribution.

        :param k which moment to calc
        :returns The k-th moment of the distribution"""

        return m.factorial(k) / self.rate ** k

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return 1 / self.rate ** 2

    def c_pdf(self, x):
        """This method calculates the density Exp(x|rate).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        rate = self.rate
        return rate * np.exp(-rate * x)

    def sample(self, num_samples = 1):
        """Generate random numbers from Exp(rate) by using inverse-transform method.

        :param num_samples How many random numbers should be generated.
        :returns A random number from Exp(rate).
        """

        # generate result list and uniform samples
        rate = self.rate
        U = self.UG.sample(num_samples)
        return (-1 / rate) * np.log(U)
