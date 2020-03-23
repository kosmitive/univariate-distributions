import math as m
import numpy as np

from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import DiscreteSpace


class DUniformDist(ProbDist):
    """Sample U(K) distribution."""

    def __init__(self, a = 1, b = 3):
        """Create U(K) distribution.

        :param rate The rate parameter.
        """

        # save params
        assert b >= a
        self.a = a
        self.b = b

        # create distribution for sampling
        self.UG = UniformDist()
        super().__init__(DiscreteSpace(a, b+ 1))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        a = self.a
        b = self.b
        return (a + b) / 2

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        a = self.a
        b = self.b

        return ((b - a) * (b - a + 2)) / 12

    def sample(self, num_samples = 1):
        """Generate random numbers from Poi(rate).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Poi(rate).
        """

        U = self.UG.sample(num_samples)
        X = np.floor()
        return

    def __density(self, x):
        """This method calculates the mass Poi(rate).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        z = np.power(self.rate, x) / m.factorial(x)
        return z * np.exp(-self.rate)
