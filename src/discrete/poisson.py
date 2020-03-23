import math as m
import numpy as np

from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import DiscreteSpace


class PoissonDist(ProbDist):
    """Simple Poi(rate) distribution."""

    def __init__(self, rate=1):
        """Create Poi(rate) distribution.

        :param rate The rate parameter.
        """

        # save params
        assert 0 < rate
        self.rate = rate

        # create distribution for sampling
        self.UG = UniformDist()
        super().__init__(DiscreteSpace(0, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.rate

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return self.rate

    def sample(self, num_samples = 1):
        """Generate random numbers from Poi(rate).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Poi(rate).
        """

        X = np.empty(num_samples)
        for k in range(len(X)):

            # starting
            n = 1
            a = 1
            Un = self.UG.sample()
            a = a * Un

            # iterate over
            while a >= np.exp(-self.rate):
                n = n + 1
                Un = self.UG.sample()
                a = a * Un

            X[k] = n - 1

        return X

    def __density(self, x):
        """This method calculates the mass Poi(rate).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        z = np.power(self.rate, x) / m.factorial(x)
        return z * np.exp(-self.rate)
