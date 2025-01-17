import math as m
import numpy as np

from src.discrete.bernoulli import BernDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import DiscreteSpace


class BinDist(ProbDist):
    """Simple binomial distribution."""

    def __init__(self, n = 1, p = 0.5):
        """Create Bin(n, p) distribution.

        :param n Which should be summed
        :param p Probability that random var is true
        """

        # save params
        assert 0 <= p <= 1
        assert 1 <= n
        self.p = p
        self.n = n

        # create distribution for sampling
        self.BG = BernDist(p)
        super().__init__(DiscreteSpace(np.arange(n + 1)))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.n * self.p

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return self.n * self.p * (1 - self.p)

    def sample(self, num_samples = 1):
        """Generate random numbers from Bin(n, p).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Bin(n, p).
        """

        X = np.zeros(num_samples)
        for _ in range(self.n):
            X += self.BG.sample(num_samples)

        return X

    def __density(self, x):
        """This method calculates the mass Bin(n, p).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        assert all([e >= 0 for e in x])
        prev = m.factorial(self.n) / (m.factorial(x) * m.factorial(self.n - x))
        return prev * np.power(self.p, x) * np.power(1 - self.p, np.subtract(self.n, x))
