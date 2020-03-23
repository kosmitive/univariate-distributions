import numpy as np

from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import DiscreteSpace


class GeometricDist(ProbDist):
    """Simple Geom(p) distribution."""

    def __init__(self, n = 1, p = 0.5):
        """Create Geom(p) distribution.

        :param n Which should be summed
        :param p Probability that random var is true
        """

        # save params
        assert 0 <= p <= 1
        assert 1 <= n
        self.p = p
        self.n = n

        # create distribution for sampling
        self.UG = UniformDist()
        super().__init__(DiscreteSpace(1, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return 1 / self.p

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return (1 - self.p) / self.p ** 2

    def sample(self, num_samples = 1):
        """Generate random numbers from Geom(p).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Geom(p).
        """

        U = self.UG.sample(num_samples)
        X = np.floor(np.log(U) / np.log(1 - self.p))
        return X

    def __density(self, x):
        """This method calculates the mass Geom(p).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        f = np.power(1 - self.p, x - 1) * self.p
        return f
