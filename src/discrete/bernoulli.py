import numpy as np

from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import DiscreteSpace


class BernDist(ProbDist):
    """Simple bernoulli distribution."""

    def __init__(self, p = 0.5):
        """Create Ber(p) distribution.

        :param p Probability that random var is true
        """

        # save params
        assert 0 <= p <= 1
        self.p = p

        # create distribution for sampling
        self.UG = UniformDist()
        super().__init__(DiscreteSpace([0, 1]))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.p

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return self.p * (1 - self.p)

    def sample(self, num_samples = 1):
        """Generate random numbers from Ber(p).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Ber(p).
        """

        # create gamma distributed vars
        U = self.UG.sample(num_samples)
        for k in range(num_samples):
            U[k] = int(U[k] <= self.p)

        return U

    def __density(self, x):
        """This method calculates the mass Ber(x|p).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        assert np.all([np.logical_or(np.equal(x, 0), np.equal(x, 1))])
        return np.power(self.p, x) * np.power(1 - self.p, np.subtract(1, x))
