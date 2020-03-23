import math as m
import numpy as np

from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import DiscreteSpace


class HyperGeometricDist(ProbDist):
    """Simple Hyp(n, r, N) distribution."""

    def __init__(self, n = 20, r = 150, N = 300):
        """Create Hyp(n, r, N) distribution.

        :param n offset vars
        :param r binomaial offset
        :param N biggest number.
        """

        # save params
        self.r = r
        self.n = n
        self.N = N

        # generate upper and lower bound
        lb = np.maximum(0, r + n - N)
        ub = np.minimum(n, r)

        # create distribution for sampling
        self.UG = UniformDist()
        super().__init__(DiscreteSpace(lb, ub + 1))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.n * self.r / self.N

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        r = self.r
        N = self.N
        n = self.n

        return n * r / N * (1 - (r / N)) * (N - n) / (N - 1)

    def sample(self, num_samples = 1):
        """Generate random numbers from Hyp(n, r, N).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Hyp(n, r, N).
        """

        pass

    def __density(self, x):
        """This method calculates the mass Hyp(n, r, N).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        r = self.r
        N = self.N
        n = self.n

        t = m.factorial(r) / (m.factorial(x) * m.factorial(r - x))
        t *= m.factorial(N - r) / (m.factorial(n - x) * m.factorial(N - r - n + x))
        return t / (m.factorial(N) / (m.factorial(n) * m.factorial(N - n)))
