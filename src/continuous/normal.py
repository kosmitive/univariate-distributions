import numpy as np

from src.continuous.exponential import ExpDist
from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import ContinuousSpace


class NormalDist(ProbDist):
    """Simple gaussian distribution."""

    def __init__(self, mean = 0, var = 1):
        """Create N(mean, var) distribution.

        :param mean Center of the gaussian
        :param var Variance around the center.
        """
        self.mean = mean
        self.var = var

        # create random generators
        self.EG = ExpDist()
        self.UG = UniformDist()
        super().__init__(ContinuousSpace(-np.inf, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.mean

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return self.var

    def sample(self, num_samples = 1):
        """Generate random numbers from N(mean, var) by using an acceptance
        rejection algorithm using Exp(1) and U(0,1).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers from N(mean, var).
        """

        # generate some samples
        elements = np.empty(num_samples)

        for k in range(num_samples):

            # generate sample
            x = self.EG.sample()
            un = self.UG.sample()

            # reject
            while un > np.exp(-(x - 1) ** 2 / 2):
                x = self.sample()
                un = self.UG.sample()

            # accept
            u = self.UG.sample()
            elements[k] = (1 - 2 * int(u <= 0.5)) * x

        return np.sqrt(self.var) * elements + self.mean

    def c_pdf(self, x):
        """This method calculates the density N(x|mean, var).

        :param x What values should be evaluated.
        :returns The probability this element occurs.
        """

        var = self.var
        mean = self.mean
        return 1 / np.sqrt(2 * var * np.pi) * np.exp(-0.5 * (np.subtract(x, mean) ** 2) / var)
