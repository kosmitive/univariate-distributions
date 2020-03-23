import math as m
import numpy as np


class BetaDist(ProbDist):
    """Simple beta distribution."""

    def __init__(self, a = 1, b = 1):
        """Create Beta(a,b) distribution.

        :param a First shape parameter of beta
        :param b Second shape parameter of beta
        """

        assert a > 0 and b > 0

        # save params
        self.a = a
        self.b = b

        # create two distributions when one wants to sample
        self.GaG = GammaDist(self.a, 1)
        self.GbG = GammaDist(self.b, 1)

        # define the space of the distribution
        super().__init__(ContinuousSpace(0, 1))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.a / (self.a + self.b)

    def moment(self, k):
        """Calculates the k-th moment for that distribution.

        :param k which moment to calc
        :returns The k-th moment of the distribution"""

        a = self.a
        b = self.b
        t = m.gamma(a + b) * m.gamma(a + k) / (m.gamma(a + k + b) * m.gamma(a))
        return t

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        a = self.a
        b = self.b
        return (a*b) / ((a + b) ** 2 * (a + b + 1))

    def c_pdf(self, x):
        """This method calculates the density Beta(x|a,b).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        assert np.all([np.logical_and(np.greater_equal(x, 0), np.greater_equal(1, x))])

        a = self.a
        b = self.b
        bab = m.gamma(a + b) / (m.gamma(a) * m.gamma(b))
        xa = np.power(x, (a - 1))
        xb = np.power(1 - x, (b - 1))
        return bab * xa * xb

    def sample(self, num_samples = 1):
        """Generate random numbers from Beta(a,b) by using acceptance rejection distributions.

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Beta(a,b).
        """

        # create gamma distributed vars
        y1 = self.GaG.sample(num_samples)
        y2 = self.GaG.sample(num_samples)

        # transform
        return y1 / (y1 + y2)
