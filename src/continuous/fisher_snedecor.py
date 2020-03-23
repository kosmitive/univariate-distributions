import math as m
import numpy as np

from src.continuous.beta import BetaDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import ContinuousSpace


class FDist(ProbDist):
    """Simple F distribution."""

    def __init__(self, m = 1, n = 1):
        """Create F(m,n) distribution.

        :param m Degrees of freedom
        :param n Degrees of freedom
        """

        # save params
        self.m = m
        self.n = n
        self.BG = BetaDist(m / 2, n / 2)
        super().__init__(ContinuousSpace(0, np.inf, open_brackets=False))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.n / (self.n - 2)

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        n = self.n
        m = self.m
        return (2 * n ** 2 * (m + n - 2)) / (m * (n - 2) ** 2 * (n - 4))

    def sample(self, num_samples = 1):
        """Generate random numbers from F(m,n) by using a beta generator.

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ F(m,n).
        """

        # generate samples
        B = self.BG.sample(num_samples)

        # transform
        return self.n * B / (self.m * (1 - B))

    def c_pdf(self, x):
        """This method calculates the density F(m,n).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        # only support for positive x
        assert x >= 0

        # shortcuts
        m_dof = self.m
        n_dof = self.n
        m2 = m_dof / 2
        n2 = n_dof / 2
        mn2 = m2 + n2
        dmn = m_dof / n_dof

        # compute all parts
        t = m.gamma(mn2) * dmn ** m2 * np.power(x, m2 - 1)
        b = m.gamma(m2) * m.gamma(n2) * (1 + np.multiply(dmn, x)) ** mn2
        return t / b
