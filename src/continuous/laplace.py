import numpy as np

from src.continuous.exponential import ExpDist
from src.continuous.normal import NormalDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import ContinuousSpace


class LaplaceDist(ProbDist):
    """Simple Laplace distribution."""

    def __init__(self, loc = 0, scale = 1):
        """Creates Laplace(loc,scale) distribution.

        :param loc Location of the distribution.
        :param scale Scale of the distribution
        """

        # save params
        self.loc = loc
        self.scale = scale

        # create generator
        self.NG = NormalDist()
        self.EG = ExpDist()
        super().__init__(ContinuousSpace(-np.inf, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.loc

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return 2 * (self.scale ** 2)

    def sample(self, num_samples = 1):
        """Generate random numbers from Laplace(loc, scale) by using acceptance rejection distributions.

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Laplace(loc, scale).
        """

        # shortcut
        loc = self.loc
        scale = self.scale

        # sample data
        E = self.EG.sample(num_samples)
        Y = self.NG.sample(num_samples)
        X = Y * np.sqrt(2 * E)
        return scale * X + loc

    def c_pdf(self, x):
        """This method calculates the density Laplace(loc, scale).

        :param x Which value should be evaluated.
        :returns The probability that this element occurs.
        """

        # shortcut
        loc = self.loc
        scale = self.scale

        # update x
        xn = np.subtract(x, loc) / scale
        f = 0.5 * np.exp(-np.abs(xn))
        return f / scale
