import numpy as np

from src.continuous.normal import NormalDist
from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import ContinuousSpace


class WaldDist(ProbDist):
    """Simple Wald distribution."""

    def __init__(self, loc = 0, scale = 1):
        """Creates Wald(loc,scale) distribution.

        :param loc Location of the distribution.
        :param scale Scale of the distribution
        """

        # save params
        self.loc = loc
        self.scale = scale

        # create generator
        self.NG = NormalDist()
        self.UG = UniformDist()
        super().__init__(ContinuousSpace(0, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        return self.loc

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        return self.loc ** 3 / self.scale

    def sample(self, num_samples = 1):
        """Generate random numbers from Wald(loc,scale).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Wald(loc,scale).
        """

        # shortcut
        loc = self.loc
        scale = self.scale

        # some sampling
        W = self.NG.sample(num_samples)
        Y = W ** 2
        Z = loc + (loc ** 2 * Y / 2 * scale) + (loc / 2 * scale) * np.sqrt(4 * loc * scale * Y + loc ** 2 * Y ** 2)
        X = Z

        # sample uniforms
        bound = loc / (loc + Z)
        B = self.UG.sample(num_samples)

        # iterate
        for k in range(num_samples):
            if B[k] > bound: X[k] = loc ** 2 / Z[k]

        return X

    def c_pdf(self, x):
        """This method calculates the density Wald(loc,scale).

        :param x Which value should be evaluated.
        :returns The probability that this element occurs.
        """

        assert x > 0

        # shortcut
        loc = self.loc
        scale = self.scale

        # update x
        z = np.sqrt(scale / (2 * np.pi * np.power(x, 3)))
        b = np.exp(-0.5 * (scale / loc ** 2) * (np.subtract(x, loc) ** 2) / x)
        return z * b
