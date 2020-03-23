from src.probs.spaces.spaces1d_leafs import *


class ProbDist:
    """Interface for distributions."""

    def __init__(self, space):
        """Remember the space to check later on if inputs are valid."""

        assert isinstance(space, Space)
        self.space = space

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""
        pass

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""
        pass

    def sample(self, num_samples=1):
        """Generate random numbers from Dist().

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ Dist().
        """
        pass

    def __density(self, x):
        """This method calculates the density Dist(x).

        :param x Which value should be evaluated.
        :returns The probability that this element occurs.
        """
        pass

    def density(self, x):
        """This method calculates the density Dist(x).

        :param x Which value should be evaluated.
        :returns The probability that this element occurs.
        """

        assert np.all(self.space.contains(x))
        return self.__density(x)
