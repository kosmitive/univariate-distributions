import numpy as np

from src.continuous.uniform import UniformDist
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import DiscreteSpace


class DPhaseTypeDist(ProbDist):
    """Simple DPH(alpha, A) distribution."""

    def __init__(self, alpha, A):
        """Create DPH(alpha, A) distribution.

        :param alpha probability vector 1xm
        :param A mxm matrix such that (I - A) is invertible.
        """

        # save params
        assert np.sum(alpha) == 1
        assert np.shape(A)[0] == np.shape(A)[1] == len(alpha)
        self.alpha = np.expand_dims(alpha, 0)
        self.A = np.array(A)
        self.m = np.shape(A)[1]

        # create distribution for sampling
        self.UG = UniformDist()
        super().__init__(DiscreteSpace(1, np.inf))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        alpha = self.alpha
        A = self.A
        m = self.m

        return alpha @ np.linalg.inv((np.eye(m) - A)) @ np.ones([m, 1])

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        alpha = self.alpha
        A = self.A
        m = self.m
        exp = self.expectation()

        return alpha @ A @ np.linalg.matrix_power(np.eye(m) - A, 2) @ np.ones([m, 1]) + exp - exp ** 2

    def sample(self, num_samples = 1):
        """Generate random numbers from DPH(alpha, A).

        :param num_samples How many random numbers should be generated.
        :returns Random numbers x ~ DPH(alpha, A).
        """

        pass

    def __density(self, x):
        """This method calculates the mass DPH(alpha, A).

        :param x Which value should be evaluated.
        :returns The probability this element occurs.
        """

        alpha = self.alpha
        A = self.A
        m = self.m

        return alpha @ np.linalg.matrix_power(A, x - 1) * (np.eye(m) - A) @ np.ones([m, 1])
