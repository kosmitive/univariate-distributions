import numpy as np
import random

# some fixed numbers for good stochastic performance.
from src.prob_distribution import ProbDist
from src.spaces.spaces1d_leafs import ContinuousSpace

mx = 2 ** 32 - 209
my = 2 ** 21 - 22853
axt1 = 1403580
axt2 = 810728
ayt0 = 527612
ayt2 = 1370589


class UniformDist(ProbDist):
    """Simple uniform distribution."""

    def __init__(self, a = 0, b = 1):
        """Create U(a, b) distribution.

        :param a The left boundary of the interval.
        :param b The right boundary of the interval.
        """

        self.a = a
        self.b = b

        # variables
        self.X = [random.randint(0, mx) for _ in range(3)]
        self.Y = [random.randint(0, my) for _ in range(3)]

        super().__init__(ContinuousSpace(a, b, open_brackets=False))

    def expectation(self):
        """Calculates the expectations for that distribution.

        :returns The expectation of the distribution"""

        a = self.a
        b = self.b
        return (a + b) / 2

    def var(self):
        """Calculates the variance for that distribution.

        :returns The variance of the distribution"""

        a = self.a
        b = self.b
        return (a - b) ** 2 / 12

    def sample(self, num_samples = 1):
        """Generate random numbers from U(a, b) by using a CMRG with parameters
        from [1] which is called MRG32k3a.

        :param num_samples How many random numbers should be generated.
        :returns A random number from U(a,b).


        Refs: [1] https://pubsonline.informs.org/doi/pdf/10.1287/opre.47.1.159.
        """

        a = self.a
        b = self.b

        elements = np.empty(num_samples)
        for k in range(num_samples):

            # two MRG
            x = (axt1 * self.X[1] + axt2 * self.X[2]) % mx
            y = (ayt0 * self.Y[0] + ayt2 * self.Y[2]) % my

            # combine
            u = (x - y + (mx if x <= y else 0)) / (mx + 1)
            elements[k] = a + u * (b - a)

            # update state
            self.X[1:] = self.X[:-1]
            self.Y[1:] = self.Y[:-1]
            self.X[0] = x
            self.Y[0] = y

        return elements

    def c_pdf(self, x):
        """This method calculates the density U(x|a,b)=U(a,b).

        :param x What values should be evaluated.
        :returns The probability this element occurs.
        """

        a = self.a
        b = self.b
        return 1 / (b - a) * np.ones(np.shape(x))
