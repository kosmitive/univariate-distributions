# ==================================================================== #
# Relationalyze                                                        #
# ==================================================================== #
# Copyright (C) 2018  Markus Semmler                                   #
#                                                                      #
# This program is free software: you can redistribute it and/or modify #
# it under the terms of the GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or    #
# (at your option) any later version.                                  #
#                                                                      #
# This program is distributed in the hope that it will be useful,      #
# but WITHOUT ANY WARRANTY; without even the implied warranty of       #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        #
# GNU General Public License for more details.                         #
#                                                                      #
# You should have received a copy of the GNU General Public License    #
# along with this program. If not, see <http://www.gnu.org/licenses/>. #
# ==================================================================== #

import numpy as np


class Space:
    """A space is a general concept, where it can be discrete or continuous."""

    def __init__(self, check):
        """Defines a space, by having a check lambda.

        :param check A lambda verifying if the element is in the space.
        """

        self.check = np.vectorize(check)

    def contains(self, x):
        """This method checks whether the space contains the list of samples.

        :param x The list of points.
        :returns True, if all elements inside.
        """

        if isinstance(x, set):
            x = list(x)
        elif not isinstance(x, list):
            x = [x]

        return self.check(x)

    def cut(self, space):
        """Check if the current space intersects with the passed space.

        :return Passes back the result as a set.
        """

        pass


class ContinuousSpace(Space):
    """A continuous space [a,b(."""

    def __init__(self, a, b, open_brackets = True):
        """Defines a continuous interval, using the boundaries.

        :param a The left side of interval.
        :param b The right side of interval.
        :param open_brackets True if brackets shall be open.
        """

        self.a = a
        self.b = b
        self.open_brackets = open_brackets

        # make check lambda
        fn = np.greater if open_brackets else np.greater_equal

        def f(x): return np.logical_and(fn(x, a), fn(b, x))
        super().__init__(f)

    def cut(self, space):
        """Check if the current space intersects with the passed space.

        :return Passes back the result as a set.
        """

        if isinstance(space, ContinuousSpace):

            if space.b < self.a or self.b < space.a:
                return NullSpace()

            term_s = np.maximum(self.a, space.a)
            term_e = np.minimum(self.b, space.b)
            return ContinuousSpace(term_s, term_e, open_brackets=self.open_brackets)

        # twist the plot
        return space.cut(self)


class DiscreteSpace(Space):
    """A discrete space which holds a set of elements."""

    def __init__(self, s, e):
        """Defines a discrete set.

        :param s Start index
        :param e End index
        """

        self.s = s
        self.e = e

        # create lambda and pass to super
        def f(x): return np.logical_and(np.greater_equal(x, s), np.greater(e, x))
        super().__init__(f)

    def cut(self, space):
        """Check if the current space intersects with the passed space.

        :return Passes back the result as a set.
        """

        la = isinstance(space, DiscreteSpace)
        lc = isinstance(space, ContinuousSpace)

        # set correctly
        lim = None
        if lc: lim = [space.a, space.b]
        elif la: lim = [space.s, space.e]

        # if it is a discrete set cut
        if la or lc:

            term_s = np.maximum(self.s, lim[0])
            term_e = np.minimum(self.e, lim[1])
            return DiscreteSpace(term_s, term_e + 1)

        space.cut(self)


class NullSpace(Space):
    """A space which holds nothing."""

    def __init__(self):
        """Simply reject all training inputs."""
        # make check lambda
        f = lambda x: False
        check = np.vectorize(f)
        super().__init__(check)

    def cut(self, space):
        """Empty set cut with any space is empty.

        :return Always the empty set.
        """
        return self
