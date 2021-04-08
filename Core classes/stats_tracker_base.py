#!/usr/bin/env python3

# Automate the following for floating point values:
#   Checking that the values are within the specified range (ValueError)
#   Track the minimum, maximum, and average values found
#      Optionally: What index they were found at
#
# See Code Design: Core functionality of simulator
#   https://docs.google.com/document/d/1n8lx0HtgjiIuXMuVhB-bQtbvZuE114Cetnb8XVvn0yM/edit?usp=sharing
#

class StatsTrackerBase:
    def __init__(self, allowable_min, allowable_max):
        """ Dimension sizes will be found in allowable min/max - they should match
        @param allowable_min - np array or single number
        @param allowable_max - np array or single number"""
        self.allowable_min = allowable_min
        self.allowable_max = allowable_max
        #
        self.min_found = allowable_max * 1e6
        self.max_found = allowable_min * 1e-6
        self.avg_found = allowable_max * 0.9
        self.count = 0

        self.reset()

    def __str__(self):
        return self.get_name() + \
               " Min: {0:0.2f}".format(self.min_found) + \
               " Max: {0:0.2f}".format(self.max_found) + \
               " Avg: {0:0.2f}".format(self.avg_found) + \
               " N: {}".format(self.count)

    def __repr__(self):
        return self.__str__()

    def reset(self):
        self.min_found = self.allowable_max * 1e6
        self.max_found = self.allowable_min * 1e-6
        self.avg_found = self.allowable_min * 0.0
        self.count = 0

    def get_name(self):
        """ This should be over-written by whatever class is inheriting from this one"""
        return "Unnamed"

    def set_value(self, val):
        """Wherever there's an equal/data, use this. It will check for allowable values and update the stats"""
        if val < self.allowable_min:
            raise ValueError("{0}, {1} less than min value {2}".format(self.get_name(), val, self.min_found))
        if val > self.allowable_max:
            raise ValueError("{0}, {1} greater than max value {2}".format(self.get_name(), val, self.max_found))
        self.min_found = min(self.min_found, val)
        self.max_found = max(self.max_found, val)
        n = self.count+1
        self.avg_found = self.avg_found * (self.count / n) + val * (1.0 / n)
        self.count = n


if __name__ == "__main__":
    from numpy import array as nparray
    foo = nparray([0.3, 0.2, 0.1])
    bar = nparray([0.7, 0.1, 0.1])
    blah = min(foo, 1)

    my_test = StatsTrackerBase(3, 4)
    print("Before any set: {0}".format(my_test))

    my_test.set_value(3.25)
    my_test.set_value(3.75)
    print("After set: {0}".format(my_test))
    if my_test.min_found != 3.25:
        raise ValueError("Min: Expected 3.25, got {0}".format(my_test.min_found))
    if my_test.max_found != 3.75:
        raise ValueError("Max: Expected 3.75, got {0}".format(my_test.max_found))
    if my_test.avg_found != 3.5:
        raise ValueError("Avg: Expected 3.5, got {0}".format(my_test.avg_found))

    try:
        my_test.set_value(5)
        print("{0}, failed check, past max: given {1}, max was {2}".format(my_test.get_name(), 5, my_test.allowable_max))
    except ValueError:
        pass

    try:
        my_test.set_value(3)
        print("{0}, failed check, past min: given {1}, min was {2}".format(my_test.get_name(), 5, my_test.allowable_min))
    except ValueError:
        pass


