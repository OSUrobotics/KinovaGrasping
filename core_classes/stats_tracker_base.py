#!/usr/bin/env python3

# Automate the following for floating point values:
#   Checking that the values are within the specified range (ValueError)
#   Track the minimum, maximum, and average values found
# Usage as a tracker, store value elsewhere:
#   my_stats_tracker_variable_x = StatsTrackerBase.create_instance(min_bds, max_bds)
#   my_stats_tracker_variable_x.set_value( ... ) for setting the value
#   my_stats_tracker_variable_x.value for getting the value back
#
# See Code Design: Core functionality of simulator
#   https://docs.google.com/document/d/1n8lx0HtgjiIuXMuVhB-bQtbvZuE114Cetnb8XVvn0yM/edit?usp=sharing
#


class StatsTrackerBase:
    def __init__(self, allowable_min, allowable_max):
        """ Dimension sizes will be found in allowable min/max - they should match
        @param allowable_min - single number
        @param allowable_max - single number"""
        self.allowable_min = allowable_min
        self.allowable_max = allowable_max
        self.value = None
        # Do it this way because we'll override in reset
        self.min_found = None
        self.max_found = None
        self.avg_found = None
        self.count = 0

        # override reset for the different data types
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
        """ Reset for floats/arrays. Should probably be NaN, but avoids having to check for min/max being set
        @param can optionally set the value"""
        if self.allowable_max < self.allowable_min:
            raise ValueError("{0} max less than min".format(self))
        self.min_found = self.allowable_max * 1e6
        self.max_found = self.allowable_min * 1e-6
        self.avg_found = self.allowable_min * 0.0
        self.count = 0
        self.value = None

    def get_name(self):
        """ This should be over-written by whatever class is inheriting from this one"""
        return "{0}: ".format(self.__class__.__name__)

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

        self.value = val

    @staticmethod
    def create_instance(min_bds, max_bds, debug=True):
        """ Create an instance based on the type (float/int or array)
        TODO Be able to pass in the get_name method so you can name the variable
        @param min_bds is either a number or an array
        @param max_bds should match dimensionality of min_bds
        @returns An instance of StatsTrackerBase (floats/ints) or StatsTrackerArray(array, numpy array) or DoNothing (debug = false"""
        if not debug:
            return StatsTrackerDoNothing()

        try:
            return StatsTrackerBase(min_bds, max_bds)
        except:
            return StatsTrackerArray(min_bds, max_bds)

class StatsTrackerArray(StatsTrackerBase):
    """ Overrides just the methods that need to do array accesses"""
    def __init__(self, allowable_min, allowable_max):
        """ Dimension sizes will be found in allowable min/max - they should match
        @param allowable_min - array
        @param allowable_max - array"""
        # Will call reset() and set the _found variables to be the right size
        super(StatsTrackerArray, self).__init__(allowable_min, allowable_max)

    def __str__(self):
        return self.get_name() + \
               " Min: [" + ",".join(["{0:0.2f}".format(v) for v in self.min_found]) + "]" + \
               " Max: [" + ",".join(["{0:0.2f}".format(v) for v in self.max_found]) + "]" + \
               " Avg: [" + ",".join(["{0:0.2f}".format(v) for v in self.avg_found]) + "]" + \
               " N: {}".format(self.count)

    def reset(self):
        """ Have to set all of the elements in the array - indirectly checks that the arrays are same size"""
        for min_v, max_v in zip(self.allowable_min, self.allowable_max):
            if max_v < min_v:
                raise ValueError("{0} max less than min".format(self))

        self.min_found = [v * 1e6 for v in self.allowable_max]
        self.max_found = [v * 1e-6 for v in self.allowable_min]
        self.avg_found = [0 for _ in self.allowable_max]
        self.count = 0

    def set_value(self, val):
        """Wherever there's an equal/data, use this. It will check for allowable values and update the stats"""
        for i, v in enumerate(val):
            if v < self.allowable_min[i]:
                raise ValueError("{0}, {1} less than min value {2}, index {3}".format(self.get_name(), val, self.min_found, i))
            if v > self.allowable_max[i]:
                raise ValueError("{0}, {1} greater than max value {2}, index {3}".format(self.get_name(), val, self.max_found, i))

            self.min_found[i] = min(self.min_found[i], v)
            self.max_found[i] = max(self.max_found[i], v)

            n = self.count+1
            self.avg_found[i] = self.avg_found[i] * (self.count / n) + v * (1.0 / n)

        self.count += 1
        self.value = val


class StatsTrackerDoNothing(StatsTrackerBase):
    def __init__(self, *args):
        self.value = None
        pass

    def reset(self):
        pass

    def __str__(self):
        return self.get_name()

    def set_value(self, val):
        """Just keep the value"""
        self.value = val


if __name__ == "__main__":
    my_test_float = StatsTrackerBase(3, 4)
    my_test_array = StatsTrackerArray([3, 4, 5], [4, 10, 12])

    print("{0}".format(my_test_float))
    print("{0}".format(my_test_array))


