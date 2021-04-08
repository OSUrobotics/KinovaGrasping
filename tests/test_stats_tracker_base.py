#!/usr/bin/env python3


import unittest
from stats_tracker_base import StatsTrackerBase

class TestStatsTracker(unittest.TestCase):
    def test_min_max_float(self):
        """ Test the set_value method by manually setting to 3 values and checking the result
        Checks: min, max, avg correctly set functionality"""
        my_test = StatsTrackerBase(3, 4)

        my_test.set_value(3.25)
        my_test.set_value(3.75)

        self.assertEqual(my_test.min_found, 3.25, msg="Min: Expected 3.25, got {0}".format(my_test.min_found))
        self.assertEqual(my_test.max_found, 3.75, msg="Max: Expected 3.75, got {0}".format(my_test.max_found))

    def test_bounds_float(self):
        """ Test the """
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

if __name__ == '__main__':
    unittest.main()
