#!/usr/bin/env python3


import unittest
from stats_tracker_base import StatsTrackerBase, StatsTrackerArray


class TestStatsTracker(unittest.TestCase):
    def test_check_print(self):
        my_test_float = StatsTrackerBase(3, 4)
        my_test_array = StatsTrackerArray([3, 4, 5], [4, 10, 12])

        print("{0}".format(my_test_float))
        print("{0}".format(my_test_array))

    def test_min_max_set_float(self):
        """ Test the set_value method by manually setting to 3 values and checking the result
        Checks: min, max, avg correctly set functionality"""
        my_test = StatsTrackerBase(3, 4)

        my_test.set_value(3.25)
        my_test.set_value(3.75)

        self.assertEqual(my_test.min_found, 3.25, msg="Min: Expected 3.25, got {0}".format(my_test.min_found))
        self.assertEqual(my_test.max_found, 3.75, msg="Max: Expected 3.75, got {0}".format(my_test.max_found))
        self.assertEqual(my_test.avg_found, 3.5, msg="Avg: Expected 3.5, got {0}".format(my_test.avg_found))
        self.assertEqual(my_test.count, 2)

    def test_bounds_float(self):
        """ Test that it generates errors for numbers out of bounds"""
        my_test = StatsTrackerBase(3, 4)
        msg = "{0}, past max: given {1}, max was {2}".format(my_test.get_name(), 5, my_test.allowable_max)
        with self.assertRaises(ValueError, msg=msg):
            my_test.set_value(5)

        msg = "{0}, past min: given {1}, min was {2}".format(my_test.get_name(), 2, my_test.allowable_min)
        with self.assertRaises(ValueError, msg=msg):
            my_test.set_value(3)

    def test_min_max_set_array(self):
        """ Test the set_value method by manually setting to 3 values and checking the result
        Checks: min, max, avg correctly set functionality"""
        my_test = StatsTrackerArray([3, 4, 5], [4, 10, 12])

        my_test.set_value([3.25, 4.25, 5.25])
        my_test.set_value([3.75, 4.75, 5.75])

        self.assertEqual(my_test.min_found, [3.25, 4.25, 5.25], msg="Min: Expected 3.25, got {0}".format(my_test.min_found))
        self.assertEqual(my_test.max_found, [3.75, 4.75, 5.75], msg="Max: Expected 3.75, got {0}".format(my_test.max_found))
        self.assertEqual(my_test.avg_found, [3.5, 4.5, 5.5], msg="Avg: Expected 3.5, got {0}".format(my_test.avg_found))

    def test_bounds_array(self):
        """ Test that it generates errors for numbers out of bounds"""
        my_test = StatsTrackerArray([3, 4, 5], [4, 10, 12])
        msg = "{0}, past max: given {1}, max was {2}".format(my_test.get_name(), 5, my_test.allowable_max)
        with self.assertRaises(ValueError, msg=msg):
            my_test.set_value([3.5, 11, 7])

        msg = "{0}, past min: given {1}, min was {2}".format(my_test.get_name(), 2, my_test.allowable_min)
        with self.assertRaises(ValueError, msg=msg):
            my_test.set_value([3.5, 8, 4])

    def test_init_checks(self):
        """ Check for array sizes, max < min """
        with self.assertRaises(ValueError):
            StatsTrackerBase(10, 6)

        with self.assertRaises(ValueError):
            StatsTrackerArray([5, 4, 5], [4, 10, 12])

        with self.assertRaises(IndexError):
            StatsTrackerArray([3, 5], [4, 10, 12])

        with self.assertRaises(IndexError):
            StatsTrackerArray([3, 5, 6], [4, 12])


if __name__ == '__main__':
    unittest.main()
