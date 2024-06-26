# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains the different parameters sets for each behavior. """


class Cautious(object):
    """Class for Cautious agent."""
    max_speed = 40
    speed_lim_dist = 6
    speed_decrease = 12
    safety_time = 3
    min_proximity_threshold = 12
    braking_distance = 6
    tailgate_counter = 0
    detect_dist     = 50


class Normal(object):
    """Class for Normal agent."""
    max_speed = 10
    speed_lim_dist = 0
    speed_decrease = 10
    safety_time = 3
    min_proximity_threshold = 55
    braking_distance = 5
    tailgate_counter = 0
    detect_dist     = 55  #detection distance in meters
    overtaking_distance = 50


class Aggressive(object):
    """Class for Aggressive agent."""
    max_speed = 70
    speed_lim_dist = 1
    speed_decrease = 8
    safety_time = 3
    min_proximity_threshold = 8
    braking_distance = 4
    tailgate_counter = -1
    detect_dist     = 28
