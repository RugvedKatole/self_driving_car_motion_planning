# Copyright (c) @ RugvedKatole
#
# Author: Rugved Katole
# Affliation: Indian Institute of Bombay
# Date: 24 June 2022

from collections import deque
from math import sqrt
import carla
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from Utils.Util_functions import distance, get_x_y, get_speed, Bezier_curve
from Utils.misc import positive
from Controller import VehiclePIDController


"""Defining parameters for simulation"""

REF_VELOCITY    = 15   # in m/s
MAX_A_LAT       = 1   # m/s^2
NUM_POINTS      = 50   # NUMBER OF POINT ON BEZIER CURVE TO BE SAMPLED FOR TRACKING
CURVE_DEG       = 3     # DEGREE OF BEZIER CURVE
SAMPLE_RAD      = 10    # distance between 2 control points of CURVE
CURVE_TYPE      = "Bezier"



class trajectory_generator():
    """planner for generating trajectory"""
    def __init__(self,sampling_rad = None):
        self.waypoints = None
        self.Control_points = deque(maxlen=4)
        if sampling_rad is None:
            self.sampling_radius = SAMPLE_RAD
        else:
            self.sampling_radius = sampling_rad
    

    def get_trajectory(self,sample_pts=None, deg_curve=None):
        
        # if len(self.Control_points) > CURVE_DEG + 1:
        if sample_pts is None:
            sample_pts = NUM_POINTS
        if deg_curve is None:
            deg_curve = CURVE_DEG
        traj_pts = []
        for i in np.linspace(0,1,sample_pts):
            traj_pts.append(Bezier_curve(i,self.Control_points))
        return np.array(traj_pts).T

    def update_Control_points(self,vehicle_waypoint=None,map=None):

        if vehicle_waypoint is None and len(self.Control_points) != 0:
            next_waypoint = [map.get_waypoint(carla.Location(x=self.Control_points[-1][0],y=self.Control_points[-1][1]))]
        elif vehicle_waypoint is not None:
            self.Control_points.append(get_x_y(vehicle_waypoint.transform.location))
            next_waypoint = vehicle_waypoint.next(self.sampling_radius)

        for i in range(CURVE_DEG):
            if len(next_waypoint) == 1:
                self.Control_points.append(get_x_y(next_waypoint[0].transform.location))
                next_waypoint = next_waypoint[0].next(self.sampling_radius)
            else:
                self.Control_points.append(get_x_y(next_waypoint[-1].transform.location))
                next_waypoint = next_waypoint[-1].next(self.sampling_radius)
        return 
    
if __name__ == "__main__":
    T = trajectory_generator()
    C=carla.Client("localhost",2000)
    w= C.get_world()
    m= w.get_map()
    way = m.get_waypoint(carla.Location(x=533,y=-17))
    T.update_Control_points(way)
    print(T.get_trajectory().shape)