#usr/bin/env python3

# Author: Rugved Katole
# Affliation: Indian Institute of Technology, Bombay
# Date: 12 june 2022
# copyright @ Rugved Katole

from collections import deque
from math import sqrt
import carla
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from Utils.Util_functions import distance, get_x_y, get_speed
from Utils.misc import positive
from controller import Controller2D

"""Parameters Required"""
REF_VELOCITY    = 15   # in m/s
MAX_A_LAT       = 1   # m/s^2
NUM_POINTS      = 200   # NUMBER OF POINT ON BEZIER CURVE TO BE SAMPLED FOR TRACKING
CURVE_DEG       = 3     # DEGREE OF BEZIER CURVE
SAMPLE_RAD      = 10    # distance between 2 control points of CURVE

class Local_planner():
    """planner for generating trajectory"""
    def __init__(self, ego_vehicle):
        
        self.vehicle        = ego_vehicle
        self.trajectory     = None
        self.Control_points = deque(maxlen=4)
        self.target_speed   = REF_VELOCITY
        self.cp_count       = 0
        self.map            = self.vehicle.get_world().get_map()
        self.sampling_radius= SAMPLE_RAD
        self.PIDcontroller  = Controller2D(None)

    def update_info(self):

        # if distance(self.Control_points[-1],get_x_y(self.vehicle.get_location)) < 2 or self.trajectory is None:
        self.update_Control_points()
        self.generateTraj(self.Control_points)
        return
        

    def update_Control_points(self):

        if len(self.Control_points) == 0:
            self.Control_points.append(get_x_y(self.vehicle.get_location())) # first control point is vehicle location

        next_waypoint = self.map.get_waypoint(carla.Location(x=self.Control_points[-1][0],y=self.Control_points[-1][1])).next(self.sampling_radius)
        for i in range(CURVE_DEG):
            if len(next_waypoint) == 1:
                self.Control_points.append(get_x_y(next_waypoint[0].transform.location))
                next_waypoint = next_waypoint[0].next(self.sampling_radius)
            else:
                self.Control_points.append(get_x_y(next_waypoint[-1].transform.location))
                next_waypoint = next_waypoint[-1].next(self.sampling_radius)
        return


    def generateTraj(self, control_points):
        self.trajectory = self.Bezier_traj(control_points,NUM_POINTS)
        return

    def Bezier_traj(self, control_points, NUM_Sample):
        """ returns a bezier curve points trajectory"""

        trajectory = []

        for u in np.linspace(0,1,NUM_Sample):

            waypoint     = self.Bezier_curve(u, control_points)
            target_speed = self.get_targetSpeed(u,control_points)    # point.format ==  [x,y,v]
            trajectory.append([waypoint[0],waypoint[1],target_speed])

        return trajectory

    def get_targetSpeed(self,u,control_points):
        """returns speed for a waypoint in Km/h"""

        derivative_points   = self.bezier_derivatives_control_points(control_points, 2)
        b_dot1              = self.Bezier_curve(u,derivative_points[1])         #returns a list of [x_dot,y_dot]
        b_dot2              = self.Bezier_curve(u,derivative_points[2])

        kappa = (b_dot1[0]*b_dot2[1] - b_dot1[0]*b_dot2[1])/(b_dot1[0]**2 + b_dot1[1]**2)**1.5
        if kappa != 0:
            v_k  = sqrt(MAX_A_LAT/kappa)
        else:
            v_k  = np.inf

        target_speed = min(REF_VELOCITY,v_k)
        return target_speed*3.6



    def Bezier_curve(self,u,control_points):
        """
        Returns a point on the bezier curve
        :param t: float parameter in [0,1]
        :param control_points: numpy array of control points
        :returnL Co-ordinates of points
        """
        control_points = np.array(control_points)
        n = len(control_points) - 1
        return np.sum([self.berstein_poly(n,i,u)*control_points[i] for i in range(n + 1)],axis = 0)

    def berstein_poly(self,n,i,u):
        """
        ith term of a Berstein polynomial of n degree
        :param n: degree of the polynomial
        :param i: ith term
        :param t: parametic constant t

        """
        return scipy.special.comb(n,i)* u**i * (1-u)**(n-i)
    
    def bezier_derivatives_control_points(self,control_points, n_derivatives):
        """
            Compute control points of the successive derivatives of a given bezier curve.
        A derivative of a bezier curve is a bezier curve.
        See https://pomax.github.io/bezierinfo/#derivatives
        for detailed explanations
        :param control_points: (numpy array)
        :param n_derivatives: (int)
        e.g., n_derivatives=2 -> compute control points for first and second derivatives
        :return: ([numpy array])
        """
        control_points = np.array(control_points)
        w = {0: control_points}
        for i in range(n_derivatives):
            n = len(w[i])
            w[i + 1] = np.array([(n - 1) * (w[i][j + 1] - w[i][j]) for j in range(n - 1)])
        return w

    def get_waypoints(self,trajectory):
        """This function converts the generated trajectory to waypoints trajectory
        :param [[x1,y1,v1]...]
        
        :return [[waypoint,speed]...]"""
        waypoints = []
        for waypoint in trajectory:
            point = self.map.get_waypoint(carla.Location(x=waypoint[0],y=waypoint[1]))
            waypoints.append([point,waypoint[2]])

        return deque(waypoints)
    
    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control


    def run_step( self,debug = False):
        """ this runs one step in local planning"""

        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)

if __name__ == "__main__":
    # client      = carla.Client('localhost', 2000)
    # world       = client.get_world()
    # ego_vehicle = carla.Vehicle()
    # planer      = Local_planner(ego_vehicle)

    Local_planner.run_step()
    