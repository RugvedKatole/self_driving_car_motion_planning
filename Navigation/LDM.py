#!usr/bin/env python3

from collections import deque
import math
import numpy as np
import carla,sys
# sys.path.append("Utils")
from Utils.Util_functions import get_speed


"""
In this module contains Lyapunov based controller for lateral and longitudinal control
"""

class LDMcontroller():
    """
    This class has 2 controllers for low level control of EGO vehicle. Each for longitudinal and lateral control
    """

    def __init__(self,vehicle , args_lateral, args_longitudinal, offset=0, max_throttle = 0.75, max_brake = 0.3, max_steering=0.3):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        
        """

        self.max_brake      = max_brake
        self.max_throtlle   = max_throttle
        self.max_steer      = max_steering

        self._vehicle       = vehicle         # EGO Vehicle to be controlled
        self._world         = self._vehicle.get_world()
        self.past_steering  = self._vehicle.get_control().steer
        
        # gains 
        self.k1 = 1
        self.k2 = 1
        self.k3 = 1
        # self._long_control  = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        # self._lat_control   = PIDLateralController(self._vehicle, **args_lateral)

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal 
        control for a given waypoint and target speed
        :param target_speed: desired vehicle speed[v_r]
        :param waypoint: target location encoded as a waypoint[x_r,y_r,yaw]
        :return: control

        Theory for lyapunov based controller 
        error system q_e(x_e,y_e,theta_e) is given as:
        x_e     =  cos(theta)*(x_ref-x) + sin(theta)*(y_ref-y)
        y_e     = -sin(theta)*(x_ref-x) + cos(theta)*(y_ref-y)
        theta_e =  theta_ref - theta

        The open loop error system is stated as:
        dot(x_e)     =  w*y_e + v_r*cos(theta_e) - v
        dot(y_e)     = -w*x_e + v_r*sin(theta_e) - v
        dot(theta_e) =  w_d - w

        The control law for the system is stable is stated as:
        [v , w]' = [k1*x_e + v_r*cos(theta_e), w_r + v_r(k2*y_e +k3*sin(theta_e))]
        """
        #Parameter decleration
        vehicle_loc   = self._vehicle.get_location()
        vehicle_speed = get_speed(self._vehicle)
        vehicle_ang   = self._vehicle.get_angular_velocity().z
        theta         = self._vehicle.get_transform().rotation.yaw
        control       = carla.VehicleControl()

        x_e     =  np.cos(theta)*(waypoint[0]-vehicle_loc.x) + np.sin(theta)*(waypoint[1]-vehicle_loc.y)
        y_e     = -np.sin(theta)*(waypoint[0]-vehicle_loc.x) + np.cos(theta)*(waypoint[1]-vehicle_loc.y)
        theta_e =  waypoint[2] - theta

        throttle = self.k1*x_e + target_speed*math.cos(theta_e)
        current_steering = target_speed*(self.k2*y_e + self.k3*math.sin(theta_e))

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1

        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)

        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        throttle = math.tanh(throttle) 
        if throttle >=0.0:
            control.throttle = min(throttle,self.max_throtlle)
            control.brake    = 0.0
        else:
            control.throttle = 0.0
            control.brake    = min(abs(throttle),self.max_brake)

        return control