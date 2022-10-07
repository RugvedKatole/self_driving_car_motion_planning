# Copyright (c) @ RugvedKatole
#
# Author: Rugved Katole
# Affliation: Indian Institute of Bombay
# Date: 24 June 2022

from enum import Enum
import numpy as np
import carla
from collections import deque
from Utils.misc import draw_waypoints, get_speed
from constraint_limits import PCC_parameters
from PCC_carla import MPC
from trajectory_generator import trajectory_generator
from Controller import VehiclePIDController
import matplotlib.pyplot as plt

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class Local_planner:

    def __init__(self,vehicle,target_speed = 20):
        """Initialize basic paramerters"""
        self.vehicle = vehicle
        self.world = self.vehicle.get_world()
        self.map = self.world.get_map()

        self.vehicle_controller = None
        self.target_waypoint = None
        self.target_road_option = None

        self.waypoints_queue = None
        self.min_waypoint_queue_length = 10
        self.stop_waypoint_creation = False

        # Base parameters
        self.dt = 1.0 / 20.0
        self.target_speed = target_speed  # Km/h
        self.sampling_radius = 2.0
        self.args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self.dt}
        self.args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': self.dt}
        self.max_throt = 0.75
        self.max_brake = 0.3
        self.max_steer = 0.8
        self.offset = 0
        self.base_min_distance = 3.0
        self.follow_speed_limit = False

        self.params = PCC_parameters()
        self.U_prev = np.array([[0],[0]])

        #traj generator
        self.traj_gen = trajectory_generator()

        #initializing controller
        self.init_controller()
        self.compute_waypoints(60)

    def init_controller(self):
        """Controller initialization"""
        self.vehicle_controller = VehiclePIDController(self.vehicle,
                                                        args_lateral        = self.args_lateral_dict,
                                                        args_longitudinal   = self.args_longitudinal_dict,
                                                        offset              = self.offset,
                                                        max_throttle        = self.max_throt,
                                                        max_brake           = self.max_brake,
                                                        max_steering        = self.max_steer)

        # Compute the current vehicle waypoint
        # current_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        # self.target_waypoint, self.target_road_option = (current_waypoint, RoadOption.LANEFOLLOW)
        # self.waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def set_speed(self, speed):
        """
        Changes the target speed

        :param speed: new target speed in Km/h
        :return:
        """
        if self.follow_speed_limit:
            print("WARNING: The max speed is currently set to follow the speed limits. "
                  "Use 'follow_speed_limits' to deactivate this")
        self.target_speed = speed

    def follow_speed_limits(self, value=True):
        """
        Activates a flag that makes the max speed dynamically vary according to the speed limits

        :param value: bool
        :return:
        """
        self.follow_speed_limit = value

    def compute_waypoints(self, length):
        current_waypoint = self.map.get_waypoint(carla.Location(self.vehicle.get_transform().get_forward_vector()*get_speed(self.vehicle)/3.6) + self.vehicle.get_location())
        iter = int(length/10 -1) 
        self.traj_gen.update_Control_points(current_waypoint)
        for i in range(iter):
            if self.waypoints_queue is not None:
                self.waypoints_queue = np.append(self.waypoints_queue,self.traj_gen.get_trajectory(),axis=-1)
            else:
                self.waypoints_queue = self.traj_gen.get_trajectory()
                
            self.traj_gen.update_Control_points(map=self.map)

        return

    def frenet2cart(self,z,u):
        """converts frenet co-ordinates to cartesisan"""

        e_wpt = self.map.get_waypoint(self.vehicle.get_location())
        x = np.array([0.0]*z.shape[1])
        y = np.array([0.0]*z.shape[1])
        for i in range(z.shape[1]):
            x[i] = e_wpt.transform.location.x
            y[i] = e_wpt.transform.location.y + carla.Vector3D(x=0,y=z[1,i]-self.params.wl/2,z=0).dot(self.vehicle.get_transform().get_right_vector())

            e_wpt = e_wpt.next(self.params.x_d)[0]
        waypoints = np.array([x,y])
        return waypoints


    def MPC_planner(self,vehicle,distance,debug=False):
        """MPC planner"""
        Lead_veh = vehicle.get_transform()
        Ego_veh = self.vehicle.get_transform()

        self.params.XE0 = 0
        self.params.XL0 = distance

        self.params.vr = self.vehicle.get_velocity().dot(Ego_veh.get_forward_vector())   #vx
        self.params.vl = vehicle.get_velocity().dot(Lead_veh.get_forward_vector())

        z_initial = np.array([self.params.vr-self.params.vl,self.params.wl/2])

        z,u = MPC(z_initial, self.U_prev, self.params.Q, self.params.R, self.params.S, self.params)

        waypoints = self.frenet2cart(z,u)

        trajectory = np.array([waypoints[0,:], waypoints[1,:], z[0,:]])
        
        self.waypoints_queue = trajectory

        # plt.plot(x,z_des[1])

    # def MPC_planner(self,vehicle, distance,debug=False):
    #     """Exucte one step of local planning for overtaking a lead vehicle"""
        
    #     Lead_veh_loc = vehicle.get_location()   #get location, x,y,z
    #     Ego_veh_loc = self.vehicle.get_location()
    #     vl = vehicle.get_velocity()
    #     ve = self.vehicle.get_velocity()

    #     self.params.XL0 = Lead_veh_loc.x
    #     self.params.YL0 = Lead_veh_loc.y

    #     self.params.XE0 = Ego_veh_loc.x
    #     self.params.YE0 = Ego_veh_loc.y

    #     self.params.vr = ve.x
    #     self.params.vl = vl.x

    #     e_wpt = self.map.get_waypoint(self.vehicle.get_location())
    #     L_wpt = self.map.get_waypoint(vehicle.get_location())
    #     wl = e_wpt.lane_width
    #     self.params.wl = wl
        

    #     N = int(self.params.x_f/self.params.x_d)
    #     vr = np.array([ve.x]*N)
    #     y = np.array([0.0]*N)
    #     x = np.array([0.0]*N)
    #     ymin = np.array([0.0]*N)
    #     ymax = np.array([0.0]*N)

    #     w = 1

    #     current_wpt = e_wpt
    #     wl = current_wpt.transform.location.y + current_wpt.lane_width/2
    #     wr = current_wpt.transform.location.y - current_wpt.lane_width/2
    #     for i in range(N):
    #         vr[i] = ve.x - vl.x
    #         if self.params.XL0 - self.params.lLF <= self.params.XE0 + self.params.x_d*i*self.params.vr/abs(self.params.vr) and self.params.XE0 + self.params.x_d*i*self.params.vr/abs(self.params.vr) <= self.params.XL0 + self.params.lLr:
    #             y[i] = current_wpt.get_right_lane().transform.location.y
    #             x[i] = current_wpt.get_right_lane().transform.location.x
                
    #             ymin[i] = wl + w

    #         else:
    #             y[i] = current_wpt.next(1)[0].transform.location.y
    #             x[i] = current_wpt.next(1)[0].transform.location.x

    #             wl = y[i] + current_wpt.lane_width/2
    #             wr = y[i] - current_wpt.lane_width/2

    #             ymin[i] = wr + w

    #         if self.params.XL0 - self.params.ls <= self.params.XE0 + self.params.x_d*i*self.params.vr/abs(self.params.vr) and self.params.XE0 + self.params.x_d*i*self.params.vr/abs(self.params.vr) <= self.params.XL0 + self.params.le:
                
    #             ymax[i] = wl + current_wpt.lane_width - w

    #         else:
    #             ymax[i] = wl - w

    #         current_wpt = current_wpt.next(1)[0]
        
    #     z_des = np.array([vr,y])
        
    #     z_initial = np.array([ve.x-vl.x,self.params.YE0])

    #     # print(z_initial)
    #     # print(self.U_prev)
    #     # print("vx",ve.x)
    #     # print("vl",vl.x)
    #     # print("XL0",self.params.XL0)
    #     # print("YLO",self.params.YL0)
    #     # print("XE0",self.params.XE0)
    #     # print("YE0",self.params.YE0)

         
    #     z,u = MPC(z_initial, self.U_prev, z_des, x, ymin, ymax, self.params.Q, self.params.R, self.params.S, self.params)
    #     y = z[1]
    #     v = z[0]
    #     # self.U_prev = u
    #     #z[0]  x velocity, z[1] optimal y co-ordinate, z is 2d array Nx2

    #     # plt.plot(x,z_des[1])
    #     # plt.plot(x,ymin)
    #     # plt.plot(x,ymax)
    #     # # plt.plot(x,z[1])
    #     # plt.legend(["ref","ymin","ymax"])
    #     # plt.show()

    #     # x = np.array([i*self.params.x_d + self.params.vl*i*self.params.x_d/z[0][i] for i in range(N)])
    #     self.U_prev = u[:,0]

    #     if z is not None:
    #         self.waypoints_queue = np.array([x,y,v])   #trajectory = (x,y,vx)
       
    #     # print(self.waypoints_queue.shape)
    #     # control = self.vehicle_controller.run_step(self.waypoints_queue[0,2]*3.6, [self.waypoints_queue[0,0],self.waypoints_queue[1,0]])
        
        
    #     if debug:
    #         draw_waypoints(self.vehicle.get_world(), [self.target_waypoint], 0)

    #     return 





    def run_step(self, debug = False):
        """ Execute one step of local planning which involves running the longitudinal and lateral PID controllers to follow the waypoints trajectory
        Param: debugL debuging parameter
        return: control """
        if self.follow_speed_limit:
            # self.target_speed = self.vehicle.get_speed_limit()
            self.target_speed = 10

        # Get waypoints for traversal
        # if len(self.waypoints_queue) < self.min_waypoint_queue_length:
        #     self.compute_waypoints(650)

        #using MPC to plan trajector
        # if len(self.waypoints_queue) < self.min_waypoint_queue_length:
        #     self.MPC_planner()

        Ego_veh_loc    = self.vehicle.get_location()
        vehicle_speed  = get_speed(self.vehicle) / 3.6
        min_distance = self.base_min_distance + vehicle_speed/2  # 0.5 speed in m/s

        if self.waypoints_queue.shape[1] > 0:
            for i in range(len(self.waypoints_queue)-1):
                fwd_vec = self.vehicle.get_transform().get_forward_vector()
                print(self.waypoints_queue.shape)
                wpt_vector = carla.Vector3D(x=self.waypoints_queue[0,0], y=self.waypoints_queue[1,0], z = Ego_veh_loc.z)
                if len(self.waypoints_queue)==1:
                    min_distance = 1
                if (wpt_vector - Ego_veh_loc).dot(fwd_vec) < min_distance:
                    self.waypoints_queue = np.delete(self.waypoints_queue,0,axis=1)
                else:
                    break


        if self.waypoints_queue.shape[1] == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
        else:
            self.target_waypoint = self.waypoints_queue[0:2,0]
            try:
                self.target_speed = self.waypoints_queue[2,0]*3.6
            except IndexError:
                self.target_speed = 10
            control = self.vehicle_controller.run_step(self.target_speed, self.target_waypoint)
        
        

        if debug:
            draw_waypoints(self.vehicle.get_world(), [self.target_waypoint], 10.0)

        return control
    
    def get_incoming_waypoint_and_direction(self, steps=3):
        """
        Returns direction and waypoint at a distance ahead defined by the user.

            :param steps: number of steps to get the incoming waypoint.
        """
        if len(self.waypoints_queue) > steps:
            return self.waypoints_queue[steps]

        else:
            try:
                wpt, direction = self.waypoints_queue[0:2,-1]
                return wpt, direction
            except IndexError as i:
                return None, RoadOption.VOID

    def get_plan(self):
        """Returns the current plan of the local planner"""
        return self.waypoints_queue