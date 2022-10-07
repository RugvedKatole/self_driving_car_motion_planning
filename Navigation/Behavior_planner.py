# Copyright (c) @ RugvedKatole
#
# Author: Rugved Katole
# Affliation: Indian Institute of Bombay
# Date: 24 June 2022

import numpy as np  
import carla
from Utils.misc import compute_distance, is_within_distance, positive
from Utils.Util_functions import get_speed
from shapely.geometry import Polygon
from behavior_types import Cautious, Aggressive, Normal
from Local_planner import Local_planner, RoadOption

class behavior :
    """ describes the behavior of car"""
        
    def __init__(self,ego_vehicle,target_speed=20):
        """ initializes """
        self.look_ahead_steps = 0

        self.ego_vehicle = ego_vehicle
        self.world = self.ego_vehicle.get_world()
        self.map = self.world.get_map()

        # Vehicle information
        self.speed = 0
        self.speed_limit = 0
        self.direction = None
        self.incoming_direction = None
        self.incoming_waypoint = None
        self.min_speed = 5
        self.sampling_resolution = 4.5

        # Parameters for agent behavior
        self.behavior = Normal()

        # Base parameters
        self.target_speed = target_speed
        self.sampling_resolution = 2.0
        self.base_tlight_threshold = 5.0  # meters
        self.base_vehicle_threshold = 75.0  # meters
        self.max_brake = 0.5

        #local planner
        self.local_planner = Local_planner(self.ego_vehicle)

    def emergency_stop(self):
        control = carla.VehicleControl()
        control.throttle    = 0.0
        control.brake       = self.max_brake
        control.hand_brake = False
        return control

    def update_information(self):
        """updates information of surrounding world of ego vehicel"""
        self.speed          = get_speed(self.ego_vehicle)
        self.speed_limit    = self.ego_vehicle.get_speed_limit()
        self.local_planner.set_speed(self.speed_limit)
        self.direction      = self.local_planner.target_road_option
        if self.direction is None:
            self.direction = RoadOption.LANEFOLLOW
        
        self.look_ahead_steps = int(self.speed_limit/10)

    def vehicle_obstacle_detected(self,vehicle_list = None, max_distance = 75, up_angle_th = 90, low_angle_th = 0, lane_offset = 0):
        """ Method to check if there is any vehicle blocking the path of our ego vehicle
        
        :param vehicle_list (list of carla.Vehicle): list containing vehicle objects
            if None, all vehicle in the scene are used
        :param max_distance: max freespace to check for obstacles.
            If None, the base threshold value is used"""

        if not vehicle_list:
            vehicle_list = self.world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self.base_vehicle_threshold

        ego_transform   = self.ego_vehicle.get_transform()
        ego_wpt         = self.map.get_waypoint(self.ego_vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1
        
        ego_forward_vector  = ego_transform.get_forward_vector()
        ego_extent          = self.ego_vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location( x = ego_extent * ego_forward_vector.x,
                                                        y = ego_extent * ego_forward_vector.y)
        for target_vehicle in vehicle_list:
            target_transform    = target_vehicle.get_transform()
            target_wpt          = self.map.get_waypoint(target_transform.location, lane_type = carla.LaneType.Any)

            # Simplified version for outside junctions
            if not ego_wpt.is_junction or not target_wpt.is_junction:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                    next_wpt = self.local_planner.get_incoming_waypoint_and_direction(steps=10)
                    next_wpt = self.map.get_waypoint(carla.Location(x = next_wpt[0],y = next_wpt[1]))
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                        continue

                target_forward_vector   = target_transform.get_forward_vector()
                target_extent           = target_vehicle.bounding_box.extent.x
                target_rear_transform   = target_transform
                target_rear_transform.location -= carla.Location(
                    x = target_extent * target_forward_vector.x,
                    y = target_extent * target_forward_vector.y
                )
                
                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))
            
            # Waypoints aren't reliable, check the proximity of the vehicle to the route
            else:
                route_bb = []
                ego_location = ego_transform.location
                extent_y = self.ego_vehicle.bounding_box.extent.y
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

                for wp in self.local_planner.get_plan():
                    wp = self.map.get_waypoint(carla.Location(x=wp[0],y=wp[1])) 
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break

                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])

                if len(route_bb) < 3:
                    # 2 points don't create a polygon, nothing to check
                    return (False, None, -1)
                ego_polygon = Polygon(route_bb)

                # Compare the two polygons
                for target_vehicle in vehicle_list:
                    target_extent = target_vehicle.bounding_box.extent.x
                    if target_vehicle.id == self.ego_vehicle.id:
                        continue
                    if ego_location.distance(target_vehicle.get_location()) > max_distance:
                        continue

                    target_bb = target_vehicle.bounding_box
                    target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                    target_list = [[v.x, v.y, v.z] for v in target_vertices]
                    target_polygon = Polygon(target_list)

                    if ego_polygon.intersects(target_polygon):
                        return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

                return (False, None, -1)

        return (False, None, -1)

    def collision_and_car_avoid_manager(self,waypoint):
        
        """This function or module will issue a warning incase of a collision 
        
        :param location : current location of the agent type carla.location
        :param waypoint: current waypoint of the agent :type carla.waypoint
        :return vehicel state: True if there is a vehicle nearby, False if not 
        :return vehicle: carla.Vehicle of nearby vehicle
        :return distance: distance to the nearby vehicle"""

        vehicle_list = self.world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < self.behavior.detect_dist and v.id != self.ego_vehicle.id]

        if self.direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self.vehicle_obstacle_detected(
                vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self.direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self.vehicle_obstacle_detected(
                vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self.vehicle_obstacle_detected(
                vehicle_list, self.behavior.min_proximity_threshold, up_angle_th=30)

        return vehicle_state, vehicle, distance

    def car_following_manager(self,vehicle, distance, debug = False):
        """ Module for following the car infornt of ego_vehicle
        
        :param vehicle: vehicle to follow
        :param distance: distance from vehicle
        :param debug: debugginf boolean
        :return controlL carla.vehicelControl"""

        vehicle_speed   = get_speed(vehicle)
        delta_v         = max(1, (self.speed - vehicle_speed)/3.6)
        ttc             = distance / delta_v if delta_v != 0 else distance / np.nextafter(0.,1.)  # time to collision T = Distance/Relative_Velocity


        if self.behavior.safety_time > ttc > 0.0:
            target_speed = min([positive(vehicle_speed - self.behavior.speed_decrease),self.behavior.max_speed, self.speed_limit])
            self.local_planner.set_speed(target_speed)
            control = self.local_planner.run_step(debug=debug)
        

        elif 2*self.behavior.safety_time > ttc >= self.behavior.safety_time:
            target_speed = min([max([self.min_speed, vehicle_speed]), self.behavior.max_speed, self.speed_limit])
            self.local_planner.set_speed(target_speed)
            control = self.local_planner.run_step(debug=debug)
        
        #Normal behavior
        else:
            target_speed = min([self.behavior.max_speed,self.speed_limit])
            self.local_planner.set_speed(target_speed)
            control = self.local_planner.run_step(debug=debug)

        return control


    def run_step(self, debug = False):
        """
        Executes one step of navigation.
        
            :param debug: boolean for debugging
            :return control: carla.VechicleControl
        """

        self.update_information()

        control = None
        ego_vehicle_loc = self.map.get_waypoint(self.ego_vehicle.get_location())
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_loc)

        if vehicle_state:
            print("vehicle detected at:",distance)
            # Distance is computed from the centre of the two cars,
            #we use bounding boxes to calculate the actual distance
            distance = distance - max(vehicle.bounding_box.extent.y,vehicle.bounding_box.extent.x) - max(
                self.ego_vehicle.bounding_box.extent.y, self.ego_vehicle.bounding_box.extent.x )

            #Emergency brake if the car is very close
            if distance < self.behavior.braking_distance:
                return self.emergency_stop()
            elif True:  #overtake if distance is enought to overtake
                if self.local_planner.min_waypoint_queue_length > 0:
                    print("MPC Planning")
                    self.local_planner.MPC_planner(vehicle,distance)
                    self.local_planner.min_waypoint_queue_length = -1
                control = self.local_planner.run_step()
            else:
                control = self.car_following_manager(vehicle,distance)
        else:
            target_speed = min([self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist])
            self.local_planner.set_speed(target_speed)
            control = self.local_planner.run_step(debug=debug)

        return control