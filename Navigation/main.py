#!usr/bin/env python3

import os, sys, glob, random
import weakref

from matplotlib import pyplot as plt

from Behavior_planner import behavior
sys.path.append("../../carla9.13/PythonAPI/carla/")
from Controller import VehiclePIDController
from Utils.Util_functions import distance
# from navy import Local_planner
from collections import deque
from Utils.misc import draw_waypoints
from Local_planner import Local_planner

try:
    sys.path.append(glob.glob('../../carla9.13/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
width = 720
height = 480

from carla import ColorConverter as cc

try: 

    import pygame
except:
    raise RuntimeError("Import Error with pygame module")

import numpy as np

class World():
    def __init__(self,carla_world):
        self.world          = carla_world
        self.map            = self.world.get_map()
        self.camera_manager = None

        self.gamma          = 2.2
        self.player         = None
        self.restart()

    def restart(self):
        """
        Restarts the whole setup or initiazes the vehicle if not spwaned"""

        cam_index = self.camera_manager.index if self.camera_manager is not None else 0

        blueprint = self.world.get_blueprint_library().filter("model3")[0]

        if self.player is not None:

            spawn_point                 = self.player.get_transform()
            # spawn_point.location.z      += 20
            # spawn_point.location.roll   = 0.0
            # spawn_point.location.pitch  = 0.0
            self.destroy()
            self.player                 = self.world.try_spawn_actor(blueprint,spawn_point)

        while self.player is None:
            
            spawn_points    = self.map.get_spawn_points()

            spawn_point     = spawn_points[196]
            self.player     = self.world.try_spawn_actor(blueprint,spawn_point)

        self.camera_manager = CameraManager(self.player, self.gamma)
        self.camera_manager.set_sensor(cam_index)

    def render(self, display):
        """renders"""
        self.camera_manager.render(display)



    def destroy(self):
        """destorys player and sensors"""

        sensors = [self.camera_manager.sensor]

        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()

        if self.player is not None:
            self.player.destroy()

    def destroy_sensors(self):
        """destorys sensor actor"""
        
        self.camera_manager.destroy()
        self.camera_manager.sensor  = None
        self.camera_manager.index   = None

class CameraManager(object):
    """camera manager to manager all the cameras"""
    def __init__(self, parent_actor, gamma = 0.0):

        self.sensor     = None
        self.surface    = None
        self.parent     = parent_actor
        Attachment      = carla.AttachmentType

        self.transform_index    = 0
        self._camera_transforms = [(carla.Transform(carla.Location(x = -7, z = 5),carla.Rotation(pitch = -20)), Attachment.Rigid)]

        self.sensors = [["sensor.camera.rgb" , cc.Raw, "Camera RGB" , {}]]

        world       = self.parent.get_world()
        bp_library  = world.get_blueprint_library()

        for item in self.sensors:

            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(width))
                bp.set_attribute('image_size_y', str(height))

                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma))

            item.append(bp)

        self.index = None

    def set_sensor(self, index, notify = True, force_respawn = False):
        """adds camera sesor to the vehicle """

        index = index % len(self.sensors)

        needs_respawn = True if self.index is None else (force_respawn or (self.sensor[index][2] != self.sensor[self.index][2]))

        if needs_respawn:

            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None

            self.sensor = self.parent.get_world().spawn_actor(self.sensors[index][-1], self._camera_transforms[self.transform_index][0], attach_to = self.parent, attachment_type = self._camera_transforms[self.transform_index][1])
            weak_self   = weakref.ref(self)

            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

        self.index = index

    def next_sensor(self):
        """increases current sensor index by 1"""
        self.set_sensor(self.index + 1)

    def render(self,  display):
        """Pata nahi kya karta hai shayad display set karta hai jaha render karna hai """
        if self.surface is not None:
            display.blit(self.surface, (0,0))

    @staticmethod
    def _parse_image(weak_self, image):
        """This function process the camera data into a 3 channnel RGB image"""
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            pass

        else:

            image.convert(self.sensors[self.index][1])
            img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img = np.reshape(img, (image.height, image.width, 4))
            array = img[:,:,:3]
            array = array[:,:,::-1]

            self.surface = pygame.surfarray.make_surface(array.swapaxes(0,1))


def game_loop(debug=False):
    """ This function loads the world and streams 
    the overhead camera of the vehicle to the pygame window
    Param: debug: If you want to plot the trajectory on the road.
    return: void"""
    try:
        pygame.init()
        pygame.font.init()


        Client = carla.Client("localhost" , 2000)
        display = pygame.display.set_mode(( width, height) , pygame.HWSURFACE | pygame.DOUBLEBUF)

        clock = pygame.time.Clock()
        world = World(Client.get_world())
        
        behavior_planner = behavior(world.player)
        while True:
            # if controller.parse_events(client , world )
            world.render(display=display)
            pygame.display.flip()

            control = behavior_planner.run_step()
            # print(control)
            world.player.apply_control(control)
            if len(behavior_planner.local_planner.waypoints_queue) < 2:
                world.player.apply_control(carla.VehicleControl(throttle= 0.0, brake=0.50))
                break


        # plt.plot(traj[:,0],traj[:,1])           #plots 2D trajectory
        # # plt.show()
        # plt.savefig("trajectory_plot.png")

                # control = pid.run_step(80,z[0])
                # world.player.apply_control(control)
                # if distance([world.player.get_location().x,world.player.get_location().y], [z[0].transform.location.x,z[0].transform.location.y] )< 10:
                #         z=z[0].next(3)


    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        world.destroy()
        pygame.quit()


if __name__ == '__main__':
    game_loop(True)
