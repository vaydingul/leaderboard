#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla
from srunner.scenariomanager.timer import GameTime

from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.envs.sensor_interface import SensorInterface


class Router:
    def __init__(self, route, distance_threshold=5, closest=False) -> None:
        self.route = route
        self.route_length = len(route)
        self.distance_threshold = distance_threshold
        self.closest = closest
        self.route_index = 0

    def step(self, ego_location):
        if self.route_index < self.route_length - 1:
            if not self.closest:
                next_waypoint = self.route[self.route_index + 1][0]
                if (
                    next_waypoint.location.distance(ego_location)
                    < self.distance_threshold
                ):
                    self.route_index += 1

            else:
                # find the closest waypoint to the ego vehicle
                min_distance = float("inf")

                for i in range(
                    self.route_index, min(self.route_index + 5, self.route_length)
                ):
                    distance = self.route[i][0].location.distance(ego_location)
                    if distance < min_distance:
                        min_distance = distance
                        route_index = i

                if route_index > self.route_index:
                    self.route_index = min(route_index, self.route_length - 1)

        return self.route[self.route_index + 1]

    def get_remaining_route(self):
        return self.route[self.route_index :]


class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """

    SENSORS = "SENSORS"
    MAP = "MAP"


class AutonomousAgent(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file, device):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        # agent's initialization
        self.setup(path_to_conf_file, device)

        self.wallclock_t0 = None

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def set_environment(self, environment):
        """
        Set the environment for the agent
        :return:
        """
        self.environment = environment

    def set_evaluator(self, evaluator):
        """
        Set the evaluator for the agent
        :return:
        """
        self.evaluator = evaluator

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data()

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()

        print(
            "======[Agent] Wallclock_time = {} / {} / Sim_time = {} / {}x".format(
                wallclock,
                wallclock_diff,
                timestamp,
                timestamp / (wallclock_diff + 0.001),
            )
        )

        control = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        return control

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """

        self._global_plan = global_plan_gps
        self._global_plan_world_coord = global_plan_world_coord

        ds_ids = downsample_route(global_plan_world_coord, 20)
        self._global_plan_world_coord_downsampled = [
            (global_plan_world_coord[x][0], global_plan_world_coord[x][1])
            for x in ds_ids
        ]
        self._global_plan_downsampled = [global_plan_gps[x] for x in ds_ids]

        self.router_world_coord = Router(
            route=self._global_plan_world_coord, distance_threshold=5, closest=True
        )
        self.router_gps = Router(
            route=self._global_plan, distance_threshold=5, closest=False
        )
        self.router_world_coord_downsampled = Router(
            route=self._global_plan_world_coord_downsampled,
            distance_threshold=5,
            closest=False,
        )
        self.router_gps_downsampled = Router(
            route=self._global_plan_downsampled, distance_threshold=5, closest=False
        )
