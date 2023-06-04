#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function
from collections import deque
import yaml
import carla
import torch
import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents.autonomous_agent_local import AutonomousAgent, Track

from utilities.kinematic_utils import acceleration_to_throttle_brake


from utilities.factory import *

from carla_env.bev import BirdViewProducer, BIRDVIEW_CROP_TYPE, PixelDimensions


def get_entry_point():
    return "MPCAgent"


class MPCAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    def setup(self, path_to_conf_file, device):
        """
        Setup the agent parameters
        """
        self.track = Track.MAP
        if path_to_conf_file is not None:
            config = yaml.safe_load(open(path_to_conf_file))
            self.config = config
        self.device = device
        self.counter = 0

        self.stuck_counter = 0
        self.stuck_counter_threshold = 200
        self.stuck_speed_threshold = 0.1
        self.stuck_avoidance_action_counter = 0
        self.stuck_avoidance_action_counter_threshold = 20

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:
        """

        sensors = [
            {
                "type": "sensor.other.radar",
                "range": 10,
                "id": "radar_0",
                "x": 3.3397709080770213,
                "y": 0.0,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "fov": 5,
            },
            {
                "type": "sensor.other.radar",
                "range": 10,
                "id": "radar_1",
                "x": 2.3615746567108156,
                "y": 2.361574656710815,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 45.0,
                "fov": 5,
            },
            {
                "type": "sensor.other.radar",
                "range": 10,
                "id": "radar_2",
                "x": 0.0,
                "y": 3.3397709080770213,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 90.0,
                "fov": 5,
            },
            {
                "type": "sensor.other.radar",
                "range": 10,
                "id": "radar_3",
                "x": -2.3615746567108156,
                "y": 2.361574656710815,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 135.0,
                "fov": 5,
            },
            {
                "type": "sensor.other.radar",
                "range": 10,
                "id": "radar_4",
                "x": -3.3397709080770213,
                "y": 0.0,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 180.0,
                "fov": 5,
            },
            {
                "type": "sensor.other.radar",
                "range": 10,
                "id": "radar_5",
                "x": -2.3615746567108156,
                "y": -2.361574656710815,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 225.0,
                "fov": 5,
            },
            {
                "type": "sensor.other.radar",
                "range": 10,
                "id": "radar_6",
                "x": 0.0,
                "y": -3.3397709080770213,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 270.0,
                "fov": 5,
            },
            {
                "type": "sensor.other.radar",
                "range": 10,
                "id": "radar_7",
                "x": 2.3615746567108156,
                "y": -2.361574656710815,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 315.0,
                "fov": 5,
            },
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        # Get the ego vehicle
        hero_actor = CarlaDataProvider.get_hero_actor()
        if self.counter == 0:
            self.environment.set_hero_actor(hero_actor)
            self.environment.set_sensor_and_bev()
            self.bev_module = self.environment.get_bev_modules()[0]
            self.world_previous_bev_deque = deque(
                maxlen=self.evaluator.num_time_step_previous
            )
            if self.evaluator.adapter is not None:
                self.evaluator.adapter.reset(
                    self.environment.get_hero_actor(), self._global_plan_world_coord
                )

                self.initial_guess = torch.from_numpy(self.evaluator.adapter.step())

        self.environment.step()

        hero_actor_location = hero_actor.get_location()
        hero_actor_speed = hero_actor.get_velocity().length()

        # Navigation
        # next_waypoint, next_command = self.router_world_coord.step(hero_actor_location)
        next_waypoint_dense, _ = self.router_world_coord.step(hero_actor_location)
        next_waypoint, next_command = self.router_world_coord_downsampled.step(
            hero_actor_location
        )

        # BEV image
        bev_image = self.bev_module["module"].step(
            hero_actor,
            waypoint=CarlaDataProvider.get_map().get_waypoint(
                next_waypoint_dense.location
            ),
        )

        processed_data = self.evaluator.process_data(
            hero_actor, input_data, bev_image, next_waypoint
        )

        if hero_actor_speed < self.stuck_speed_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.stuck_avoidance_action_counter = 0

        print("Stuck Counter: ", self.stuck_counter)
        print("Stuck Avoidance Action Counter: ", self.stuck_avoidance_action_counter)

        if self.stuck_counter > self.stuck_counter_threshold:
            if (
                self.stuck_avoidance_action_counter
                < self.stuck_avoidance_action_counter_threshold
            ):
                control = carla.VehicleControl()
                control.steer = 0.0

                if processed_data["occupancy"][0, 0] > 0.5:
                    control.throttle = 1.0
                    control.brake = 0.0
                    print("Stuck avoidance action: Throttle")
                else:
                    control.throttle = 0.0
                    control.brake = 1.0
                    print("Stuck avoidance action: Brake")

                self.stuck_avoidance_action_counter += 1

                return control

            else:
                self.stuck_avoidance_action_counter = 0
                self.stuck_counter = 0

        ego_previous = processed_data["ego_previous"]
        bev_tensor = processed_data["bev_tensor"]
        target = processed_data["target"]

        if self.counter == 0:
            for _ in range(self.evaluator.num_time_step_previous):
                self.world_previous_bev_deque.append(bev_tensor)

        else:
            self.world_previous_bev_deque.append(bev_tensor)

        # Feed previous bev to world model
        world_previous_bev = (
            torch.stack(list(self.world_previous_bev_deque), dim=1)
            .to(self.device)
            .requires_grad_(True)
        )

        # It is allowed to calculate a new action
        if (self.evaluator.skip_counter == 0) and (self.evaluator.repeat_counter == 0):
            out = self.evaluator.step(
                ego_previous=ego_previous,
                world_previous_bev=world_previous_bev,
                target=target,
            )
            self.ego_future_action_predicted = out["action"]
            self.world_future_bev_predicted = out["world_future_bev_predicted"]
            self.mask_dict = out["cost"]["mask_dict"]
            self.ego_future_location_predicted = out["ego_future_location_predicted"]
            self.cost = out["cost"]

        # Fetch predicted action
        control_selected = self.ego_future_action_predicted[0][
            self.evaluator.skip_counter
        ]

        # Convert to environment control
        acceleration = control_selected[0].item()
        steer = control_selected[1].item()

        throttle, brake = acceleration_to_throttle_brake(
            acceleration=acceleration,
        )

        if (self.counter % 100) <= 3:
            throttle = 1.0
        env_control = [throttle, steer, brake]

        print(f"Counter: {self.counter}")
        print(f"Throttle: {throttle}, steer: {steer}, brake: {brake}")

        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        control.hand_brake = False

        self.counter += 1

        self.environment.render(
            bev_world=bev_image,
            frame_counter=self.evaluator.frame_counter,
            skip_counter=self.evaluator.skip_counter,
            repeat_counter=self.evaluator.repeat_counter,
            route_progress=f"{self.router_world_coord_downsampled.route_index} / {self.router_world_coord_downsampled.route_length}",
            adapter_action=self.initial_guess[0] if self.evaluator.adapter is not None else None,
            mpc_action=env_control,
            **self.cost,
            cost_viz={  # Some dummy arguments for visualization
                "world_future_bev_predicted": self.world_future_bev_predicted,
                "mask_dict": self.mask_dict,
                "bev_selected_channels": self.evaluator.bev_selected_channels,
                "bev_calculate_offroad": self.evaluator.bev_calculate_offroad,
            },  # It looks like there is not any other way
            ego_viz={
                "ego_future_location_predicted": self.ego_future_location_predicted,
                "control_selected": control_selected,
            },
            route_viz={
                "next_waypoint": next_waypoint,
            },
        )

        if self.evaluator.adapter is not None:
            self.initial_guess = torch.from_numpy(self.evaluator.adapter.step())  # Shape: (1,2)

            self.evaluator.reset(
                initial_guess=self.initial_guess
                .unsqueeze(1)
                .repeat((1, self.evaluator.num_time_step_future, 1))
            )

        else:
            self.evaluator.reset()

        # Update counters
        self.evaluator.frame_counter += 1
        self.evaluator.skip_counter = (
            self.evaluator.skip_counter
            + (self.evaluator.repeat_counter + 1 == (self.evaluator.repeat_frames))
        ) % self.evaluator.skip_frames
        self.evaluator.repeat_counter = (
            self.evaluator.repeat_counter + 1
        ) % self.evaluator.repeat_frames

        return control
