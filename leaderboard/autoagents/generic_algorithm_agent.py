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


def get_entry_point():
    return "GenericAlgorithmAgent"


class GenericAlgorithmAgent(AutonomousAgent):

    """
    Generic algorithm agent (MPC) to control the ego vehicle
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

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:
        """

        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        # Get the ego vehicle
        hero_actor = CarlaDataProvider.get_hero_actor()

        if self.counter == 0:
            self._initial_configuration(hero_actor)

        hero_actor_location = hero_actor.get_location()

        # Navigation

        next_waypoint_dense, next_waypoint, next_command = self._step_routes(
            hero_actor_location
        )

        # BEV image
        bev_image = self.bev_module["module"].step(
            hero_actor,
            waypoint=next_waypoint_dense,
        )

        self.environment.step(next_waypoint=next_waypoint, next_command=next_command)

        ego_previous, bev_tensor, target, situation = self._process_data(
            input_data, hero_actor, next_waypoint, bev_image
        )

        self._populate_bev_deque(bev_tensor)

        # Covnert BEV deque to bev tensor
        world_previous_bev = self._convert_deque_to_tensor()

        # It is allowed to calculate a new action
        self._step_algorithm(ego_previous, target, world_previous_bev, situation)

        # Fetch predicted action
        mpc_control_selected = self._select_control()

        control_selected = self._fuse_adapter(mpc_control_selected)

        # Convert to environment control
        control = self._convert_to_carla_control(control_selected)

        self._render(
            next_waypoint,
            next_command,
            situation,
            bev_image,
            mpc_control_selected,
            control_selected,
        )

        self._step_adapter()

        self._increment_counter()

        return control

    def _process_data(self, input_data, hero_actor, next_waypoint, bev_image):
        processed_data = self.evaluator.process_data(
            hero_actor, input_data, bev_image, next_waypoint
        )

        ego_previous = processed_data["ego_previous"]
        bev_tensor = processed_data["bev_tensor"]
        target = processed_data["target"]
        situation = (
            processed_data["situation"] if "situation" in processed_data else None
        )
        return ego_previous, bev_tensor, target, situation

    def _increment_counter(self):
        self.counter += 1

        # Update counters
        self.evaluator.frame_counter += 1
        self.evaluator.skip_counter = (
            self.evaluator.skip_counter
            + (self.evaluator.repeat_counter + 1 == (self.evaluator.repeat_frames))
        ) % self.evaluator.skip_frames
        self.evaluator.repeat_counter = (
            self.evaluator.repeat_counter + 1
        ) % self.evaluator.repeat_frames

    def _step_adapter(self):
        if self.evaluator.adapter is not None:
            self.initial_guess = (
                torch.from_numpy(
                    self.evaluator.adapter.step(
                        self.router_world_coord.get_remaining_route()
                    )
                )
                .view(1, 1, self.evaluator.action_size)
                .to(self.evaluator.device)
            )  # Shape: (1,2)

            self.evaluator.reset(
                initial_guess=self.initial_guess.repeat(
                    1, self.evaluator.num_time_step_future, 1
                )
            )

        else:
            self.evaluator.reset()

    def _render(
        self,
        next_waypoint,
        next_command,
        situation,
        bev_image,
        mpc_control_selected,
        control_selected,
    ):
        self.environment.render(
            bev_world=bev_image,
            adapter_render=self.evaluator.adapter.render()
            if (self.evaluator.adapter is not None) and (self.counter > 1)
            else None,
            frame_counter=self.evaluator.frame_counter,
            skip_counter=self.evaluator.skip_counter,
            repeat_counter=self.evaluator.repeat_counter,
            route_progress=f"{self.router_world_coord_downsampled.route_index} / {self.router_world_coord_downsampled.route_length}",
            next_waypoint=next_waypoint.transform,
            next_command=next_command,
            situation=situation,
            adapter_action=self.initial_guess
            if (self.evaluator.adapter is not None) and (self.counter > 1)
            else None,
            mpc_action=mpc_control_selected,
            action=control_selected,
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

    def _convert_to_carla_control(self, control_selected):
        acceleration = control_selected[0].item()
        steer = control_selected[1].item()

        throttle, brake = acceleration_to_throttle_brake(
            acceleration=acceleration,
        )

        # if (self.counter % 100) <= 3:
        #     throttle = 1.0
        env_control = [throttle, steer, brake]

        print(f"Counter: {self.counter}")
        print(f"Throttle: {throttle}, steer: {steer}, brake: {brake}")

        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        control.hand_brake = False
        return control

    def _fuse_adapter(self, mpc_control_selected):
        if (self.evaluator.adapter is not None) and (self.initial_guess is not None):
            control_selected = self.initial_guess * self.evaluator.adapter_weight + (
                mpc_control_selected * self.evaluator.mpc_weight
            )

        else:
            control_selected = mpc_control_selected

        control_selected = control_selected.view(self.evaluator.action_size)
        return control_selected

    def _select_control(self):
        mpc_control_selected = self.ego_future_action_predicted[0][
            self.evaluator.skip_counter
        ].view(1, 1, self.evaluator.action_size)

        return mpc_control_selected

    def _step_algorithm(self, ego_previous, target, world_previous_bev, situation):
        if (self.evaluator.skip_counter == 0) and (self.evaluator.repeat_counter == 0):
            out = self.evaluator.step(
                ego_previous=ego_previous,
                world_previous_bev=world_previous_bev,
                target=target,
                situation=situation,
            )
            self.ego_future_action_predicted = out["action"]
            self.world_future_bev_predicted = out["world_future_bev_predicted"]
            self.mask_dict = out["cost"]["mask_dict"]
            self.ego_future_location_predicted = out["ego_future_location_predicted"]
            self.cost = out["cost"]

    def _populate_bev_deque(self, bev_tensor):
        if self.counter == 0:
            for _ in range(self.evaluator.num_time_step_previous):
                self.world_previous_bev_deque.append(bev_tensor)

        else:
            self.world_previous_bev_deque.append(bev_tensor)

    def _convert_deque_to_tensor(self):
        world_previous_bev = (
            torch.stack(list(self.world_previous_bev_deque), dim=1)
            .to(self.device)
            .requires_grad_(True)
        )

        return world_previous_bev

    def _step_routes(self, hero_actor_location):
        next_waypoint_dense, next_command_dense = self.router_world_coord.step(
            hero_actor_location
        )
        next_waypoint, next_command = self.router_world_coord_downsampled.step(
            hero_actor_location
        )

        next_waypoint = CarlaDataProvider.get_map().get_waypoint(next_waypoint.location)
        next_waypoint_dense = CarlaDataProvider.get_map().get_waypoint(
            next_waypoint_dense.location
        )

        return next_waypoint_dense, next_waypoint, next_command

    def _initial_configuration(self, hero_actor):
        self.environment.set_hero_actor(hero_actor)
        self.environment.set_sensor_and_bev()
        self.bev_module = self.environment.get_bev_modules()[0]
        self.world_previous_bev_deque = deque(
            maxlen=self.evaluator.num_time_step_previous
        )
        if self.evaluator.adapter is not None:
            self.evaluator.adapter.reset(
                vehicle=self.environment.get_hero_actor(),
                global_plan_world_coord=self._global_plan_world_coord,
                global_plan_world_coord_downsampled=self._global_plan_world_coord_downsampled,
                global_plan_gps=self._global_plan_downsampled,
                global_plan_gps_downsampled=self._global_plan_downsampled,
            )

            self.initial_guess = None
            # torch.from_numpy(
            #     self.evaluator.adapter.step(
            #         self.router_world_coord.get_remaining_route()
            #     )
            # )

        CarlaDataProvider.get_world().tick()
