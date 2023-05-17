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
from agents.navigation.basic_agent import BasicAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents.autonomous_agent_local import AutonomousAgent, Track

from utils.kinematic_utils import acceleration_to_throttle_brake
from utils.model_utils import convert_standard_bev_to_model_bev

from utils.model_utils import (
    fetch_checkpoint_from_wandb_run,
    fetch_run_from_wandb_link,
)
from utils.factory import *

from carla_env.bev import BirdViewProducer, BIRDVIEW_CROP_TYPE, PixelDimensions


def get_entry_point():
    return "ModelAgent"


class ModelAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    def setup(self, path_to_conf_file, device):
        """
        Setup the agent parameters
        """
        self.track = Track.MAP
        config = yaml.safe_load(open(path_to_conf_file))
        self.config = config
        self.device = device

        # ---------------------------------------------------------------------------- #
        #                       POLICY MODEL WANDB RUN CHECKPOINT                      #
        # ---------------------------------------------------------------------------- #

        self.policy_model_run = fetch_run_from_wandb_link(
            config["wandb_policy_model"]["link"]
        )
        policy_model_checkpoint_object = fetch_checkpoint_from_wandb_run(
            run=self.policy_model_run,
            checkpoint_number=config["wandb_policy_model"]["checkpoint_number"],
        )
        policy_model_checkpoint_path = policy_model_checkpoint_object.name

        # Create the model
        policy_model_class = policy_model_factory(self.policy_model_run.config)
        # Initialize the model
        self.policy_model = policy_model_class.load_model_from_wandb_run(
            config=self.policy_model_run.config["policy_model"]["config"],
            checkpoint_path=policy_model_checkpoint_path,
            device=device,
        )

        self.policy_model.eval().to(device)

        # ---------------------------------------------------------------------------- #
        #                    EGO FORWARD MODEL WANDB RUN CHECKPOINT                    #
        # ---------------------------------------------------------------------------- #

        if "wandb_ego_forward_model" not in config.keys():
            ego_forward_model_wandb_link = self.policy_model_run.config[
                "wandb_ego_forward_model"
            ]["link"]
            ego_forward_model_checkpoint_number = self.policy_model_run.config[
                "wandb_ego_forward_model"
            ]["checkpoint_number"]
        else:
            ego_forward_model_wandb_link = config["wandb_ego_forward_model"]["link"]
            ego_forward_model_checkpoint_number = config["wandb_ego_forward_model"][
                "checkpoint_number"
            ]

        ego_forward_model_run = fetch_run_from_wandb_link(ego_forward_model_wandb_link)
        ego_forward_model_checkpoint_object = fetch_checkpoint_from_wandb_run(
            run=ego_forward_model_run,
            checkpoint_number=ego_forward_model_checkpoint_number,
        )
        ego_forward_model_checkpoint_path = ego_forward_model_checkpoint_object.name

        # Create the model
        ego_forward_model_class = ego_forward_model_factory(
            ego_forward_model_run.config
        )
        # Initialize the model
        self.ego_forward_model = ego_forward_model_class.load_model_from_wandb_run(
            config=ego_forward_model_run.config["ego_forward_model"]["config"],
            checkpoint_path=ego_forward_model_checkpoint_path,
            device=device,
        )

        self.ego_forward_model.eval().to(device)
        # ---------------------------------------------------------------------------- #
        #                    WORLD FORWARD MODEL WANDB RUN CHECKPOINT                  #
        # ---------------------------------------------------------------------------- #

        if "wandb_world_forward_model" not in config.keys():
            world_forward_model_wandb_link = self.policy_model_run.config[
                "wandb_world_forward_model"
            ]["link"]
            world_forward_model_checkpoint_number = self.policy_model_run.config[
                "wandb_world_forward_model"
            ]["checkpoint_number"]
        else:
            world_forward_model_wandb_link = config["wandb_world_forward_model"]["link"]
            world_forward_model_checkpoint_number = config["wandb_world_forward_model"][
                "checkpoint_number"
            ]

        world_forward_model_run = fetch_run_from_wandb_link(
            world_forward_model_wandb_link
        )
        world_forward_model_checkpoint_object = fetch_checkpoint_from_wandb_run(
            run=world_forward_model_run,
            checkpoint_number=world_forward_model_checkpoint_number,
        )
        world_forward_model_checkpoint_path = world_forward_model_checkpoint_object.name

        # Create the model
        world_forward_model_class = world_forward_model_factory(
            world_forward_model_run.config
        )
        # Initialize the model
        self.world_forward_model = world_forward_model_class.load_model_from_wandb_run(
            config=world_forward_model_run.config["world_forward_model"]["config"],
            checkpoint_path=world_forward_model_checkpoint_path,
            device=device,
        )

        self.world_forward_model.eval().to(device)

        # ---------------------------------------------------------------------------- #
        #                                   COST                                       #
        # ---------------------------------------------------------------------------- #

        cost_class = cost_factory(self.policy_model_run.config)
        cost = cost_class(device, self.policy_model_run.config["cost"]["config"])

        self.world_previous_bev_deque = deque(maxlen=20)
        self.counter = 0

        # ------------------------------- BEV Handling ------------------------------- #

        if "bev_agent_channel" in config["evaluator"]:
            self.bev_agent_channel = config["evaluator"]["bev_agent_channel"]
        else:
            self.bev_agent_channel = self.policy_model_run.config["bev_agent_channel"]

        if "bev_vehicle_channel" in config["evaluator"]:
            self.bev_vehicle_channel = config["evaluator"]["bev_vehicle_channel"]
        else:
            self.bev_vehicle_channel = self.policy_model_run.config[
                "bev_vehicle_channel"
            ]

        if "bev_selected_channels" in config["evaluator"]:
            self.bev_selected_channels = config["evaluator"]["bev_selected_channels"]
        else:
            self.bev_selected_channels = self.policy_model_run.config[
                "bev_selected_channels"
            ]

        if "bev_calculate_offroad" in config["evaluator"]:
            self.bev_calculate_offroad = config["evaluator"]["bev_calculate_offroad"]
        else:
            self.bev_calculate_offroad = self.policy_model_run.config[
                "bev_calculate_offroad"
            ]

    def configure_bev_module(self):
        # ---------------------------------------------------------------------------- #
        #                                      BEV                                     #
        # ---------------------------------------------------------------------------- #

        self.bev_module = BirdViewProducer(
            client=CarlaDataProvider.get_client(),
            target_size=PixelDimensions(
                self.config["bev"]["config"]["width"],
                self.config["bev"]["config"]["height"],
            ),
            render_lanes_on_junctions=self.config["bev"]["config"][
                "render_lanes_on_junctions"
            ],
            pixels_per_meter=self.config["bev"]["config"]["pixels_per_meter"],
            crop_type=BIRDVIEW_CROP_TYPE[self.config["bev"]["config"]["crop_type"]],
            road_on_off=self.config["bev"]["config"]["road_on_off"],
            road_light=self.config["bev"]["config"]["road_light"],
            light_circle=self.config["bev"]["config"]["light_circle"],
            lane_marking_thickness=self.config["bev"]["config"][
                "lane_marking_thickness"
            ],
        )

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

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        occupancy = torch.zeros((1, 8), dtype=torch.float32, device=self.device)
        for i in range(8):
            radar_data = input_data[f"radar_{i}"][1][:, 0]
            occupancy[0, i] = float(
                np.nanmin(radar_data) if radar_data.shape[0] > 0 else 10
            )

        occupancy[occupancy > 10] = 10

        # Get the ego vehicle
        hero_actor = CarlaDataProvider.get_hero_actor()
        self.environment.set_hero_actor(hero_actor)

        hero_actor_location = hero_actor.get_location()
        hero_actor_rotation = hero_actor.get_transform().rotation
        hero_actor_velocity = hero_actor.get_velocity()
        hero_actor_speed = hero_actor_velocity.length()

        # BEV image
        bev_image = self.bev_module.step(
            hero_actor,
        )

        # Navigation
        # next_waypoint, next_command = self.router_world_coord.step(hero_actor_location)
        next_waypoint, next_command = self.router_world_coord_downsampled.step(
            hero_actor_location
        )

        ego_previous_location = torch.zeros((1, 2), device=self.device)
        ego_previous_yaw = torch.zeros((1, 1), device=self.device)
        ego_previous_speed = torch.zeros((1, 1), device=self.device)

        ego_previous_location[..., 0] = hero_actor_location.x
        ego_previous_location[..., 1] = hero_actor_location.y
        ego_previous_yaw[..., 0] = hero_actor_rotation.yaw
        ego_previous_speed[..., 0] = hero_actor_speed

        ego_previous_location.requires_grad_(True)
        ego_previous_yaw.requires_grad_(True)
        ego_previous_speed.requires_grad_(True)

        ego_previous = {
            "location": ego_previous_location,
            "yaw": ego_previous_yaw,
            "speed": ego_previous_speed,
        }

        target_location = torch.zeros((1, 2), device=self.device)
        target_location[..., 0] = next_waypoint.location.x
        target_location[..., 1] = next_waypoint.location.y
        target_location.requires_grad_(True)

        navigational_command = torch.zeros((1,), device=self.device)
        navigational_command[..., 0] = next_command.value - 1
        navigational_command = torch.nn.functional.one_hot(
            navigational_command.long(), num_classes=self.policy_model.command_size
        ).float()

        bev_tensor = convert_standard_bev_to_model_bev(
            bev_image,
            agent_channel=self.bev_agent_channel,
            vehicle_channel=self.bev_vehicle_channel,
            selected_channels=self.bev_selected_channels,
            calculate_offroad=self.bev_calculate_offroad,
            device=self.device,
        )
        bev_tensor.requires_grad_(True)
        if self.counter == 0:
            for _ in range(self.policy_model_run.config["num_time_step_previous"]):
                self.world_previous_bev_deque.append(bev_tensor)

        else:
            self.world_previous_bev_deque.append(bev_tensor)

        # Feed previous bev to world model
        world_previous_bev = torch.stack(list(self.world_previous_bev_deque), dim=1).to(
            self.device
        )

        # Get encoded states
        (
            world_previous_bev_encoded,
            world_future_bev,
        ) = self.world_forward_model(world_previous_bev, sample_latent=True)

        # Feed to policy model
        # Policy model
        action = self.policy_model(
            ego_previous,
            world_previous_bev_encoded,
            navigational_command,
            target_location,
            occupancy,
        )

        control_selected = action[0]

        # Convert to environment control
        acceleration = control_selected[0].item()
        steer = control_selected[1].item()

        throttle, brake = acceleration_to_throttle_brake(
            acceleration=acceleration,
        )

        print("Throttle: ", throttle, "Steer: ", steer, "Brake: ", brake, "")

        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        control.hand_brake = False

        self.counter += 1

        self.environment.render(counter=self.counter)

        return control
