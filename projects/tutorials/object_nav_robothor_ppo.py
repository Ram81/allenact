from math import ceil
from typing import Dict, Any, List, Optional

import ai2thor
import os
import glob
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from allenact_plugins.robothor_plugin.robothor_task_samplers import ObjectNavDatasetTaskSampler
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from allenact_plugins.navigation_plugin.objectnav.models import ObjectNavActorCritic


class ObjectNavRoboThorPPOExperimentConfig(ExperimentConfig):
    """A simple object navigation experiment in THOR.

    Training with PPO.
    """

    # A simple setting, train/valid/test are all the same single scene
    # and we're looking for a single object
    OBJECT_TYPES = ["Tomato"]
    TRAIN_SCENES = ["FloorPlan13_physics"]
    VALID_SCENES = ["FloorPlan13_physics"]
    TEST_SCENES = ["FloorPlan13_physics"]

    # Dataset Parameters
    TRAIN_DATASET_DIR = "datasets/robothor-objectnav/train"
    VAL_DATASET_DIR = "datasets/robothor-objectnav/val"

    # Setting up sensors and basic environment details
    SCREEN_SIZE = 224
    SENSORS = [
        RGBSensorThor(
            height=SCREEN_SIZE, width=SCREEN_SIZE, use_resnet_normalization=True,
        ),
        GoalObjectTypeThorSensor(object_types=OBJECT_TYPES),
    ]

    ENV_ARGS = {
        "player_screen_height": SCREEN_SIZE,
        "player_screen_width": SCREEN_SIZE,
        "quality": "Very Low",
        #"platform": ai2thor.platform.CloudRendering
    }

    REWARD_CONFIG = {
        "step_penalty": -0.01,
        "goal_success_reward": 10.0,
        "failed_stop_reward": 0.0,
        "shaping_weight": 1.0,
    }

    MAX_STEPS = 128
    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    VALID_SAMPLES_IN_SCENE = 10
    TEST_SAMPLES_IN_SCENE = 10

    @classmethod
    def tag(cls):
        return "ObjectNavRoboThorPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(1e6)
        lr = 2.5e-4
        num_mini_batch = 2 if not torch.cuda.is_available() else 6
        update_repeats = 4
        num_steps = 128
        metric_accumulate_interval = cls.MAX_STEPS * 10  # Log every 10 max length tasks
        save_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={
                "ppo_loss": PPO(clip_decay=LinearDecay(ppo_steps), **PPOConfig),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps,),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs):
        num_gpus = torch.cuda.device_count()
        has_gpu = num_gpus != 0

        if mode == "train":
            nprocesses = 4 if has_gpu else 4
            gpu_ids = [0] if has_gpu else []
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [1 % num_gpus] if has_gpu else []
        elif mode == "test":
            nprocesses = 1
            gpu_ids = [0] if has_gpu else []
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        return MachineParams(nprocesses=nprocesses, devices=gpu_ids,)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjectNavActorCritic(
            action_space=gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            rgb_uuid=cls.SENSORS[0].uuid,
            depth_uuid=None,
            goal_sensor_uuid="goal_object_type_ind",
            hidden_size=512,
            object_type_embedding_dim=8,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavDatasetTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes_dir: str,
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        path = os.path.join(scenes_dir, "*.json.gz")
        scenes = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(path)]
        if len(scenes) == 0:
            raise RuntimeError(
                (
                    "Could find no scene dataset information in directory {}."
                    " Are you sure you've downloaded them? "
                    " If not, see https://allenact.org/installation/download-datasets/ information"
                    " on how this can be done."
                ).format(scenes_dir)
            )
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if total_processes % len(scenes) != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisible by the number of scenes"
                )
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        else:
            if len(scenes) % total_processes != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisor of the number of scenes"
                )
        inds = self._partition_inds(len(scenes), total_processes)
        print(scenes_dir, total_processes, len(scenes))

        return {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            #"object_types": self.OBJECT_TYPES,
            "env_args": self.ENV_ARGS,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            os.path.join(self.TRAIN_DATASET_DIR, "episodes"),
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        #res["scene_period"] = "manual"
        res["scene_directory"] = self.TRAIN_DATASET_DIR
        res["loop_dataset"] = False
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            os.path.join(self.VAL_DATASET_DIR, "episodes"),
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        # res["scene_period"] = self.VALID_SAMPLES_IN_SCENE
        # res["max_tasks"] = self.VALID_SAMPLES_IN_SCENE * len(res["scenes"])
        res["scene_directory"] = self.VAL_DATASET_DIR
        res["loop_dataset"] = True
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        res["allow_flipping"] = True
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            os.path.join(self.VAL_DATASET_DIR, "episodes"),
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        # res["scene_period"] = self.TEST_SAMPLES_IN_SCENE
        # res["max_tasks"] = self.TEST_SAMPLES_IN_SCENE * len(res["scenes"])
        res["scene_directory"] = self.VAL_DATASET_DIR
        res["loop_dataset"] = True
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res
