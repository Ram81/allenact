#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import gym
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from gym.spaces.dict import Dict as SpaceDict


class TaskController:
    """Abstract class defining a task parameter controller.

    When defining a new task controller, you should subclass this class and implement the abstract methods.

    # Attributes

    action_space : The space of actions available to the agent. This is of type `gym.spaces.Space`.
    observation_space: The observation space expected by the agent. This is of type `gym.spaces.dict`.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def next_task_parameters(
        self,
        observations: ObservationType,
        total_steps: int,
    ):
        raise NotImplementedError()
