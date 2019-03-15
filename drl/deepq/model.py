from typing import NamedTuple

import torch
from torch import nn
from torch.optim import Optimizer

from drl.deepq.replay_memory import ReplayMemory
from drl.deepq.status_log import TrainingStatus


class ExecutionModel(NamedTuple):
  policy_net: nn.Module
  device: torch.device
  game_name: str
  strategy_name: str
  trained_for_epochs: int


class LearningModel(NamedTuple):
  policy_net: nn.Module
  target_net: nn.Module  # for double dq
  optimizer: Optimizer
  memory: ReplayMemory
  device: torch.device  # Device to train on
  game_name: str
  strategy_name: str
  status: TrainingStatus = TrainingStatus()

  def exec(self):
    return ExecutionModel(
      policy_net=self.policy_net,
      device=self.device,
      strategy_name=self.strategy_name,
      game_name=self.game_name,
      trained_for_epochs=self.status.trained_for_epochs
    )
