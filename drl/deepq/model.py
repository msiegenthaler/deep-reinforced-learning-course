from typing import NamedTuple, Dict

from torch import nn
from torch.optim import Optimizer

from drl.deepq.game import Game
from drl.deepq.replay_memory import ReplayMemory
from drl.utils.stats import FloatStat
from drl.utils.timings import Timings


class ValidationLog(NamedTuple):
  at_training_epoch: int
  episodes: int
  steps: int
  duration_seconds: float
  episode_reward: FloatStat
  actions_taken: Dict[str, float]


class EpochTrainingLog(NamedTuple):
  episodes: int
  """how many times the net was updated (trained)"""
  trainings: int
  game_steps: int
  parameter_values: dict
  loss: FloatStat
  episode_reward: FloatStat
  duration_seconds: float


class TrainingStatus:
  def __init__(self):
    self.training_log: [EpochTrainingLog] = []
    self.validation_log: [ValidationLog] = []
    self.timings = Timings()

  @property
  def trained_for_epochs(self) -> int:
    return len(self.training_log)

  @property
  def trained_for_steps(self) -> int:
    return sum([l.game_steps for l in self.training_log])

  @property
  def trained_for_episodes(self) -> int:
    return sum([l.episodes for l in self.training_log])


class ExecutionModel(NamedTuple):
  policy_net: nn.Module
  game: Game
  device: object
  strategy_name: str
  trained_for_epochs: int


class LearningModel(NamedTuple):
  policy_net: nn.Module
  target_net: nn.Module  # for double dq
  optimizer: Optimizer
  game: Game
  memory: ReplayMemory
  device: object  # Device to train on
  strategy_name: str
  status: TrainingStatus = TrainingStatus()

  def exec(self):
    return ExecutionModel(
      policy_net=self.policy_net,
      game=self.game,
      device=self.device,
      strategy_name=self.strategy_name,
      trained_for_epochs=self.status.trained_for_epochs
    )
