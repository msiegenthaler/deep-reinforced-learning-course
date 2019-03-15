from typing import NamedTuple, Dict

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


class EpisodeLog(NamedTuple):
  at_training_epoch: int
  reward: float
  steps: int
  exploration_rate: float
  at_training_step: int = 0


class TrainingStatus:
  def __init__(self):
    self.training_log: [EpochTrainingLog] = []
    self.training_episodes: [EpisodeLog] = []
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
    return len(self.training_episodes)
