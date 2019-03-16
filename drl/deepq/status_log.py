from typing import NamedTuple, Dict, Optional

from drl.utils.stats import FloatStat
from drl.utils.timings import Timings
from tensorboardX import SummaryWriter


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
  def __init__(self, summary_writer: Optional[SummaryWriter] = None):
    self._training_epochs: [EpochTrainingLog] = []
    self._training_episodes: [EpisodeLog] = []
    self._validation_episodes: [ValidationLog] = []
    self._writer = summary_writer
    self.timings = Timings()

  def epoch_trained(self, log: EpochTrainingLog):
    self._training_epochs.append(log)
    if self._writer is not None:
      iteration = self.trained_for_steps
      self._writer.add_scalar('epoch_reward', log.episode_reward.mean, iteration)
      self._writer.add_scalar('epoch_steps', log.game_steps, iteration)
      self._writer.add_scalar('epoch_duration', log.duration_seconds, iteration)
      self._writer.add_scalar('loss', log.loss.mean, iteration)
      self._writer.add_scalar('speed', log.game_steps / log.duration_seconds, iteration)
      self._writer.add_scalar('beta', log.parameter_values['beta'], iteration)
      self._writer.add_scalar('exploration_rate', log.parameter_values['exploration_rate'], iteration)

  def episode_trained(self, log: EpisodeLog):
    self._training_episodes.append(log)
    iteration = log.at_training_step
    if self._writer is not None:
      self._writer.add_scalar('train_episode_steps', log.steps, iteration)
      self._writer.add_scalar('train_reward', log.reward, iteration)
      self._writer.add_scalar('exploration_rate', log.exploration_rate, iteration)

  def episode_validated(self, log: ValidationLog):
    self._validation_episodes.append(log)
    iteration = self.trained_for_steps
    if self._writer is not None:
      self._writer.add_scalar('validation_reward', log.episode_reward.mean, iteration)
      self._writer.add_scalar('validation_steps', log.steps, iteration)
      self._writer.add_scalar('validation_duration', log.duration_seconds, iteration)

  @property
  def validation_episodes(self):
    return self._validation_episodes

  @property
  def training_epochs(self):
    return self._training_epochs

  @property
  def training_episodes(self):
    return self._training_episodes

  @property
  def trained_for_epochs(self) -> int:
    return len(self._training_epochs)

  @property
  def trained_for_steps(self) -> int:
    return sum([l.game_steps for l in self._training_epochs])

  @property
  def trained_for_episodes(self) -> int:
    return len(self._training_episodes)
