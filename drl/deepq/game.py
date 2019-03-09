import abc
from collections import deque
from typing import NamedTuple

import torch


class Action(NamedTuple):
  name: str
  key: any  # private for usage by the game
  index: int


State = torch.Tensor


class Experience(NamedTuple):
  state_before: State
  action: Action
  state_after: State
  reward: float
  done: bool
  state_difference_in_steps: int = 1  # number of steps between state_before and state_after


class Game(abc.ABC):
  """Some kind of game that has a visual output and an action->state sequence"""

  @abc.abstractmethod
  def step(self, action: Action) -> Experience:
    pass

  @abc.abstractmethod
  def current_state(self) -> State:
    pass

  @abc.abstractmethod
  def reset(self) -> State:
    pass

  @property
  @abc.abstractmethod
  def actions(self) -> [Action]:
    pass

  @property
  @abc.abstractmethod
  def name(self) -> str:
    pass


class Frames:
  """Holds a history of the t last frames and provides them as the state"""

  def __init__(self, t: int):
    self.t = t
    self.deque = deque(maxlen=t)

  def add_initial_frame(self, frame: torch.Tensor):
    for i in range(self.t):
      self.deque.append(frame)

  def add_frame(self, frame: torch.Tensor):
    self.deque.append(frame)

  def state(self) -> torch.Tensor:
    return torch.stack([t for t in self.deque])
