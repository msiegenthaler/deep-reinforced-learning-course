import abc
from collections import deque
from typing import NamedTuple

import torch


class Action(NamedTuple):
  name: str
  key: any  # private for usage by the game
  index: int


class State(abc.ABC):
  @abc.abstractmethod
  def as_tensor(self) -> torch.Tensor:
    pass


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


class LazyFrameState(State):
  """Much more memory efficient that stacking right on return, this way each frame is only held in memory once"""

  def __init__(self, frames: [torch.Tensor]):
    self._frames = frames

  def as_tensor(self):
    return torch.stack(self._frames)


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

  def state(self) -> LazyFrameState:
    return LazyFrameState([t for t in self.deque])
