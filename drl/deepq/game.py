import abc
from collections import deque
from typing import NamedTuple, Callable, List

import torch


class Action(NamedTuple):
  name: str
  key: any  # private for usage by the game
  index: int


class State(NamedTuple):
  """
  Much more memory efficient than stacking upfront.
  This way each frame is only held in memory once but used in mutiple states
  """
  frames: List[torch.Tensor]

  def as_tensor(self) -> torch.Tensor:
    """Combined tensor of the complete state"""
    return torch.stack(tuple(self.frames))


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

  @abc.abstractmethod
  def close(self) -> None:
    pass

  @property
  @abc.abstractmethod
  def actions(self) -> [Action]:
    pass

  @property
  @abc.abstractmethod
  def name(self) -> str:
    pass

  def __enter__(self):
    self.reset()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()


GameFactory = Callable[[], Game]


class Frames:
  """Holds a history of the t last frames and provides them as the state"""

  def __init__(self, t: int, device: torch.device):
    self.t = t
    self.deque = deque(maxlen=t)
    self.device = device

  def add_initial_frame(self, frame: torch.Tensor):
    frame = frame.to(self.device, non_blocking=True)
    for i in range(self.t):
      self.deque.append(frame)

  def add_frame(self, frame: torch.Tensor):
    frame = frame.to(self.device, non_blocking=True)
    self.deque.append(frame)

  def state(self) -> State:
    return State(frames=[t for t in self.deque])
