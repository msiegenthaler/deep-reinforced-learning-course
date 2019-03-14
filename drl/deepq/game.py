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

  def to_device(self, device):
    frames = [f.to(device, non_blocking=True) for f in self.frames]
    return State(frames)


class Experience(NamedTuple):
  state_before: State
  action: Action
  state_after: State
  reward: float
  done: bool
  state_difference_in_steps: int = 1  # number of steps between state_before and state_after

  def to_device(self, device: torch.device):
    return self._replace(state_before=self.state_before.to_device(device),
                         state_after=self.state_after.to_device(device))


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

  def __init__(self, t: int, pin_memory=True):
    self.t = t
    self.deque = deque(maxlen=t)
    self.pin_memory = pin_memory

  def add_initial_frame(self, frame: torch.Tensor):
    frame = self._wrap_frame(frame)
    for i in range(self.t):
      self.deque.append(frame)

  def add_frame(self, frame: torch.Tensor):
    self.deque.append(self._wrap_frame(frame))

  def _wrap_frame(self, frame):
    if self.pin_memory:
      frame = frame.pin_memory()
    return frame

  def state(self) -> State:
    return State(frames=[t for t in self.deque])
