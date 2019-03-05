import abc
from collections import namedtuple, deque

import torch

Action = namedtuple("Action", "name key index")
Experience = namedtuple("Experience", "state_before action state_after reward done")
State = torch.Tensor


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
