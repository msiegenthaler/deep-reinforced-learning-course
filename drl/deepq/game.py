import abc
from collections import deque
from typing import NamedTuple, Callable, List, Optional

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
  scale: Optional[float]
  on_device: Optional[torch.Tensor] = None

  def as_tensor(self, dtype: torch.dtype = torch.float, device: Optional[torch.device] = None) -> torch.Tensor:
    """Combined tensor of the complete state"""
    if self.on_device is not None:
      tensor = self.on_device.reshape(-1, *self.frames[0].shape)
    else:
      tensor = torch.stack(tuple(self.frames))
    tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
    if self.scale is not None:
      tensor = tensor * self.scale
    return tensor

  def to_device(self, device):
    if self.on_device:
      return
    frames = [f.to(device, non_blocking=True) for f in self.frames]
    t = torch.cat(tuple(frames)).to(device, non_blocking=True)
    return State(self.frames, self.scale, t)


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

  def __init__(self, t: int, dtype: torch.dtype, scale: Optional[float] = None, pin_memory=True):
    self.t = t
    self.deque = deque(maxlen=t)
    self.pin_memory = pin_memory
    self.dtype = dtype
    self.scale = scale

  def add_initial_frame(self, frame: torch.Tensor):
    frame = self._wrap_frame(frame)
    for i in range(self.t):
      self.deque.append(frame)

  def add_frame(self, frame: torch.Tensor):
    self.deque.append(self._wrap_frame(frame))

  def _wrap_frame(self, frame):
    if self.pin_memory:
      frame = frame.pin_memory()
    if self.scale is not None:
      frame = frame * self.scale
    if self.dtype:
      frame = frame.type(self.dtype)
    return frame

  def state(self) -> State:
    scale = 1. / self.scale if self.scale is not None else None
    return State(frames=[t for t in self.deque], scale=scale)
