import abc
from collections import deque
from typing import NamedTuple, Callable, List, Optional

import torch


class Action(NamedTuple):
  name: str
  key: any  # private for usage by the game
  index: int


class State():
  def __init__(self, frames: List[torch.Tensor], scale_to_apply: Optional[float] = None):
    self._frame_shape = frames[0].shape
    self._scale_to_apply = scale_to_apply
    self._frames: Optional[List[torch.Tensor]] = frames
    self._on_device: Optional[torch.Tensor] = None

  def as_tensor(self, dtype: torch.dtype = torch.float, device: Optional[torch.device] = None) -> torch.Tensor:
    """Combined tensor of the complete state"""
    if self._on_device is not None:
      return self._on_device.to(dtype=dtype)
    else:
      if self._frames[0].dtype == torch.half:
        # cat/stack on cpu not supported, so move to device fist
        frames = [f.to(device, non_blocking=True) for f in self._frames]
      else:
        frames = self._frames
      tensor = torch.stack(tuple(frames))
      tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
      if self._scale_to_apply is not None:
        tensor = tensor * self._scale_to_apply
      return tensor

  def to_device(self, device, non_blocking=True):
    if self._on_device is not None:
      return self
    frames = [f.to(device, non_blocking=non_blocking) for f in self._frames]
    tensor = torch.stack(tuple(frames))
    if self._scale_to_apply is not None:
      tensor = tensor * self._scale_to_apply
    self._on_device = tensor
    self._frames = None
    return self


class Experience(NamedTuple):
  state_before: State
  action: Action
  state_after: State
  reward: float
  done: bool
  state_difference_in_steps: int = 1  # number of steps between state_before and state_after

  def to_device(self, device: torch.device, non_blocking=True):
    return self._replace(state_before=self.state_before.to_device(device, non_blocking=non_blocking),
                         state_after=self.state_after.to_device(device, non_blocking=non_blocking))


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
    return State(frames=[t for t in self.deque], scale_to_apply=scale)
