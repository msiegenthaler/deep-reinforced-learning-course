import abc
from abc import ABC

import gym
import numpy as np
from torch import Tensor
import torchvision.transforms as T

from drl.deepq.game import Game, Action, State, Experience, Frames


class OpenAIGame(Game, ABC):
  """OpenAI Gym game"""

  def __init__(self, scenario: str, t: int):
    self.env = gym.make(scenario)
    self.frames = Frames(t)
    self._new_episode()

  def reset(self) -> State:
    self._new_episode()
    return self.frames.state()

  def _new_episode(self):
    env_state = self.env.reset()
    frame = self._get_frame(env_state)
    self.frames.add_initial_frame(frame)

  @abc.abstractmethod
  def _get_frame(self, env_state) -> Tensor:
    pass

  def step(self, action: Action):
    old_state = self.frames.state()
    env_state, reward, terminal, _ = self.env.step(action.key)
    frame = self._get_frame(env_state)
    self.frames.add_frame(frame)
    if terminal:
      self._new_episode()
    return Experience(state_before=old_state,
                      state_after=self.frames.state(),
                      action=action,
                      reward=reward,
                      done=terminal)

  def current_state(self) -> State:
    return self.frames.state()


class OpenAIFrameGame(OpenAIGame, ABC):
  """OpenAI Gym game where we just look at the rendered output, not at the state"""

  def __init__(self, scenario: str, t: int):
    super().__init__(scenario, t)

  def _get_raw_frame(self):
    return self.env.render(mode='rgb_array')


class CartPoleVisual(OpenAIFrameGame):
  actions = [Action('left', 0, 0),
             Action('right', 1, 1)]

  def __init__(self, x: int, y: int, t: int):
    self.transform = T.Compose([T.ToPILImage(), T.Resize((y, x)), T.Grayscale(), T.ToTensor()])
    super().__init__('CartPole-v0', t)

  @property
  def name(self) -> str:
    return 'cardpole'

  def _get_frame(self, env_state) -> Tensor:
    image = self.transform(self._get_raw_frame()[330:660, 0:1200, :])
    return image.squeeze(0)


class Pong(OpenAIGame):
  actions = [Action('up', 2, 0),
             Action('down', 3, 1)]

  def __init__(self, x: int, y: int, t: int):
    super().__init__('Pong-v0', x, y, t)
    self.x = x
    self.y = y

  @property
  def name(self) -> str:
    return 'pong'

  def _get_frame(self, env_state):
    return np.dot(env_state[..., :3], [0.299, 0.587, 0.114]) * 255
