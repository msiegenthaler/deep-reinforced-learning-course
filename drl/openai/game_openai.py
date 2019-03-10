import abc
from abc import ABC

import gym
from torch import Tensor

from drl.deepq.game import Game, Action, State, Experience, Frames


class OpenAIGame(Game, ABC):
  """OpenAI Gym game"""

  def __init__(self, scenario: str, t: int):
    self.env = gym.make(scenario)
    self.frames = Frames(t)

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
    env_state, reward, terminal, info = self.env.step(action.key)
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
