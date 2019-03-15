from abc import ABC

import torch
from torch import Tensor
from vizdoom import DoomGame
import torchvision.transforms as T
import numpy as np

from drl.deepq.game import Game, Action, Frames, Experience


class VizdoomGame(Game, ABC):
  def __init__(self, scenario: str, x: int, y: int, t: int, visible: bool = False):
    self.game = self._setup_game(scenario, visible)
    self.frames = Frames(t)
    self.x = x
    self.y = y
    self.transform = T.Compose([T.ToPILImage(), T.Resize((y, x)), T.Grayscale(), T.ToTensor()])

  @staticmethod
  def _setup_game(scenario: str, visible: bool):
    game = DoomGame()
    game.load_config('data/vizdoom/%s.cfg' % scenario)
    game.set_doom_scenario_path('data/vizdoom/%s.wad' % scenario)
    game.set_window_visible(visible)
    game.set_sound_enabled(False)
    game.init()
    return game

  def step(self, action: Action) -> Experience:
    old_state = self.frames.state()
    reward, done = self._perform_action(action.key)
    new_state = self.frames.state()
    if done:
      self._new_episode()  # restart
    return Experience(state_before=old_state,
                      state_after=new_state,
                      action=action,
                      reward=reward,
                      done=done)

  def current_state(self):
    return self.frames.state()

  def reset(self):
    self._new_episode()
    return self.frames.state()

  def close(self):
    self.game.close()

  def _new_episode(self):
    self.game.new_episode()
    frame = self._get_frame()
    self.frames.add_initial_frame(frame)

  def _perform_action(self, action):
    reward = self.game.make_action(action)
    done = self.game.is_episode_finished()
    if done:
      frame = torch.zeros([self.x, self.y])
    else:
      frame = self._get_frame()
    self.frames.add_frame(frame)
    return reward, done

  def _get_frame(self) -> Tensor:
    raw = self.game.get_state().screen_buffer
    trans = np.transpose(raw, (1, 2, 0))
    frame = self.transform(trans).squeeze(0)
    return frame
