import numpy as np
from vizdoom import DoomGame

from drl.deepq.game import Game, Action, Frames, Experience


class VizdoomBasicGame(Game):
  """Vizdoom Basic => One room with a single non-moving opponent"""

  actions = [Action("left", [1, 0, 0], 0),
             Action("right", [0, 1, 0], 1),
             Action("fire", [0, 0, 1], 2)]

  def __init__(self, x: int, y: int, t: int):
    self.game = self._setup_game()
    self.frames = Frames(x, y, t)
    self._new_episode()
    self.frame_shape = self.game.get_state().screen_buffer.shape

  @property
  def name(self) -> str:
    return "vizdoom-basic"

  @staticmethod
  def _setup_game():
    game = DoomGame()
    game.load_config("data/vizdoom/simpler_basic.cfg")
    game.set_doom_scenario_path("data/vizdoom/simpler_basic.wad")
    game.set_window_visible(False)
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

  def _new_episode(self):
    self.game.new_episode()
    frame = self.game.get_state().screen_buffer
    self.frames.add_initial_frame(frame)

  def _perform_action(self, action):
    reward = self.game.make_action(action)
    done = self.game.is_episode_finished()
    if done:
      frame = np.zeros(self.frame_shape)
      self.frames.add_frame(frame)
      return reward, True
    else:
      frame = self.game.get_state().screen_buffer
      self.frames.add_frame(frame)
      return reward, False