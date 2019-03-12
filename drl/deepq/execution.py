import random
from time import time
from typing import Dict, NamedTuple, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor

from drl.deepq.game import Game, Experience
from drl.deepq.model import ExecutionModel, ValidationLog
from drl.deepq.multistep import create_experience_buffer
from drl.utils.stats import FloatStatCollector
from drl.utils.timings import Timings


def best_action(device, policy_net: nn.Module, state: Tensor) -> int:
  """
  :returns index of the chosen action
  """
  with torch.no_grad():
    s = state.unsqueeze(0).to(device)
    # t.max(1) will return largest column value of each row.
    # second column on max result is index of where max element was
    # found, so we pick action with the larger expected reward.
    qs = policy_net(s)
    index = qs.max(1)[1].view(1, 1).item()
    return index


class ChosenAction(NamedTuple):
  action_index: int
  is_best: bool


def chose_action(network: nn.Module, device: torch.device, state: Tensor, exploration_rate: float) -> ChosenAction:
  """
  :returns (index of the chosen action, whether the action is 'best' (True) or random (False))
  """
  if random.random() < exploration_rate:
    return ChosenAction(random.randrange(network.action_count), False)
  action = best_action(device, network, state)
  return ChosenAction(action, True)


class EpisodeCompleted(NamedTuple):
  steps: int
  reward: float


class GameExecutor:
  def __init__(self, game: Game, timings: Timings, multi_step_n: int, gamma: float):
    self.game = game
    self.timings = timings
    self.experience_buffer = create_experience_buffer(multi_step_n, gamma)
    self.episode_steps = 0
    self.episode_reward = 0.
    self.state = self.game.reset().as_tensor()

  def step(self, network: nn.Module, device: torch.device,
           exploration_rate: float) -> (Optional[EpisodeCompleted], [Experience]):
    with self.timings['forward action']:
      action = chose_action(network, device, self.state, exploration_rate)
    with self.timings['game']:
      exp = self.game.step(self.game.actions[action.action_index])
      self.state = exp.state_after.as_tensor()
      self.episode_steps += 1
      self.episode_reward += exp.reward
      if exp.done:
        episode_completed = EpisodeCompleted(self.episode_steps, self.episode_reward)
        self.episode_steps = 0
        self.episode_reward = 0.
      else:
        episode_completed = None
    with self.timings['remember']:
      exps = self.experience_buffer.process(exp, action.is_best)
    return episode_completed, exps

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.game.close()


def run_episode(model: ExecutionModel, game: Game) -> (float, int, Dict[str, int]):
  """
  Run a single episode (mostly until you died).
  :return: (total reward, number of steps taken, action map)
  """
  model.policy_net.eval()
  state = game.reset().as_tensor()
  total_reward = 0
  steps = 0
  actions = {}
  for a in game.actions:
    actions[a.name] = 0
  while True:
    action_index = best_action(model.device, model.policy_net, state)
    action = game.actions[action_index]
    actions[action.name] += 1

    exp = game.step(action)
    steps += 1
    total_reward += exp.reward
    state = exp.state_after.as_tensor()

    if exp.done:
      break
  return total_reward, steps, actions


def run_validation(model: ExecutionModel, game: Game, episodes_to_play: int) -> ValidationLog:
  """
  Run multiple episodes to verify the performance
  :return: (average rewards, array of episode rewards, total number of steps, action map)
  """
  actions = {}
  for a in game.actions:
    actions[a.name] = 0
  t0 = time()
  rewards = FloatStatCollector()
  steps = 0
  for _ in range(episodes_to_play):
    reward, s, episode_actions = run_episode(model, game)
    for name, c in episode_actions.items():
      actions[name] += c
    rewards.record(reward)
    steps += s
  return ValidationLog(
    at_training_epoch=model.trained_for_epochs,
    episodes=episodes_to_play,
    steps=steps,
    duration_seconds=time() - t0,
    episode_reward=rewards.get(),
    actions_taken=actions
  )


def play_example(model: ExecutionModel, game: Game, name='example', silent: bool = False) -> (str, int, float):
  """
  Plays an episode of the game and saves a video of it.
  :returns the name of the video file
  """
  model.policy_net.eval()
  step = 0
  total_reward = 0
  actions = {}
  for a in game.actions:
    actions[a.name] = 0
  images = []
  state = game.reset().as_tensor()
  while True:
    action_index = best_action(model.device, model.policy_net, state)
    action = game.actions[action_index]
    actions[action.name] += 1

    exp = game.step(action)
    state = exp.state_after.as_tensor()
    if not silent:
      print('- %4d  reward=%3.0f  action=%s' % (step, exp.reward, action.name))
    total_reward += exp.reward

    images.append({
      'step': step,
      'image': state[0],
      'action': action
    })
    step += 1
    if exp.done:
      if not silent:
        print('Done')
      break

  fig = plt.figure()
  movie_frames = []
  for f in images:
    plt.title('%s (%d -> %.0f)' % (name, len(images), total_reward))
    movie_frames.append([plt.imshow(f['image'], animated=True)])
  ani = animation.ArtistAnimation(fig, movie_frames, interval=40, blit=True, repeat=False)
  movie_name = 'videos/%s-%s-%s.mp4' % (game.name, model.strategy_name, name)
  ani.save(movie_name)
  if not silent:
    plt.show()
    print('done after %d steps. Total reward: %.0f' % (step, total_reward))
    print('- ', actions)
    print('- Saved movie to %s' % movie_name)
  plt.close()
  return movie_name, step, total_reward


def load(model: nn.Module, file: str) -> nn.Module:
  model.load_state_dict(torch.load(file))
  model.eval()
  return model
