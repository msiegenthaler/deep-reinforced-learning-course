from typing import Dict, NamedTuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor

from drl.deepq.game import Game


class ExecutionModel(NamedTuple):
  policy_net: nn.Module
  game: Game
  device: object
  strategy_name: str


def best_action(device, policy_net: nn.Module, state: Tensor) -> int:
  """
  :returns index of the chosen action
  """
  with torch.no_grad():
    s = state.unsqueeze(0).to(device)
    # t.max(1) will return largest column value of each row.
    # second column on max result is index of where max element was
    # found, so we pick action with the larger expected reward.
    return policy_net(s).max(1)[1].view(1, 1).item()


def run_episode(model: ExecutionModel) -> (float, int, Dict[str, int]):
  """
  Run a single episode (mostly until you died).
  :param model: the model to execute
  :return: (total reward, number of steps taken, action map)
  """
  model.policy_net.eval()
  state = model.game.reset()
  total_reward = 0
  steps = 0
  actions = {}
  for a in model.game.actions:
    actions[a.name] = 0
  while True:
    action_index = best_action(model.device, model.policy_net, state)
    action = model.game.actions[action_index]
    actions[action.name] += 1

    exp = model.game.step(action)
    steps += 1
    total_reward += exp.reward
    state = exp.state_after

    if exp.done:
      break
  return total_reward, steps, actions


def run_validation(model: ExecutionModel, count: int) -> object:
  """
  Run multiple episodes to verify the performance
  :param model: the model to execute
  :param count: number of episodes to execute
  :return: (average rewards, array of episode rewards, total number of steps, action map)
  """
  rewards = []
  steps = 0
  actions = {}
  for a in model.game.actions:
    actions[a.name] = 0
  for _ in range(count):
    reward, s, episode_actions = run_episode(model)
    for name, c in episode_actions.items():
      actions[name] += c
    rewards.append(reward)
    steps += s
  return sum(rewards) / count, rewards, steps, actions


def play_example(model: ExecutionModel, name='example', silent: bool = False) -> str:
  """
  Plays an episode of the game and saves a video of it.
  :returns the name of the video file
  """
  model.policy_net.eval()
  step = 0
  total_reward = 0
  actions = {}
  for a in model.game.actions:
    actions[a.name] = 0
  images = []
  state = model.game.reset()
  while True:
    action_index = best_action(model.device, model.policy_net, state)
    action = model.game.actions[action_index]
    actions[action.name] += 1

    exp = model.game.step(action)
    if not silent:
      print('- %4d  reward=%3.0f  action=%s' % (step, exp.reward, action.name))
    total_reward += exp.reward

    images.append({
      'step': step,
      'image': exp.state_after[0],
      'action': action
    })
    if (exp.done):
      if not silent:
        print('Done')
      break
    step += 1

  fig = plt.figure()
  movie_frames = []
  for f in images:
    plt.title('%s (%d -> %.0f)' % (name, len(images), total_reward))
    movie_frames.append([plt.imshow(f['image'], animated=True)])
  ani = animation.ArtistAnimation(fig, movie_frames, interval=40, blit=True, repeat=False)
  movie_name = 'videos/%s-%s-%s.mp4' % (model.game.name, model.strategy_name, name)
  ani.save(movie_name)
  if not silent:
    plt.show()
    print('done after %d steps. Total reward: %.0f' % (step, total_reward))
    print('- ', actions)
    print('- Saved movie to %s' % movie_name)
  return movie_name
