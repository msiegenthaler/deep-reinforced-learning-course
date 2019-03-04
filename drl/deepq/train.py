import math
import os
import random
from time import time
from typing import NamedTuple, Callable

import torch
from torch import Tensor

from drl.deepq.execution import best_action, run_validation, ExecutionModel, play_example
from drl.deepq.game import Experience
from drl.deepq.learn import learn_from_memory, LearningModel
from drl.utils.timings import timings


def linear_decay(delta: float, min_value: float = 0., max_value: float = 1.):
  return lambda epoch: max(min_value, max_value - epoch * delta)


def linear_increase(delta: float, min_value: float = 0., max_value: float = 1.):
  return lambda epoch: min(max_value, min_value + epoch * delta)


class TrainingHyperparameters(NamedTuple):
  # number of items to process in one optimization step (backprop)
  batch_size: int

  # rate of random action (vs 'best' actions) [0,1]: epoch -> rate
  exploration_rate: Callable[[int], float] = linear_decay(0.05, min_value=0.1, max_value=0.8)

  # weighting of priorized experiences [0=no correction, 1=uniform]: epoch -> beta
  beta: Callable[[int], float] = linear_increase(0.05)

  # discounting factor for future (next-step) rewards (e.g. 0.99)
  gamma: float = 0.9

  # number of game steps (actions) to perform per batch
  game_steps_per_step: int = 1

  # how often (steps) the policy net is copied to the target net
  copy_to_target_every: int = 100

  # Game steps (actions) per training epoch
  game_steps_per_epoch: int = 1000


def prefill_memory(model: LearningModel, n):
  """Fill the memory with n experiences"""
  model.game.reset()
  for i in range(n):
    action = random.randrange(len(model.game.actions))
    exp = model.game.step(model.game.actions[action])
    model.memory.remember(exp)


def warm_up(model: LearningModel, rounds: int = 100) -> None:
  """
  Warm up the training model to prevent NaN losses and such bad things
  """
  for r in range(rounds):
    loss = learn_from_memory(model, 8, 0.9, 1.0)
    if math.isnan(loss) or math.isinf(loss):
      raise ValueError('infinite loss after part 1 round %d' % r)
  for r in range(rounds):
    loss = learn_from_memory(model, 16, 0.9, 1.0)
    if math.isnan(loss) or math.isinf(loss):
      raise ValueError('infinite loss after part 2 round %d' % r)


def pretrain(model: LearningModel, hyperparams: TrainingHyperparameters) -> None:
  prefill_memory(model, hyperparams.batch_size)
  warm_up(model, 500)


def chose_action(model: LearningModel, state: Tensor, exploration_rate: float) -> int:
  """
  :returns index of the chosen action
  """
  if random.random() < exploration_rate:
    return random.randrange(model.policy_net.action_count)
  return best_action(model.device, model.policy_net, state)


def train_step(model: LearningModel, hyperparameters: TrainingHyperparameters,
               beta, exploration_rate) -> (float, [Experience]):
  """
  :return: (average loss, experiences)
  """
  with timings['step']:
    exps = []
    with timings['play']:
      state = model.game.current_state()
      for _ in range(hyperparameters.game_steps_per_step):
        with timings['forward action']:
          action = chose_action(model, state, exploration_rate)
        with timings['game']:
          exp = model.game.step(model.game.actions[action])
        state = exp.state_after
        with timings['remember']:
          model.memory.remember(exp)
        exps.append(exp)

    with timings['learn']:
      loss = learn_from_memory(model, hyperparameters.batch_size, hyperparameters.gamma, beta)
    if math.isnan(loss) or math.isinf(loss):
      raise ValueError('infinite loss')

    model.status.trained_for_steps += 1
    return loss, exps


def train_epoch(model: LearningModel, hyperparameters: TrainingHyperparameters, beta, exploration_rate,
                steps: int) -> (int, [Experience], float):
  """
  :return: (episodes, experiences, average loss)
  """
  with timings['epoch']:
    episode_reward = 0
    episode_rewards = []
    total_loss = 0.
    model.policy_net.train()
    for step in range(steps):
      loss, exps = train_step(model, hyperparameters, beta, exploration_rate)
      total_loss += loss

      for exp in exps:
        episode_reward += exp.reward
        if exp.done:
          model.status.trained_for_episodes += 1
          episode_rewards.append(episode_reward)
          episode_reward = 0

      if step % hyperparameters.copy_to_target_every == 0:
        model.target_net.load_state_dict(model.policy_net.state_dict())

    model.target_net.load_state_dict(model.policy_net.state_dict())
    model.status.trained_for_epochs += 1
    return len(episode_rewards), episode_rewards, total_loss / steps


def train(model: LearningModel, hyperparams: TrainingHyperparameters, train_epochs,
          save_every=10, example_every=10, validation_episodes=10) -> None:
  """
  Train the model to get better at the game
  :param model:
  :param hyperparams: exploration_rate and beta will be calculated by this method
  :param train_epochs: how many epochs to train for
  :param save_every: save the model state every x epochs
  :param example_every: saves an example gameplay every x epochs
  :param validation_episodes: validation after each epoch for that many episodes
  :return: None
  """
  learning_steps_per_epoch = math.ceil(hyperparams.game_steps_per_epoch / hyperparams.game_steps_per_step)

  print('Starting training for %d epochs a %d steps (with batch_size %d)' % (train_epochs,
                                                                             hyperparams.game_steps_per_epoch,
                                                                             hyperparams.batch_size))

  for epoch in range(train_epochs):
    exploration_rate = hyperparams.exploration_rate(model.status.trained_for_epochs)
    beta = hyperparams.beta(model.status.trained_for_epochs)

    t0 = time()
    episodes, rewards, avg_loss = train_epoch(model, hyperparams, beta, exploration_rate, learning_steps_per_epoch)
    steps_per_second = learning_steps_per_epoch / (time() - t0) * hyperparams.game_steps_per_step
    if episodes != 0:
      print(
        ('Epoch %3d: %2d eps.\texpl: %4.2f beta %4.2f\trewards: %4.0f (%4.0f to %4.0f)' +
         '\tloss: %4.2f\t%4.1f step/s (%d steps, %d episodes)') %
        (model.status.trained_for_epochs, episodes, exploration_rate, beta,
         sum(rewards) / episodes, min(rewards), max(rewards), avg_loss, steps_per_second,
         model.status.trained_for_steps, model.status.trained_for_episodes))
    else:
      print('Epoch %3d:  0 eps.\texpl: %.2f  beta %.2f\tloss: %.2f\t%4.1f step/s (%d steps, %d episodes)' %
            (model.status.trained_for_epochs, exploration_rate, beta, avg_loss, steps_per_second,
             model.status.trained_for_steps, model.status.trained_for_episodes))

    if validation_episodes > 0:
      print_validation(model, validation_episodes)

    if save_every != 0 and model.status.trained_for_epochs % save_every == 0:
      save_checkpoint(model)

    if example_every != 0 and model.status.trained_for_epochs % example_every == 0:
      video, v_s, v_r = play_example(model_to_exec(model), 'epoch%04d' % model.status.trained_for_epochs, silent=True)
      print(' - saved example gameplay video to %s (reward: %.0f, steps: %d)' % (video, v_r, v_s))

  print('Done training for %d epochs' % train_epochs)


def save_checkpoint(model: LearningModel):
  data = {
    'epoch': model.status.trained_for_epochs,
    'episodes': model.status.trained_for_episodes,
    'steps': model.status.trained_for_steps,
    'optimizer_state_dict': model.optimizer.state_dict(),
    'model_state_dict': model.policy_net.state_dict()
  }
  file = 'checkpoints/%s-%s-%04d.pt' % (model.game.name, model.strategy_name, model.status.trained_for_epochs)
  torch.save(data, file)
  last_file = 'checkpoints/%s-%s-last.pt' % (model.game.name, model.strategy_name)
  torch.save(data, last_file)
  print(' - saved checkpoint to', file)


def resume_if_possible(model: LearningModel, suffix: str = 'last') -> bool:
  file = 'checkpoints/%s-%s-%s.pt' % (model.game.name, model.strategy_name, suffix)
  if os.path.isfile(file):
    data = torch.load(file)
    model.status.trained_for_epochs = data['epoch']
    model.status.trained_for_episodes = data['episodes']
    model.status.trained_for_steps = data['steps']
    model.optimizer.load_state_dict(data['optimizer_state_dict'])
    model.policy_net.load_state_dict(data['model_state_dict'])
    model.target_net.load_state_dict(data['model_state_dict'])
    print('loaded checkpoint from', file)
    return True
  else:
    return False


def play_and_remember_steps(model: LearningModel, hyperparameters: TrainingHyperparameters, batches: int = 10) -> None:
  exploration_rate = hyperparameters.exploration_rate(model.status.trained_for_epochs)
  steps = hyperparameters.batch_size * batches
  with timings['play']:
    state = model.game.current_state()
    for _ in range(steps):
      with timings['forward action']:
        action = chose_action(model, state, exploration_rate)
      with timings['game']:
        exp = model.game.step(model.game.actions[action])
      state = exp.state_after
      with timings['remember']:
        model.memory.remember(exp)


def model_to_exec(model: LearningModel) -> ExecutionModel:
  return ExecutionModel(policy_net=model.policy_net, device=model.device, game=model.game,
                        strategy_name=model.strategy_name)


def print_validation(model: LearningModel, episodes: int):
  # noinspection PyTypeChecker
  val_rewards_avg, val_rewards, val_steps, val_actions = run_validation(model, episodes)

  actions = []
  for name, cnt in val_actions.items():
    actions.append('%s: %.2f' % (name, cnt / val_steps))

  print(' - validation @%3d: %.0f (min: %4.0f, max %4.0f)\t%s' % (
    model.status.trained_for_epochs, val_rewards_avg, min(val_rewards), max(val_rewards), ", ".join(actions)))
