import math
import random
from time import time
from typing import NamedTuple, Callable

import numpy as np
from torch import Tensor

from drl.deepq.checkpoint import save_checkpoint
from drl.deepq.execution import best_action, run_validation, play_example
from drl.deepq.game import Game, GameFactory
from drl.deepq.learn import learn_from_memory
from drl.deepq.model import LearningModel, EpochTrainingLog, EpisodeLog
from drl.deepq.multistep import create_experience_buffer
from drl.utils.stats import FloatStatCollector


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

  # for multi step Q-learning: number of steps to accumulate the rewards
  multi_step_n: int = 1

  # number of game steps (actions) to perform per batch
  game_steps_per_step: int = 1

  # how often (steps) the policy net is copied to the target net
  copy_to_target_every: int = 100

  # Game steps (actions) per training epoch
  game_steps_per_epoch: int = 1000

  # Number of steps to have in the memory before beginning with training
  init_memory_steps: int = 1000

  # number of 'warmup' rounds (to avoid NaN/infinite loss)
  warmup_rounds: int = 0


def _prefill_memory_random(model: LearningModel, game: Game, n: int):
  """Fill the memory with n experiences"""
  game.reset()
  for i in range(n):
    action = random.randrange(len(game.actions))
    exp = game.step(game.actions[action])
    model.memory.remember(exp)
  game.close()


def _play_and_remember_steps(model: LearningModel, game: Game, hyperparams: TrainingHyperparameters, steps: int):
  exploration_rate = hyperparams.exploration_rate(model.status.trained_for_epochs)
  state = game.reset().as_tensor()
  experience_buffer = create_experience_buffer(hyperparams.multi_step_n, hyperparams.gamma)
  with model.status.timings['play']:
    for _ in range(steps):
      with model.status.timings['forward action']:
        action, best = chose_action(model, state, exploration_rate)
      with model.status.timings['game']:
        exp = game.step(game.actions[action])
      state = exp.state_after.as_tensor()
      with model.status.timings['remember']:
        for e in experience_buffer.process(exp, best):
          model.memory.remember(e)


def _warm_up(model: LearningModel, params: TrainingHyperparameters) -> None:
  """
  Warm up the training model to prevent NaN losses and such bad things
  """
  for r in range(params.warmup_rounds//2):
    loss = learn_from_memory(model, 8, params.gamma, params.beta(0))
    if math.isnan(loss) or math.isinf(loss):
      raise ValueError('infinite loss after part 1 round %d' % r)
  for r in range(params.warmup_rounds//2):
    loss = learn_from_memory(model, 16, params.gamma, params.beta(0))
    if math.isnan(loss) or math.isinf(loss):
      raise ValueError('infinite loss after part 2 round %d' % r)


def chose_action(model: LearningModel, state: Tensor, exploration_rate: float) -> (int, bool):
  """
  :returns (index of the chosen action, whether the action is 'best' (True) or random (False))
  """
  if random.random() < exploration_rate:
    return random.randrange(model.policy_net.action_count), False
  action = best_action(model.device, model.policy_net, state)
  return action, True


def train_epoch(model: LearningModel, game: Game, hyperparams: TrainingHyperparameters, beta: float,
                exploration_rate: float, episode_reward=0., episode_steps=0) -> (EpochTrainingLog, int, float):
  model.policy_net.train()
  t0 = time()
  episode_rewards = FloatStatCollector()
  total_loss = FloatStatCollector()
  experience_buffer = create_experience_buffer(hyperparams.multi_step_n, hyperparams.gamma)
  state = game.current_state().as_tensor()
  steps = math.ceil(hyperparams.game_steps_per_epoch // hyperparams.game_steps_per_step)
  with model.status.timings['epoch']:
    for step in range(steps):
      with model.status.timings['play']:
        for _ in range(hyperparams.game_steps_per_step):
          episode_steps += 1
          with model.status.timings['forward action']:
            action_index, best = chose_action(model, state, exploration_rate)

          with model.status.timings['game']:
            exp = game.step(game.actions[action_index])
            state = exp.state_after.as_tensor()
            episode_reward += exp.reward
            if exp.done:
              episode_rewards.record(episode_reward)
              model.status.training_episodes.append(EpisodeLog(
                at_training_epoch=model.status.trained_for_epochs + 1,
                reward=episode_reward,
                steps=episode_steps,
                exploration_rate=exploration_rate
              ))
              episode_reward = 0
              episode_steps = 0

          with model.status.timings['remember']:
            for e in experience_buffer.process(exp, best):
              model.memory.remember(e)

      with model.status.timings['learn']:
        if step % hyperparams.copy_to_target_every == 0:
          model.target_net.load_state_dict(model.policy_net.state_dict())
        loss = learn_from_memory(model, hyperparams.batch_size, hyperparams.gamma, beta)
        if math.isnan(loss) or math.isinf(loss):
          raise ValueError('infinite loss')
        total_loss.record(loss)

  er = episode_rewards.get()
  params = hyperparams._asdict()
  params['beta'] = beta
  params['exploration_rate'] = exploration_rate
  return EpochTrainingLog(
    episodes=er.count,
    trainings=steps,
    game_steps=hyperparams.game_steps_per_epoch,
    parameter_values=params,
    loss=total_loss.get(),
    episode_reward=er,
    duration_seconds=time() - t0
  ), episode_reward, episode_steps


def train(model: LearningModel, game_factory: GameFactory, hyperparams: TrainingHyperparameters, train_epochs,
          save_every=10, example_every=0, validation_episodes=0) -> None:
  """
  Train the model to get better at the game
  :param model: the model to train
  :param game_factory: creates instances of the game to play
  :param hyperparams: exploration_rate and beta will be calculated by this method
  :param train_epochs: how many epochs to train for
  :param save_every: save the model state every x epochs
  :param example_every: saves an example gameplay every x epochs
  :param validation_episodes: validation after each epoch for that many episodes
  :param avg_over_last_episodes: Length of running average window (for logging)
  :return: None
  """
  print('Starting training for %d epochs a %d steps (with batch_size %d)' % (train_epochs,
                                                                             hyperparams.game_steps_per_epoch,
                                                                             hyperparams.batch_size))

  validation_game = game_factory()
  train_game = game_factory()
  if model.memory.size() < hyperparams.init_memory_steps:
    print('- Prefilling memory with %d steps' % hyperparams.init_memory_steps)
    if model.status.trained_for_epochs == 0:
      _prefill_memory_random(model, train_game, hyperparams.init_memory_steps)
      if hyperparams.warmup_rounds > 0:
        print('- warming up the model')
        _warm_up(model, hyperparams)
    else:
      _play_and_remember_steps(model, train_game, hyperparams, hyperparams.init_memory_steps)

  train_episode_steps = 0
  train_episode_reward = 0
  train_game.reset()
  for epoch in range(train_epochs):
    print('Epoch: %3d' % (model.status.trained_for_epochs + 1))
    exploration_rate = hyperparams.exploration_rate(model.status.trained_for_epochs)
    beta = hyperparams.beta(model.status.trained_for_epochs)

    epoch_log, train_episode_reward, train_episode_steps = train_epoch(model, train_game,
                                                                       hyperparams, beta, exploration_rate,
                                                                       train_episode_reward, train_episode_steps)

    model.status.training_log.append(epoch_log)
    log_training(model, epoch_log)

    if validation_episodes > 0:
      print_validation(model, validation_game, validation_episodes)

    if save_every != 0 and model.status.trained_for_epochs % save_every == 0:
      save_checkpoint(model)

    if example_every != 0 and model.status.trained_for_epochs % example_every == 0:
      video, v_s, v_r = play_example(model.exec(), validation_game, 'epoch%04d' % model.status.trained_for_epochs,
                                     silent=True)
      print(' - saved example gameplay video to %s (reward: %.0f, steps: %d)' % (video, v_r, v_s))

  train_game.close()
  validation_game.close()
  print('Done training for %d epochs' % train_epochs)


def log_training(model: LearningModel, epoch_log: EpochTrainingLog) -> None:
  def avg_over(n: int) -> float:
    return np.mean([episode.reward for episode in model.status.training_episodes[-n:]])
  avgs = [5, 10, 25, 50, 100]
  if epoch_log.episodes != 0:
    avgs_over = ['%6.2f (%d)' % (avg_over(n), n) for n in avgs]
    print(' - completed %3d episodes, reward: %5.1f (%4.0f to %4.0f)  => total %d episodes: %s' % (
      epoch_log.episodes,
      epoch_log.episode_reward.mean, epoch_log.episode_reward.min, epoch_log.episode_reward.max,
      model.status.trained_for_episodes, ', '.join(avgs_over)))
  else:
    print(' - completed   0 episodes in %6d frames' % epoch_log.game_steps)
  print(' - expl: %4.2f beta %4.2f    loss: %4.2f    %4.1f step/s (%4.0fs)   %6d steps total' % (
    epoch_log.parameter_values['exploration_rate'], epoch_log.parameter_values['beta'],
    epoch_log.loss.mean, epoch_log.game_steps / epoch_log.duration_seconds, epoch_log.duration_seconds,
    model.status.trained_for_steps))


def print_validation(model: LearningModel, game: Game, episodes: int):
  log = run_validation(model.exec(), game, episodes)
  model.status.validation_log.append(log)

  actions = []
  for name, cnt in log.actions_taken.items():
    actions.append('%s: %.2f' % (name, cnt / log.steps))

  print(' - validation %5.1f    (min: %4.0f, max %4.0f)  %4d steps \t%s' % (
    log.episode_reward.mean, log.episode_reward.min, log.episode_reward.max, log.steps, ",  ".join(actions)))
