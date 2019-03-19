# %%
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam

from drl.deepq.checkpoint import load_checkpoint, save_checkpoint
from drl.deepq.learn import LearningModel
from drl.deepq.networks import DQN_RBP
from drl.deepq.replay_memory import SimpleReplayMemory, PrioritizedReplayMemory
from drl.deepq.status_log import TrainingStatus
from drl.deepq.train import TrainingHyperparameters, linear_increase, linear_decay, train, print_validation
from drl.openai.pong import Pong, Pong30Min


def create_game():
  return Pong30Min(torch.float)


def train_with(device: torch.device, steps_to_train: int,
               game_steps_per_step: int, prio_memory: bool):
  episode_factor = 5
  w = h = 84
  t = 4
  memory_size = 50000
  batch_per_game_step = 32
  batch_size = game_steps_per_step * batch_per_game_step
  hyperparams = TrainingHyperparameters(
    gamma=0.99,
    beta=linear_increase(0.01 * episode_factor),
    exploration_rate=linear_decay(0.008 * episode_factor, max_value=1., min_value=0.01),
    batch_size=batch_size,
    game_steps_per_step=game_steps_per_step,
    copy_to_target_every=1000,
    game_steps_per_epoch=1000 * episode_factor,
    multi_step_n=4,
    warmup_rounds=500,
    init_memory_steps=1000,
    parallel_game_processes=2,
    max_batches_prefetch=10,
    states_on_device=True
  )

  with create_game() as _game:
    strategy_name = 'floaton-steps%d-%s' % (game_steps_per_step, 'prm' if prio_memory else 'srm')
    if prio_memory:
      memory = PrioritizedReplayMemory(memory_size)
    else:
      memory = SimpleReplayMemory(memory_size)
    policy_net = DQN_RBP(w, h, t, len(_game.actions)).to(device)
    target_net = DQN_RBP(w, h, t, len(_game.actions)).to(device)
    optimizer = Adam(policy_net.parameters(), lr=1e-4)

    summary_writer = SummaryWriter('runs/%s-%s-%s' % (_game.name, strategy_name, datetime.now().isoformat()))
    model = LearningModel(
      memory=memory,
      policy_net=policy_net,
      target_net=target_net,
      input_dtype=torch.float,
      optimizer=optimizer,
      strategy_name=strategy_name,
      game_name=_game.name,
      device=device,
      status=TrainingStatus(summary_writer)
    )
  print('%s: Model prepared' % strategy_name)

  # %%
  train(model, create_game, hyperparams, steps_to_train // hyperparams.game_steps_per_epoch, save_every=0)
  save_checkpoint(model)

  with create_game() as game:
    print('Running validation of', strategy_name)
    print_validation(model, game, 5)
  print('%s completed' % strategy_name)


if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn', force=True)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Using device %s' % device)

  steps = 500000
  game_steps = [1, 2, 4, 8, 16, 32]
  memory_type = [True, False]

  for gs in game_steps:
    for m in memory_type:
      train_with(device, steps, game_steps_per_step=gs, prio_memory=m)

  print('Done with all experiments.')
