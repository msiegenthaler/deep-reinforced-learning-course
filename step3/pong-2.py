# %%
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam

from drl.deepq.checkpoint import load_checkpoint, save_checkpoint
from drl.deepq.learn import LearningModel
from drl.deepq.networks import DuelingDQN_RBP
from drl.deepq.replay_memory import PrioritizedReplayMemory
from drl.deepq.status_log import TrainingStatus
from drl.deepq.train import TrainingHyperparameters, linear_increase, linear_decay, train, print_validation
from drl.openai.pong import Pong

steps_to_train = 500000
episode_factor = 5
w = h = 84
t = 4
memory_size = 50000
game_steps_per_step = 4
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
  warmup_rounds=100,
  init_memory_steps=10000,
  parallel_game_processes=2,
  max_batches_prefetch=2,
  states_on_device=True
)


def create_game():
  return Pong(w, h, t)


if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn', force=True)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Using device %s' % device)

  with create_game() as _game:
    strategy_name = 'pdddq_rbp'
    memory = PrioritizedReplayMemory(memory_size)
    policy_net = DuelingDQN_RBP(w, h, t, len(_game.actions)).to(device)
    target_net = DuelingDQN_RBP(w, h, t, len(_game.actions)).to(device)

    summary_writer = SummaryWriter('runs/%s-%s-%s' % (_game.name, strategy_name, datetime.now().isoformat()))
    model = LearningModel(
      memory=memory,
      policy_net=policy_net,
      target_net=target_net,
      optimizer=Adam(policy_net.parameters(), lr=1e-4),
      strategy_name=strategy_name,
      game_name=_game.name,
      device=device,
      status=TrainingStatus(summary_writer)
    )
  print('Model prepared')

  # %%
  if load_checkpoint(model):
    print('Resuming')
  else:
    print('Starting fresh')

  # %%
  train(model, create_game, hyperparams, steps_to_train // hyperparams.game_steps_per_step,
        save_every=25 // episode_factor)
  save_checkpoint(model)

  with create_game() as game:
    print('Running validation')
    print_validation(model, game, 5)
  #   print('Playing example')
  #   play_example(model.exec(), game, 'final')

  print('Done.')
