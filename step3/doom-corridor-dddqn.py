# %%
import datetime

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam

from drl.deepq.checkpoint import load_checkpoint, save_checkpoint
from drl.deepq.execution import play_example
from drl.deepq.model import LearningModel
from drl.deepq.networks import DuelingDQN, DuelingDQN_RBP
from drl.deepq.replay_memory import PrioritizedReplayMemory
from drl.deepq.status_log import TrainingStatus
from drl.deepq.train import TrainingHyperparameters, linear_increase, linear_decay, print_validation, train
from drl.vizdoom.vizdoom_corridor import VizdoomCorridorGame

steps_to_train = 500000

episode_factor = 5
w = h = 84
t = 4
memory_size = 40000
game_steps_per_step = 8
batch_per_game_step = 32
batch_size = game_steps_per_step * batch_per_game_step
hyperparams = TrainingHyperparameters(
  gamma=0.9,
  beta=linear_increase(0.01 * episode_factor),
  exploration_rate=linear_decay(0.008 * episode_factor, max_value=0.8, min_value=0.01),
  batch_size=batch_size,
  game_steps_per_step=game_steps_per_step,
  copy_to_target_every=500,
  game_steps_per_epoch=1000 * episode_factor,
  multi_step_n=4,
  warmup_rounds=10,
  init_memory_steps=1000,
  parallel_game_processes=4,
  max_batches_prefetch=10,
  states_on_device=True
)


def create_game():
  return VizdoomCorridorGame(w, h, t, visible=False)


if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn', force=True)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Using device %s' % device)

  with create_game() as game:
    strategy_name = 'pdddqn_rbp'
    memory = PrioritizedReplayMemory(memory_size)
    policy_net = DuelingDQN_RBP(w, h, t, len(game.actions)).to(device)
    target_net = DuelingDQN_RBP(w, h, t, len(game.actions)).to(device)

    summary_writer = SummaryWriter('runs/%s-%s-%s' % (game.name, strategy_name, datetime.datetime.now().isoformat()))
    model = LearningModel(
      memory=memory,
      policy_net=policy_net,
      target_net=target_net,
      optimizer=Adam(policy_net.parameters(), lr=1e-4),
      strategy_name=strategy_name,
      game_name=game.name,
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
        save_every=25 // episode_factor,
        validation_episodes=5)
  save_checkpoint(model)

  with create_game() as game:
    print('Running validation')
    print_validation(model, game, 10)
    print('Playing example')
    for i in range(3):
      play_example(model.exec(), game, '%04d' % i)

  print('Done.')
