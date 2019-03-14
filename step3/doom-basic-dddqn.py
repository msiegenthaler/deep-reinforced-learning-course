# %%
import torch
from torch.optim import Adam

from drl.deepq.checkpoint import load_checkpoint, save_checkpoint
from drl.deepq.model import LearningModel
from drl.deepq.networks import DuelingDQN
from drl.deepq.replay_memory import PrioritizedReplayMemory
from drl.deepq.train import TrainingHyperparameters, linear_increase, linear_decay, train
from drl.vizdoom.vizdoom_basic import VizdoomBasicGame

if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn', force=True)

  episode_factor = 5
  w = h = 86
  t = 4
  memory_size = 50000
  game_steps_per_step = 8
  batch_per_game_step = 32
  batch_size = game_steps_per_step * batch_per_game_step
  hyperparams = TrainingHyperparameters(
    gamma=0.9,
    beta=linear_increase(0.01 * episode_factor),
    exploration_rate=linear_decay(0.008 * episode_factor, max_value=0.8, min_value=0.01),
    batch_size=batch_size,
    game_steps_per_step=game_steps_per_step,
    copy_to_target_every=1000,
    game_steps_per_epoch=1000 * episode_factor,
    multi_step_n=4,
    warmup_rounds=10,
    init_memory_steps=1000,
    parallel_game_processes=0,
    max_batches_prefetch=2,
    states_on_device=True
  )

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Using device %s' % device)

  def create_game():
    return VizdoomBasicGame(w, h, t, visible=False)

  with create_game() as game:
    memory = PrioritizedReplayMemory(memory_size)
    policy_net = DuelingDQN(w, h, t, len(game.actions)).to(device)
    target_net = DuelingDQN(w, h, t, len(game.actions)).to(device)

    model = LearningModel(
      memory=memory,
      policy_net=policy_net,
      target_net=DuelingDQN(w, h, t, len(game.actions)).to(device),
      optimizer=Adam(policy_net.parameters(), lr=1e-4),
      strategy_name='pdddqn',
      game_name=game.name,
      device=device
    )
  print('Model prepared')

  # %%
  if load_checkpoint(model):
    print('Resuming')
  else:
    print('Starting fresh')

  # %%
  train(model, create_game, hyperparams, 200 // episode_factor, save_every=25 // episode_factor)
  save_checkpoint(model)

  print('Done.')
