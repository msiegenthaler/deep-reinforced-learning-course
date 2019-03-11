# %%
import torch
from torch.optim import RMSprop, Adam

from drl.deepq.checkpoint import save_checkpoint, load_checkpoint
from drl.deepq.execution import play_example, run_validation
from drl.deepq.model import LearningModel
from drl.deepq.networks import DuelingDQN
from drl.deepq.replay_memory import PrioritizedReplayMemory
from drl.deepq.train import TrainingHyperparameters, pretrain, linear_decay, linear_increase, train, print_validation, \
  play_and_remember_steps, prefill_memory
from drl.utils import timings
from drl.vizdoom.vizdoom_corridor import VizdoomCorridorGame

game_steps_per_step = 2
batch_per_game_step = 64
batch_size = game_steps_per_step * batch_per_game_step

w = h = 86
t = 4
memory_size = 50000

hyperparams = TrainingHyperparameters(
  gamma=0.9,
  beta=linear_increase(0.02),
  exploration_rate=linear_decay(0.02, max_value=0.9, min_value=0.02),
  batch_size=batch_size,
  game_steps_per_step=game_steps_per_step,
  copy_to_target_every=1000,
  game_steps_per_epoch=5000,
  multi_step_n=5
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

game = VizdoomCorridorGame(w, h, t, visible=False)
memory = PrioritizedReplayMemory(memory_size)
policy_net = DuelingDQN(w, h, t, len(game.actions)).to(device)

model = LearningModel(
  game=game,
  memory=memory,
  policy_net=policy_net,
  target_net=DuelingDQN(w, h, t, len(game.actions)).to(device),
  optimizer=Adam(policy_net.parameters(), lr=1e-4),
  strategy_name='ms5_duelingdoubledqn',
  device=device
)
print('Model prepared')

# %%
if load_checkpoint(model):
  play_and_remember_steps(model, hyperparams, 100)
  print('Resuming completed')
else:
  prefill_memory(model, 10000)
  print('Pretraining finished')


train(model, hyperparams, 30, save_every=3, example_every=0 )
save_checkpoint(model)

print_validation(model, 100)
