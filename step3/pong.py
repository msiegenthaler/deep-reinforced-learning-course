# %%
import torch
from torch.optim import RMSprop, Adam

from drl.deepq.checkpoint import load_checkpoint, save_checkpoint
from drl.deepq.execution import play_example
from drl.deepq.learn import LearningModel
from drl.deepq.networks import DuelingDQN
from drl.deepq.replay_memory import PrioritizedReplayMemory
from drl.deepq.train import TrainingHyperparameters, pretrain, linear_increase, linear_decay, \
  play_and_remember_steps, train, print_validation
from drl.openai.pong import Pong

game_steps_per_step = 4
batch_per_game_step = 32
batch_size = game_steps_per_step * batch_per_game_step

w = h = 86
t = 4
memory_size = 100000

f = 5
hyperparams = TrainingHyperparameters(
  gamma=0.99,
  beta=linear_increase(0.01 * f),
  exploration_rate=linear_decay(0.01 * f, max_value=0.8, min_value=0.02),
  batch_size=batch_size,
  game_steps_per_step=game_steps_per_step,
  copy_to_target_every=1000 // game_steps_per_step,
  game_steps_per_epoch=1000 * f,
  multi_step_n=4,
  init_memory_steps=10000
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

game = Pong(w, h, t)
memory = PrioritizedReplayMemory(memory_size)
policy_net = DuelingDQN(w, h, t, len(game.actions)).to(device)

model = LearningModel(
  game=game,
  memory=memory,
  policy_net=policy_net,
  target_net=DuelingDQN(w, h, t, len(game.actions)).to(device),
  optimizer=Adam(policy_net.parameters(), lr=1e-4),
  strategy_name='ms4_dddq',
  device=device
)
print('Model prepared')

# %%
if load_checkpoint(model):
  play_and_remember_steps(model, hyperparams)
  print('Resuming completed')
else:
  pretrain(model, hyperparams, warm_up_iterations=10)
  print('Pretraining finished')

# %%
print_validation(model, 1)
train(model, hyperparams, 200, validation_episodes=3, example_every=0, save_every=5 )
save_checkpoint(model)

print_validation(model, 50)

# %%
# play_example(model)
