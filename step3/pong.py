# %%
import torch
from torch.optim import RMSprop

from drl.deepq.execution import play_example
from drl.deepq.learn import LearningModel
from drl.deepq.networks import DuelingDQN
from drl.deepq.replay_memory import PrioritizedReplayMemory
from drl.deepq.train import TrainingHyperparameters, pretrain, resume_if_possible, linear_increase, linear_decay, \
  play_and_remember_steps, train
from drl.openai.pong import Pong

game_steps_per_step = 4
batch_per_game_step = 64
batch_size = game_steps_per_step * batch_per_game_step

w = h = 86
t = 4
memory_size = 200000

hyperparams = TrainingHyperparameters(
  gamma=0.99,
  beta=linear_increase(0.05),
  exploration_rate=linear_decay(0.02, max_value=0.8, min_value=0.02),
  batch_size=batch_size,
  game_steps_per_step=game_steps_per_step,
  copy_to_target_every=200,
  game_steps_per_epoch=1000
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
  optimizer=RMSprop(policy_net.parameters()),
  strategy_name='pdddq',
  device=device
)
print('Model prepared')

# %%
if resume_if_possible(model):
  play_and_remember_steps(model, hyperparams)
  print('Resuming completed')
else:
  pretrain(model, hyperparams)
  print('Pretraining finished')

# %%
# train(model, hyperparams, 10)

# %%
# play_example(model)
