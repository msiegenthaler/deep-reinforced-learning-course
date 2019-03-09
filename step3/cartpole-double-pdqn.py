# %%
import torch
from torch.optim import RMSprop

from drl.deepq.model import LearningModel
from drl.deepq.networks import DQN
from drl.deepq.replay_memory import PrioritizedReplayMemory
from drl.deepq.train import TrainingHyperparameters, pretrain, linear_increase, linear_decay, \
  play_and_remember_steps
from drl.deepq.checkpoint import load_checkpoint
from drl.openai.cartpole import CartPoleVisual

game_steps_per_step = 2
batch_per_game_step = 64
batch_size = game_steps_per_step * batch_per_game_step

w = 128
h = 64
t = 4
memory_size = 100000

hyperparams = TrainingHyperparameters(
  gamma=1,
  beta=linear_increase(0.05),
  exploration_rate=linear_decay(0.05, max_value=0.8, min_value=0.2),
  batch_size=batch_size,
  game_steps_per_step=game_steps_per_step,
  copy_to_target_every=300,
  game_steps_per_epoch=1000
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

game = CartPoleVisual(w, h, t)
memory = PrioritizedReplayMemory(memory_size)
policy_net = DQN(w, h, t, len(game.actions)).to(device)

model = LearningModel(
  game=game,
  memory=memory,
  policy_net=policy_net,
  target_net=DQN(w, h, t, len(game.actions)).to(device),
  optimizer=RMSprop(policy_net.parameters()),
  strategy_name='prio_double_dq',
  device=device
)
print('Model prepared')

# %%
if load_checkpoint(model):
  play_and_remember_steps(model, hyperparams)
  print('Resuming completed')
else:
  pretrain(model, hyperparams)
  print('Pretraining finished')

# %%
#train(model, hyperparams, 10)

# %%
#play_example(model)
