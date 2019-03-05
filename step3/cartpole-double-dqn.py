# %%
import torch
from torch.optim import RMSprop

from drl.deepq.execution import play_example
from drl.deepq.learn import LearningModel
from drl.deepq.networks import DQN
from drl.deepq.replay_memory import PrioritizedReplayMemory, SimpleReplayMemory
from drl.deepq.train import TrainingHyperparameters, pretrain, train, resume_if_possible, linear_increase, linear_decay, \
  play_and_remember_steps
from step3.game_openai import CartPoleVisual

game_steps_per_step = 2
batch_per_game_step = 64
batch_size = game_steps_per_step * batch_per_game_step

w = 128
h = 64
t = 4
memory_size = 100000

hyperparams = TrainingHyperparameters(
  gamma=0.9,
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
# memory = PrioritizedReplayMemory(memory_size)
memory = SimpleReplayMemory(memory_size)
policy_net = DQN(w, h, t, len(game.actions)).to(device)

model = LearningModel(
  game=game,
  memory=memory,
  policy_net=policy_net,
  target_net=DQN(w, h, t, len(game.actions)).to(device),
  optimizer=RMSprop(policy_net.parameters()),
  strategy_name='doubledqn',
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
#train(model, hyperparams, 10)

# %%
#play_example(model)
