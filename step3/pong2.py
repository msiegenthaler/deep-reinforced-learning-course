# %%
import matplotlib.pyplot as plt
import torch
from torch.optim import RMSprop

from drl.deepq.checkpoint import load_checkpoint, save_checkpoint
from drl.deepq.execution import play_example
from drl.deepq.model import LearningModel
from drl.deepq.networks import DuelingDQN
from drl.deepq.replay_memory import PrioritizedReplayMemory
from drl.deepq.train import TrainingHyperparameters, pretrain, linear_increase, linear_decay, \
  play_and_remember_steps, train, print_validation, prefill_memory
from drl.openai.pong import Pong

game_steps_per_step = 1
batch_per_game_step = 32
batch_size = game_steps_per_step * batch_per_game_step

w = h = 86
t = 4
memory_size = 200000

hyperparams = TrainingHyperparameters(
  gamma=0.99,
  beta=linear_increase(0.05),
  # exploration_rate=linear_decay(0.02, max_value=0.8, min_value=0.02),
  exploration_rate=linear_decay(0., max_value=0., min_value=0.),
  batch_size=batch_size,
  game_steps_per_step=game_steps_per_step,
  copy_to_target_every=10,
  game_steps_per_epoch=100,
  multi_step_n=10
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
  strategy_name='mspdddq',
  device=device
)
print('Model prepared')

# %%
# if load_checkpoint(model):
#   play_and_remember_steps(model, hyperparams)
#   print('Resuming completed')
# else:
#   pretrain(model, hyperparams, 0)
  # print('Pretraining finished')

# %%
def go():
  train(model, hyperparams, 10)
  print_validation(model, 20)
  play_example(model.exec())

# %%
# train(model, hyperparams, 5, validation_episodes=2, example_every=0)
# save_checkpoint(model)

# print_validation(model, 20)

# %%
# for i in range(3):
#   play_example(model.exec(), 'run-%d' % i)


prefill_memory(model, 32)
train(model, hyperparams, 1, validation_episodes=0, example_every=0)

for k in range(90, memory.size()):
  exp = memory.sum_tree.data[k]
  fig = plt.figure()
  fig.add_subplot(1,2,1)
  plt.title('%d - %d - %4.2f' % (k, exp.state_difference_in_steps, exp.reward))
  plt.imshow(exp.state_before[0].numpy())
  fig.add_subplot(1,2,2)
  plt.imshow(exp.state_after[0].numpy())
  plt.show()
