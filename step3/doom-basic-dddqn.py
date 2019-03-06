# %%
import torch
from torch.optim import RMSprop

from drl.deepq.execution import play_example
from drl.deepq.learn import LearningModel
from drl.deepq.networks import DuelingDQN
from drl.deepq.replay_memory import PrioritizedReplayMemory
from drl.deepq.train import TrainingHyperparameters, pretrain, linear_increase, linear_decay, train
from drl.vizdoom.vizdoom_corridor import VizdoomCorridorGame

game_steps_per_step = 2
batch_per_game_step = 64
batch_size = game_steps_per_step * batch_per_game_step

w = h = 86
t = 4
memory_size = 100000

hyperparams = TrainingHyperparameters(
  gamma=0.9,
  beta=linear_increase(0.05),
  exploration_rate=linear_decay(0.02, max_value=0.8, min_value=0.1),
  batch_size=batch_size,
  game_steps_per_step=game_steps_per_step,
  copy_to_target_every=300,
  game_steps_per_epoch=1000
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

game = VizdoomCorridorGame(w, h, t, visible = True)
memory = PrioritizedReplayMemory(memory_size)
policy_net = DuelingDQN(w, h, t, len(game.actions)).to(device)

model = LearningModel(
  game=game,
  memory=memory,
  policy_net=policy_net,
  target_net=DuelingDQN(w, h, t, len(game.actions)).to(device),
  optimizer=RMSprop(policy_net.parameters()),
  strategy_name='duelingdoubledqn',
  device=device
)
print('Model prepared')

# %%
pretrain(model, hyperparams, 1000)
print('Pretraining finished')


# %%
def go():
  train(model, hyperparams, 10)
  play_example(model)