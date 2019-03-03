# %%
import torch
from torch.optim import RMSprop

from drl.deepq.execution import play_example
from drl.deepq.learn import LearningModel
from drl.deepq.networks import DQN
from drl.deepq.replay_memory import PrioritizedReplayMemory
from drl.deepq.train import TrainingHyperparameters, pretrain, train
from step3.game_vizdoom import VizdoomBasicGame

game_steps_per_step = 2
batch_per_game_step = 64
batch_size = game_steps_per_step * batch_per_game_step

w = h = 86
t = 4
memory_size = 10000

hyperparams = TrainingHyperparameters(
  gamma=0.9,
  beta_increment=0.02,
  batch_size=batch_size,
  game_steps_per_step=game_steps_per_step,
  copy_to_target_every=300,
  game_steps_per_epoch=1000
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

game = VizdoomBasicGame(w, h, t)
memory = PrioritizedReplayMemory(memory_size)
# memory = SimpleReplayMemory(memory_size)
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
pretrain(model, hyperparams)
print('Pretraining finished')

# %%
#train(model, hyperparams, 10)

#%%
#play_example(model)
