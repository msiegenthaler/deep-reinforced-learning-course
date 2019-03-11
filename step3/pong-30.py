# %%
import torch
from torch.optim import Adam

from drl.deepq.checkpoint import load_checkpoint, save_checkpoint
from drl.deepq.learn import LearningModel
from drl.deepq.networks import DQN
from drl.deepq.replay_memory import SimpleReplayMemory
from drl.deepq.train import TrainingHyperparameters, linear_increase, linear_decay, train, print_validation
from drl.openai.pong import Pong30Min

game_steps_per_step = 1
batch_per_game_step = 32
batch_size = game_steps_per_step * batch_per_game_step

w = h = 84
t = 4
memory_size = 100000

hyperparams = TrainingHyperparameters(
  gamma=0.99,
  beta=linear_increase(0.02),
  exploration_rate=linear_decay(0.006, max_value=1., min_value=0.02),
  batch_size=batch_size,
  game_steps_per_step=game_steps_per_step,
  copy_to_target_every=1000,
  game_steps_per_epoch=1000,
  init_memory_steps=10000,
  warmup_rounds=100
)


def create_game():
  return Pong30Min(w, h)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

with create_game() as _game:
  memory = SimpleReplayMemory(memory_size)
  policy_net = DQN(w, h, t, len(_game.actions)).to(device)

  model = LearningModel(
    memory=memory,
    policy_net=policy_net,
    target_net=DQN(w, h, t, len(_game.actions)).to(device),
    optimizer=Adam(policy_net.parameters(), lr=1e-5),
    game_name=_game.name,
    strategy_name='30min',
    device=device
  )
print('Model prepared')

# %%
if load_checkpoint(model):
  print('Resuming')
else:
  print('Starting fresh')

# %%
train(model, create_game, hyperparams, 500000 // hyperparams.game_steps_per_epoch,
      validation_episodes=0, save_every=25, example_every=0)
save_checkpoint(model)

with create_game() as game:
  print_validation(model, game, 1)
