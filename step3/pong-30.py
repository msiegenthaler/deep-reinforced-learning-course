# %%
import torch
from torch.optim import Adam

from drl.deepq.checkpoint import load_checkpoint, save_checkpoint
from drl.deepq.learn import LearningModel
from drl.deepq.networks import DQN
from drl.deepq.replay_memory import SimpleReplayMemory
from drl.deepq.train import TrainingHyperparameters, linear_increase, linear_decay, \
  play_and_remember_steps, train, print_validation, prefill_memory
from drl.openai.pong import Pong30Min

game_steps_per_step = 1
batch_per_game_step = 32
batch_size = game_steps_per_step * batch_per_game_step

w = h = 84
t = 4
memory_size = 100000

f = 5
hyperparams = TrainingHyperparameters(
  gamma=0.99,
  beta=linear_increase(0.02 * f),
  exploration_rate=linear_decay(0.006 * f, max_value=1., min_value=0.02),
  batch_size=batch_size,
  game_steps_per_step=game_steps_per_step,
  copy_to_target_every=1000,
  game_steps_per_epoch=1000 * f
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

game = Pong30Min(w, h)
# memory = PrioritizedReplayMemory(memory_size)
memory = SimpleReplayMemory(memory_size)
policy_net = DQN(w, h, t, len(game.actions)).to(device)

model = LearningModel(
  game=game,
  memory=memory,
  policy_net=policy_net,
  target_net=DQN(w, h, t, len(game.actions)).to(device),
  optimizer=Adam(policy_net.parameters(), lr=1e-4),
  strategy_name='30min',
  device=device
)
print('Model prepared')

# %%
if load_checkpoint(model):
  play_and_remember_steps(model, hyperparams)
  print('Resuming completed')
else:
  prefill_memory(model, 10000)
  # warm_up(model, 100)
  print('Pretraining finished')

# %%
print_validation(model, 3)
train(model, hyperparams, 25000000 // hyperparams.game_steps_per_epoch, validation_episodes=0, save_every=25 // f,
      example_every=0)
save_checkpoint(model)

print_validation(model, 1)

# %%
# play_example(model)
