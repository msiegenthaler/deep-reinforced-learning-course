# %% [markdown]
# # Vizdoom with PyTorch
# %% [markdown]
# ## Game

# %%
from time import time
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import random
import numpy as np
from skimage import transform
from collections import deque
from vizdoom import *
from collections import namedtuple

Action = namedtuple("Action", "name key index")
Experience = namedtuple(
    "Experience", "state_before action state_after reward done")


class GameRepr:
  def __init__(self, x, y, t):
    left = Action("left", [1, 0, 0], 0)
    right = Action("right", [0, 1, 0], 1)
    fire = Action("fire", [0, 0, 1], 2)
    self.actions = [left, right, fire]

    self.game = self._setupGame()
    self.frames = Frames(x, y, t)
    self._new_episode()
    self.frame_shape = self.game.get_state().screen_buffer.shape

  def _setupGame(self):
    game = DoomGame()
    game.load_config("step3/basic.cfg")
    game.set_doom_scenario_path("step3/basic.wad")
    game.set_window_visible(False)
    game.init()
    return game

  def step(self, action):
    old_state = self.frames.state()
    reward, done = self._perform_action(action.key)
    new_state = self.frames.state()
    if done:
      self._new_episode()  # restart
    return Experience(state_before=old_state,
                      state_after=new_state,
                      action=action,
                      reward=reward,
                      done=done)

  def current_state(self):
    return self.frames.state()

  def reset(self):
    self._new_episode()
    return self.frames.state()

  def _new_episode(self):
    self.game.new_episode()
    frame = self.game.get_state().screen_buffer
    self.frames.add_initial_frame(frame)

  def _perform_action(self, action):
    reward = self.game.make_action(action)
    done = self.game.is_episode_finished()
    if done:
      frame = np.zeros(self.frame_shape)
      self.frames.add_frame(frame)
      return reward, True
    else:
      frame = self.game.get_state().screen_buffer
      self.frames.add_frame(frame)
      return reward, False

# %% [markdown]
# ## Frame Processing


# %%


class Frames:
  def __init__(self, x, y, t):
    self.x = x
    self .y = y
    self.t = t
    self.deque = deque(maxlen=t)

  def add_initial_frame(self, frame):
    processed = self.preprocess_frame(frame)
    for i in range(self.t):
      self.deque.append(processed)

  def add_frame(self, frame):
    processed = self.preprocess_frame(frame)
    self.deque.append(processed)

  def state(self):
    return np.stack(self.deque)

  def preprocess_frame(self, frame):
    cropped = frame[30:-20, :]
    normalized = cropped / 255.0
    return transform.resize(normalized, [self.x, self.y], mode='constant', anti_aliasing=False)

# %% [markdown]
# ## Replay Memory


# %%


class ReplayMemory:
  def __init__(self, size):
    self.deque = deque(maxlen=size)

  def remember(self, experience):
    self.deque.append(experience)

  def size(self):
    return len(self.deque)

  def sample(self, n):
    return random.sample(self.deque, n)

# %% [markdown]
# ## Neuronal Network


# %%


# %%
class DQN(nn.Module):
  def __init__(self, w, h, t, action_count):
    super(DQN, self).__init__()

    def conv2d_size_out(size, kernel_size=4, stride=2):
      return (size - (kernel_size - 1) - 1) // stride + 1

    self.conv = nn.Sequential(
        nn.Conv2d(t, 32, kernel_size=8, stride=2),
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2),
        nn.BatchNorm2d(128),
        nn.ELU()
    )

    conv_out_w = conv2d_size_out(
        conv2d_size_out(conv2d_size_out(w, kernel_size=8)))
    conv_out_h = conv2d_size_out(
        conv2d_size_out(conv2d_size_out(h, kernel_size=8)))
    linear_in = conv_out_w * conv_out_h * 128

    self.action_count = action_count
    self.linear = nn.Sequential(
        nn.Linear(linear_in, 512),
        nn.ELU(),
        nn.Linear(512, action_count))

    self._loss = nn.MSELoss()

  def forward(self, state):
    r = self.conv(state)
    r = r.view(r.size(0), -1)
    r = self.linear(r)
    return r

  def loss(self, predicted, actual):
    return self._loss(predicted, actual)

  def find_best_action(self, state):
    """returns index of best action"""
    with torch.no_grad():
      # t.max(1) will return largest column value of each row.
      # second column on max result is index of where max element was
      # found, so we pick action with the larger expected reward.
      return self.forward(state).max(1)[1].view(1, 1)

# %% [markdown]
# ## Training

# %%


def experience_to_tensor(exp):
  return Experience(
      state_before=torch.from_numpy(exp.state_before).float(),
      state_after=torch.from_numpy(exp.state_after).float(),
      action=exp.action,
      reward=exp.reward,
      done=exp.done
  )


# %%
def pretrain(game, memory, n):
  game.reset()
  for i in range(n):
    action = random.randrange(len(game.actions))
    exp = game.step(game.actions[action])
    texp = experience_to_tensor(exp)
    memory.remember(texp)


# %%
def chose_action(device, dqn, state, exploration_rate=None):
  """return index of action"""
  if exploration_rate != None and random.random() < exploration_rate:
    return random.randrange(dqn.action_count)
  dqn.eval()
  s = torch.from_numpy(state).unsqueeze(0).float().to(device)
  return dqn.find_best_action(s).item()


# %%
def calculate_loss(device, dqn, memory, batch_size, gamma):
  if memory.size() < batch_size:
    return
  exps = memory.sample(batch_size)

  # Calculate
  dqn.eval()
  with torch.no_grad():
    next_states = torch.stack(
        [e.state_after for e in exps if not e.done]).to(device)
    non_final_mask = torch.tensor(
        tuple(map(lambda e: not e.done, exps)), device=device, dtype=torch.uint8)
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = dqn(next_states).max(1)[0].detach()
    rewards = torch.tensor([e.reward for e in exps], device=device)
    expected_action_values = (next_state_values * gamma) + rewards

  dqn.train()
  states = torch.stack([e.state_before for e in exps]).to(device)
  actions = torch.stack([torch.tensor([e.action.index])
                         for e in exps]).to(device)
  predicted_action_values = dqn(states).gather(1, actions)

  loss = dqn.loss(expected_action_values.unsqueeze(1),
                  predicted_action_values)
  return loss


# %%
def learn_from_memory(device, dqn, optimizer, memory, batch_size, gamma):
  loss = calculate_loss(device, dqn, memory, batch_size, gamma)
  optimizer.zero_grad()
  loss.backward()
  # TODO clamp?
  optimizer.step()
  return loss

# %%


def train_step(device, dqn, optimizer, game, memory, batch_size, gamma, exploration_rate):
  state = game.current_state()
  action = chose_action(device, dqn, state, exploration_rate)
  exp = game.step(game.actions[action])
  state = exp.state_after
  memory.remember(experience_to_tensor(exp))

  loss = learn_from_memory(device, dqn, optimizer, memory, batch_size, gamma)

  return loss, exp


# %%


def train_epoch(device, dqn, optimizer, game, memory, batch_size, gamma, exploration_rate, steps):
  episode_reward = 0
  episode_rewards = []
  total_loss = 0
  start_t = time()
  for _ in range(steps):
    loss, exp = train_step(device, dqn, optimizer, game,
                           memory, batch_size, gamma, exploration_rate)
    total_loss += loss
    episode_reward += exp.reward
    if (exp.done):
      episode_rewards.append(episode_reward)
      episode_reward = 0
  return len(episode_rewards), episode_rewards, float(total_loss)/steps, time()-start_t

# %%


def run_episode(device, dqn, game):
  dqn.eval()
  game.reset()
  total_reward = 0
  steps = 0
  actions = {}
  for a in game.actions: actions[a.name] = 0
  while True:
    action_index = chose_action(device, dqn, game.current_state())
    action = game.actions[action_index]
    actions[action.name] += 1
    exp = game.step(action)
    steps += 1
    total_reward += exp.reward

    if (exp.done):
      break
  return total_reward, steps, actions

# %%
def run_validation(device, dqn, game, count):
  rewards = []
  steps = 0
  actions = {}
  for a in game.actions: actions[a.name] = 0
  for _ in range(count):
    reward, s, episode_actions = run_episode(device, dqn, game)
    for a in game.actions: actions[a.name] += episode_actions[a.name]
    rewards.append(reward)
    steps += s
  return sum(rewards)/count, rewards, steps, actions

# %% [markdown]
# ## Execution


# %%
w = 86
h = 86
t = 4

batch_size = 64
memory_size = 1000000

gamma = 0.99  # Discounting


# %%
game = GameRepr(w, h, t)


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dqn = DQN(w, h, t, len(game.actions))
total_epochs = 0
dqn.to(device)
device


# %%
memory = ReplayMemory(memory_size)
pretrain(game, memory, batch_size)


# %%
optimizer = optim.RMSprop(dqn.parameters())

#%%
def print_validation(episodes):
  val_rewards_avg, val_rewards, val_steps, val_actions = run_validation(
      device, dqn, game, episodes)
  print('Validation @%d: %.0f (min: %.0f, max %.0f) \tleft: %.2f right: %.2f fire %.2f' % (
      total_epochs, val_rewards_avg, min(val_rewards), max(val_rewards),
      val_actions['left']/val_steps, val_actions['right']/val_steps, val_actions['fire']/val_steps))

# %%
train_epochs = 20
learning_steps_per_epoch = 2000
validation_episodes = 10

if total_epochs == 0:
  max_exploration_rate = 0.8
else:
  max_exploration_rate = 0.2
min_exploration_rate = 0.1

save_every = 10


print('Starting training for %d epochs' % train_epochs)
for epoch in range(train_epochs):
  exploration_rate = max_exploration_rate - \
      (float(epoch)/train_epochs) * \
      (max_exploration_rate - min_exploration_rate)
  episodes, rewards, avg_loss, duration = train_epoch(
      device, dqn, optimizer, game, memory, batch_size, gamma, exploration_rate, learning_steps_per_epoch)
  steps_per_second = learning_steps_per_epoch/duration
  total_epochs += 1
  print('Epoch %d: %d eps.\texploration: %.2f \trewards: avg %.1f, min %.1f, max %.1f \tloss: %.0f \t %.1f step/s' %
        (total_epochs, episodes, exploration_rate, sum(rewards)/episodes, min(rewards), max(rewards), avg_loss, steps_per_second))

  print_validation(validation_episodes)

  if total_epochs % save_every == 0:
    file = './vs-doom-pytorch-%d' % total_epochs
    torch.save(dqn.state_dict(), file)
    print('saved model to', file)

print('Done training for %d epochs' % train_epochs)

# %%
print_validation(validation_episodes*3)