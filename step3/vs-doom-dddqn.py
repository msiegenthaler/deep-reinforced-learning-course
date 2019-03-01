# %% Imports
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import random
import numpy as np
from skimage import transform

# %% Game
from collections import namedtuple
Action = namedtuple("Action", "name key index")
Experience = namedtuple(
    "Experience", "state_before action state_after reward done")

from vizdoom import DoomGame
class VizdoomBasicGame:
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
    game.load_config("step3/simpler_basic.cfg")
    game.set_doom_scenario_path("step3/simpler_basic.wad")
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

# %% Frame
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
    normalized = frame / 255.0
    return transform.resize(normalized, [self.x, self.y], mode='constant', anti_aliasing=False)

# %% Sum Tree
import numpy as np
import math

class SumTree:
  def __init__(self, capacity):
    self.capacity = capacity
     # [--------------Parent nodes (n-1)----------][-------leaves to recode priority (n) -----]
    self.tree = np.zeros(2 * capacity - 1)
    self.data = np.zeros(capacity, dtype=object)
    self.pointer = 0
    self.leaf_offset = capacity - 1
    self.at_max_capacity = False

  def add(self, data, priority):
    self.data[self.pointer] = data
    self.update(self.pointer + self.leaf_offset, priority)
    self.pointer += 1
    if self.pointer >= self.capacity:
      self.at_max_capacity = True
      self.pointer = 0 # overwrite oldest if full

  def total_priority(self):
    return self.tree[0] # root node

  def max_prio(self):
    if self.size() == 0: return 0
    else: return np.max(self._valid_leafs())

  def min_prio(self):
    if self.size() == 0: return 0
    else: return np.min(self._valid_leafs())

  def size(self):
    if self.at_max_capacity: return self.capacity
    else: return self.pointer

  def get(self, prio_position):
    """prio_position: value between 0 and total_priority"""
    tree_index, data_index = self._get_leaf_position(prio_position)
    return self.tree[tree_index], self.data[data_index], tree_index

  def _valid_leafs(self):
    if self.at_max_capacity: return self.tree[-self.capacity:]
    else: return self.tree[self.leaf_offset:self.leaf_offset+self.pointer]

  def _get_leaf_position(self, prio_position):
    tree_index = 0 #start at root node
    while tree_index < self.leaf_offset: # stop when we are at leaf level
      index_left_child = 2 * tree_index + 1
      index_right_child = index_left_child + 1

      prio_left_child = self.tree[index_left_child]
      if prio_position <= prio_left_child:
        tree_index = index_left_child
      else:
        prio_position -= prio_left_child
        tree_index = index_right_child
    data_index = tree_index - self.leaf_offset
    if (data_index >= self.size()):
      # basically the prio position was too large, just return the last element
      data_index = self.size() - 1
      tree_index = data_index + self.leaf_offset
    return tree_index, data_index

  def update(self, tree_index, priority):
    if math.isnan(priority) or math.isinf(priority): priority = 1.0
    delta = priority - self.tree[tree_index]
    self.tree[tree_index] = priority
    while tree_index != 0:
      tree_index = (tree_index - 1) // 2 # parent node
      self.tree[tree_index] += delta

# %% Replay Memory
import abc
from collections import namedtuple

Sample = namedtuple("Sample", "experience weight more")

class ReplayMemory(abc.ABC):
  @abc.abstractmethod
  def remember(self, experience): pass

  @abc.abstractmethod
  def size(self): pass

  @abc.abstractmethod
  def sample(self, n): pass

  @abc.abstractmethod
  def update_weights(self, samples, td_errors): pass

# %% Simple Replay Memory
from collections import deque
class SimpleReplayMemory(ReplayMemory):
  def __init__(self, size):
    super(ReplayMemory)
    self.deque = deque(maxlen=size)

  def remember(self, experience):
    self.deque.append(experience)

  def size(self):
    return len(self.deque)

  def sample(self, n):
    noop = lambda error: ()
    w = lambda beta: 1.0
    return [Sample(e, w, noop) for e in random.sample(self.deque, n)]

  def update_weights(self, samples, td_errors): pass


# %% Prioritized Replay Memory
# see https://arxiv.org/abs/1511.05952
class PrioritizedReplayMemory(ReplayMemory):
  def __init__(self, capacity, epsilon=0.01, alpha=0.6, max_error=1.0):
    """alpha = randomness: tradeoff between uniform (alpha=0) and weighted (alpha=1)"""
    self.sum_tree = SumTree(capacity)
    self.epsilon = epsilon
    self.alpha = alpha
    self.max_error = max_error

  def remember(self, experience):
    prio = self.sum_tree.max_prio()
    if prio == 0: prio = self.max_error
    self.sum_tree.add(experience, prio)

  def size(self):
    return self.sum_tree.size()

  def sample(self, n):
    total_prio = self.sum_tree.total_priority()
    rvs = np.random.uniform(high=total_prio, size=n)
    # for weight normalization (only scale weight downwards)
    min_prob = self.sum_tree.min_prio() / total_prio

    result = []
    for rv in rvs:
      prio, exp, tree_index = self.sum_tree.get(rv)
      sampling_prob = prio / total_prio
      sample = Sample(
        experience = exp,
        weight = lambda beta, sampling_prob=sampling_prob, min_prob=min_prob:
          # shortened version of:
          # np.power(sampling_prob*size, -beta) / np.power(min_prob*size, -beta),
          (min_prob/sampling_prob) ** beta,
        more = tree_index)
      result.append(sample)
    return result

  def update_weights(self, samples, td_errors):
    clipped = torch.clamp(torch.abs(td_errors) + self.epsilon, max=self.max_error)
    prios = torch.pow(clipped, self.alpha)
    ps = prios.cpu().detach().numpy()
    for i, s in enumerate(samples):
      self.sum_tree.update(s.more, ps[i])

# %% DQN
class DQN(nn.Module):
  def __init__(self, w, h, t, action_count):
    super(DQN, self).__init__()

    def conv2d_size_out(size, kernel_size=4, stride=2):
      return (size - (kernel_size - 1) - 1) // stride + 1

    self.conv = nn.Sequential(
        nn.Conv2d(t, 32, kernel_size=8, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )

    conv_out_w = conv2d_size_out(
        conv2d_size_out(conv2d_size_out(w, kernel_size=8)))
    conv_out_h = conv2d_size_out(
        conv2d_size_out(conv2d_size_out(h, kernel_size=8)))
    linear_in = conv_out_w * conv_out_h * 128

    self.action_count = action_count
    self.linear = nn.Sequential(
        nn.Linear(linear_in, 512),
        nn.ReLU(),
        nn.Linear(512, action_count))

  def forward(self, state):
    r = self.conv(state)
    r = r.view(r.size(0), -1)
    r = self.linear(r)
    return r

  def find_best_action(self, state):
    """returns index of best action"""
    with torch.no_grad():
      # t.max(1) will return largest column value of each row.
      # second column on max result is index of where max element was
      # found, so we pick action with the larger expected reward.
      return self.forward(state).max(1)[1].view(1, 1)

# %% DuelingDQN
class DuelingDQN(nn.Module):
  def __init__(self, w, h, t, action_count):
    super(DuelingDQN, self).__init__()

    def conv2d_size_out(size, kernel_size=4, stride=2):
      return (size - (kernel_size - 1) - 1) // stride + 1

    self.conv = nn.Sequential(
        nn.Conv2d(t, 32, kernel_size=8, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )

    conv_out_w = conv2d_size_out(
        conv2d_size_out(conv2d_size_out(w, kernel_size=8)))
    conv_out_h = conv2d_size_out(
        conv2d_size_out(conv2d_size_out(h, kernel_size=8)))
    linear_in = conv_out_w * conv_out_h * 128

    self.action_count = action_count

    self.state_value_linear = nn.Sequential(
      nn.Linear(linear_in, 512),
      nn.ReLU(),
      nn.Linear(512, 1)
    )
    self.action_value_linear = nn.Sequential(
      nn.Linear(linear_in, 512),
      nn.ReLU(),
      nn.Linear(512, action_count)
    )


  def forward(self, state):
    r = self.conv(state)
    r = r.view(r.size(0), -1)
    state_value = self.state_value_linear(r)
    action_value = self.action_value_linear(r)
    value = state_value + action_value - torch.mean(action_value, dim=0)
    return value

  def find_best_action(self, state):
    """returns index of best action"""
    with torch.no_grad():
      # t.max(1) will return largest column value of each row.
      # second column on max result is index of where max element was
      # found, so we pick action with the larger expected reward.
      return self.forward(state).max(1)[1].view(1, 1)

# %%
def experience_to_tensor(exp):
  return Experience(
      state_before=torch.from_numpy(exp.state_before).float(),
      state_after=torch.from_numpy(exp.state_after).float(),
      action=exp.action,
      reward=exp.reward,
      done=exp.done
  )

#%% Timings
from time import time
class Timer:
  def __init__(self, name):
    self.name = name
    self.total_duration = 0.
    self.n = 0

  @property
  def avg_duration(self):
    if self.n == 0: return 0
    return self.total_duration / self.n

  def __enter__(self):
    self.t0 = time()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.total_duration += (time() - self.t0) * 1000
    self.n += 1
  
  def __str__(self):
    return '%s: %.0fms (%0.3fms per call)' % (self.name, self.total_duration, self.avg_duration)
  def format(self, length):
    return ('%-'+str(length)+'s: %10.0fms (%10.1fms per call, %10d calls)') % (self.name, self.total_duration, self.avg_duration, self.n)

class Timings:
  def __init__(self):
    self.reset()
  def _add(self, name):
    t = Timer(name)
    self.timers.append(t)
    return t

  def reset(self):
    self.timers = []
    # Elementar (don't sum)
    self.sample   = self._add("Sample")
    self.remember  = self._add("Remember")
    self.memory_weights = self._add("Memory Weights")
    self.backprop = self._add("Backprop")
    self.forward  = self._add("Forward")
    self.game     = self._add("Game")
    # Components
    self.learn    = self._add("Learn")
    self.play     = self._add("Play")
    # Totals
    self.step     = self._add("Step")
    self.epoch    = self._add("Epoch")

  def __str__(self):
    length = max(map(lambda t: len(t.name), self.timers))
    s = []
    for t in self.timers: s.append(t.format(length))
    return "\n".join(s)
  def __repr__(self):
    return str(self)

timings = Timings()

# %%
def pretrain(game, memory, n):
  game.reset()
  for i in range(n):
    action = random.randrange(len(game.actions))
    exp = game.step(game.actions[action])
    texp = experience_to_tensor(exp)
    memory.remember(texp)

# %%
def chose_action(device, policy_net, state, exploration_rate=None):
  """return index of action"""
  if exploration_rate != None and random.random() < exploration_rate:
    return random.randrange(policy_net.action_count)
  s = torch.from_numpy(state).unsqueeze(0).float().to(device)
  return policy_net.find_best_action(s).item()

# %%
def get_target_action_values(device, target_net, gamma, exps):
  next_states = torch.stack(
      [e.state_after for e in exps if not e.done]).to(device)
  non_final_mask = torch.tensor(
      tuple(map(lambda e: not e.done, exps)), device=device, dtype=torch.uint8)
  next_state_values = torch.zeros(len(exps), device=device)
  next_state_values[non_final_mask] = target_net(next_states).max(1)[
      0].detach()

  rewards = torch.tensor([e.reward for e in exps], device=device)
  target_action_values = (next_state_values * gamma) + rewards
  return target_action_values.unsqueeze(1)

#%%
def calculate_losses(device, target_net, policy_net, exps, gamma):
  target_action_values = get_target_action_values(device, target_net, gamma, exps).detach()

  states = torch.stack([e.state_before for e in exps])
  states = states.to(device)
  actions = torch.stack([torch.tensor([e.action.index]) for e in exps])
  actions = actions.to(device)
  predicted_action_values = policy_net(states).gather(1, actions)

  losses = F.smooth_l1_loss(predicted_action_values, target_action_values, reduction='none')
  return losses

# %%
def learn_from_memory(device, target_net, policy_net, optimizer, memory, batch_size, gamma, beta):
  if memory.size() < batch_size:
    raise ValueError('memory contains less than batch_size (%d) samples' % batch_size)

  with timings.sample:
    sample = memory.sample(batch_size)
    exps = [s.experience for s in sample]
    weights = torch.tensor([s.weight(beta) for s in sample])
    weights = weights.to(device)

  with timings.forward:
    losses = calculate_losses(device, target_net, policy_net, exps, gamma)
    loss = torch.mean(losses * weights)

  with timings.backprop:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

  with timings.memory_weights:
    memory.update_weights(sample, losses.detach())

  return loss

# %%
def train_step(device, target_net, policy_net, optimizer, game, memory, batch_size, game_steps_per_train_step, gamma, beta, exploration_rate):
  with timings.step:
    exps = []
    with timings.play:
      state = game.current_state()
      for _ in range(game_steps_per_train_step):
        with timings.forward:
          action = chose_action(device, policy_net, state, exploration_rate)
        with timings.game:
          exp = game.step(game.actions[action])
        state = exp.state_after
        with timings.remember:
          memory.remember(experience_to_tensor(exp))
        exps.append(exp)

    with timings.learn:
      loss = learn_from_memory(device, target_net, policy_net,
                              optimizer, memory, batch_size, gamma, beta)
    if math.isnan(loss.item()) or math.isinf(loss.item()): raise ValueError('infinite loss')

    return loss, exps


# %%
def train_epoch(device, target_net, policy_net, optimizer, game, memory, batch_size, game_steps_per_train_step, gamma, beta, exploration_rate, steps, copy_to_target_every):
  with timings.epoch:
    episode_reward = 0
    episode_rewards = []
    total_loss = 0
    start_t = time()
    policy_net.train()
    for step in range(steps):
      loss, exps = train_step(device, target_net, policy_net, optimizer, game,
                            memory, batch_size, game_steps_per_train_step, gamma, beta, exploration_rate)
      total_loss += loss

      for exp in exps:
        episode_reward += exp.reward
        if exp.done:
          episode_rewards.append(episode_reward)
          episode_reward = 0

      if step % copy_to_target_every == 0:
        target_net.load_state_dict(policy_net.state_dict())

    target_net.load_state_dict(policy_net.state_dict())
    return len(episode_rewards), episode_rewards, float(total_loss)/steps, time()-start_t

# %%
def run_episode(device, dqn, game):
  dqn.eval()
  game.reset()
  total_reward = 0
  steps = 0
  actions = {}
  for a in game.actions:
    actions[a.name] = 0
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
  for a in game.actions:
    actions[a.name] = 0
  for _ in range(count):
    reward, s, episode_actions = run_episode(device, dqn, game)
    for a in game.actions:
      actions[a.name] += episode_actions[a.name]
    rewards.append(reward)
    steps += s
  return sum(rewards)/count, rewards, steps, actions

# %% [markdown]
# ## Execution


# %%
w = 86
h = 86
t = 4

use_dueling = True
use_prioritized_replay = True

base_batch_size = 64          # number of game steps
game_steps_per_train_step = 2 # number of game steps per learning batch -> optimize for fastests steps/s

batch_size = base_batch_size * game_steps_per_train_step
memory_size = 100000

gamma = 0.9  # Discounting

# %%
game = VizdoomBasicGame(w, h, t)
game_name = 'vizdoom-basic'

# %% Deep Network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_network():
  global strategy_name
  if use_dueling:
    # DuelingDQN Network
    target_net = DuelingDQN(w, h, t, len(game.actions))
    policy_net = DuelingDQN(w, h, t, len(game.actions))
    strategy_name = 'ddqn'
  else:
    # DQN Network
    target_net = DQN(w, h, t, len(game.actions))
    policy_net = DQN(w, h, t, len(game.actions))
    strategy_name = 'dqn'
  target_net.to(device)
  policy_net.to(device)
  global total_epochs
  total_epochs = 0
  return target_net, policy_net

target_net, policy_net = init_network()
optimizer = optim.RMSprop(policy_net.parameters())
print('Using device %s' % device)

# %% memory
if use_prioritized_replay:
  memory = PrioritizedReplayMemory(memory_size)
else:
  memory = SimpleReplayMemory(memory_size)
pretrain(game, memory, batch_size*2)

# %% print validation
def print_validation(episodes):
  val_rewards_avg, val_rewards, val_steps, val_actions = run_validation(
      device, policy_net, game, episodes)
  print('Validation @%d: %.0f (min: %.0f, max %.0f) \tleft: %.2f right: %.2f fire %.2f' % (
      total_epochs, val_rewards_avg, min(val_rewards), max(val_rewards),
      val_actions['left']/val_steps, val_actions['right']/val_steps, val_actions['fire']/val_steps))

# %% play_example
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def play_example(name='example', silent=False):
  policy_net.eval()
  game.reset()
  total_reward = 0
  actions = {}
  for a in game.actions:
    actions[a.name] = 0
  images = []
  for step in range(300):
    action_index = chose_action(device, policy_net, game.current_state())
    action = game.actions[action_index]
    actions[action.name] += 1
    exp = game.step(action)
    s = experience_to_tensor(exp).state_after.unsqueeze(0).to(device)
    expected_reward = policy_net(s).cpu().detach().numpy()
    if not silent:
      print('%d  reward=%.0f  action=%s' % (step, exp.reward, action.name))
      print('    exp reward at state: %s' % (expected_reward))
    total_reward += exp.reward

    images.append({
      'step': step,
      'expected_reward': expected_reward,
      'image': exp.state_after[0],
      'action': action
    })
    if (exp.done):
      if not silent: print('Done')
      break
  fig = plt.figure()
  movie_frames = []
  for f in images:
    plt.title('%s (%d -> %.0f)' % (name, len(images), total_reward))
    movie_frames.append([plt.imshow(f['image'], animated=True)])
  plt.show()
  ani = animation.ArtistAnimation(fig, movie_frames, interval=200, blit=True, repeat=False)
  movie_name = '%s-%s-%s.mp4' % (game_name, strategy_name, name)
  ani.save(movie_name)
  if not silent:
    print('done after %d steps. Total reward: %.0f' % (step, total_reward))
    print(actions)
    print('Saved movie to %s' % movie_name)

# %%
def warm_up(rounds=10):
  print('warming up for %d rounds' % rounds)
  opt = optim.RMSprop(policy_net.parameters(), lr=1e-5)
  for r in range(rounds):
    # print(list(policy_net.parameters())[0][0][0])
    loss = learn_from_memory(device, target_net, policy_net, opt, memory, 8, gamma, 1.0).item()
    # print(loss)
    # print(list(policy_net.parameters())[0][0][0])
    if math.isnan(loss) or math.isinf(loss):
      raise ValueError('infinite loss after round %d' % r)
  print('warm up done')

# %% Train
from math import ceil
def train(train_epochs, game_steps_per_epoch=1000, save_every=10, example_every=5, validation_episodes=5, beta_increment=0.03, max_exploration_rate=0.8):
  learning_steps_per_epoch = ceil(game_steps_per_epoch / game_steps_per_train_step)

  copy_to_target_every = 100  # steps

  global total_epochs
  min_exploration_rate = 0.1

  beta = min(1, total_epochs * beta_increment)

  print('Starting training for %d epochs a %d steps (with batch_size %d)' % (train_epochs, game_steps_per_epoch, batch_size))
  for epoch in range(train_epochs):
    exploration_rate = max_exploration_rate - \
        (float(epoch)/train_epochs) * \
        (max_exploration_rate - min_exploration_rate)
    beta = min(1, beta+beta_increment)
    episodes, rewards, avg_loss, duration = train_epoch(device, target_net, policy_net, optimizer, game, memory,
                                                        batch_size, game_steps_per_train_step, gamma, beta, exploration_rate, learning_steps_per_epoch,
                                                        copy_to_target_every)
    steps_per_second = learning_steps_per_epoch/duration * game_steps_per_train_step
    total_epochs += 1
    if episodes != 0:
      print('Epoch %d: %d eps.\texpl: %.2f  beta %.2f\trewards: avg %.1f, min %.1f, max %.1f \tloss: %.2f \t %.1f step/s (%.1fs total)' %
            (total_epochs, episodes, exploration_rate, beta, sum(rewards)/(episodes), min(rewards), max(rewards), avg_loss, steps_per_second, duration))
    else:
      print('Epoch %d: 0 eps.\texpl: %.2f  beta %.2f\tloss: %.2f' %
            (total_epochs, exploration_rate, beta, avg_loss))

    if validation_episodes > 0: print_validation(validation_episodes)

    if save_every!=0 and total_epochs % save_every == 0:
      file = './state-%s-%s-%d.py' % (game_name, strategy_name, total_epochs)
      torch.save(policy_net.state_dict(), file)
      print('saved model to', file)

    if example_every!=0 and total_epochs % example_every == 0:
      play_example('epoch%d' % total_epochs, silent=True)

  print('Done training for %d epochs' % train_epochs)

def train_single(steps=game_steps_per_train_step):
  train(train_epochs=1, game_steps_per_epoch=steps, save_every=0, example_every=0, validation_episodes=0)


# %%
target_net, policy_net = init_network()
warm_up(500)

# %%
train(10)

# %%
print_validation(30)

# %%
play_example()
