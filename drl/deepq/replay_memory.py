import abc
import random
from collections import namedtuple, Iterable, deque

import numpy as np
import torch

from drl.deepq.game import Experience
from drl.deepq.sumtree import SumTree


Sample = namedtuple("Sample", "experience weight more")

class ReplayMemory(abc.ABC):
  """Remembers past experiences and can sample from them"""
  @abc.abstractmethod
  def remember(self, experience: Experience) -> None: pass

  @abc.abstractmethod
  def size(self) -> int: pass

  @abc.abstractmethod
  def sample(self, n: int) -> [Experience]: pass

  @abc.abstractmethod
  def update_weights(self, samples: [Experience], td_errors: torch.Tensor): pass


class SimpleReplayMemory(ReplayMemory):
  """bounded (FIFO) memory with uniform random sampling"""
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


class PrioritizedReplayMemory(ReplayMemory):
  """
  Bounded Prioritized Replay Memory.
  See  https://arxiv.org/abs/1511.05952 for more details.
  """
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
