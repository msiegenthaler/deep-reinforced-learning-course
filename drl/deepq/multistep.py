import abc
from abc import ABC
from collections import deque
from typing import NamedTuple

from drl.deepq.game import Experience


class ExperienceBuffer(ABC):
  @abc.abstractmethod
  def process(self, exp: Experience, best: bool) -> [Experience]:
    pass


class SingleStep(ExperienceBuffer):
  def process(self, exp: Experience, best: bool) -> [Experience]:
    return [exp]

  def process_random(self, exp: Experience) -> [Experience]:
    return [exp]


class MultiStepBuffer(ExperienceBuffer):
  def __init__(self, n: int, gamma: float):
    self.n = n
    self.gamma = gamma
    self.buffer: deque[Experience] = deque()

  def process(self, exp: Experience, best: bool) -> [Experience]:
    if best:
      return self._process_best(exp)
    else:
      return self._process_random(exp)

  def _process_random(self, exp: Experience) -> [Experience]:
    exps = []
    reward = 0
    for t in range(len(self.buffer) - 1, -1, -1):
      e = self.buffer[t]
      reward = e.reward + reward * self.gamma
      exps.append(e._replace(
        reward=reward, state_after=exp.state_after, state_difference_in_steps=len(self.buffer) - t))
    self.buffer = deque([exp])
    return exps

  def _process_best(self, exp: Experience) -> [Experience]:
    self.buffer.append(exp)
    if exp.done:
      exps = [exp]
      reward = exp.reward
      for t in range(len(self.buffer) - 2, -1, -1):  # skip exp
        e = self.buffer[t]
        reward = e.reward + reward * self.gamma
        exps.append(e._replace(
          reward=reward, state_after=exp.state_after, state_difference_in_steps=len(self.buffer) - t))
      self.buffer = deque()
      return exps.reverse()  # just because it easier to debug
    elif len(self.buffer) >= self.n:
      reward = 0
      for t in range(self.n):
        reward += (self.gamma ** t) * self.buffer[t].reward
      ret = self.buffer.popleft()
      ret = ret._replace(reward=reward, state_after=exp.state_after, state_difference_in_steps=self.n)
      return [ret]
    else:
      return []


def create_experience_buffer(n: int, gamma: float) -> ExperienceBuffer:
  if n == 1:
    return SingleStep()
  else:
    return MultiStepBuffer(n, gamma)
