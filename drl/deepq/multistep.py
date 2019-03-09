import abc
from abc import ABC
from collections import deque

from drl.deepq.game import Experience


class ExperienceBuffer(ABC):
  @abc.abstractmethod
  def process(self, exp: Experience) -> [Experience]:
    pass


class SingleStep(ExperienceBuffer):
  def process(self, exp: Experience) -> [Experience]:
    return [exp]


class MultiStepBuffer(ExperienceBuffer):
  def __init__(self, n: int, gamma: float):
    self.n = n
    self.gamma = gamma
    self.buffer: deque[Experience] = deque()

  def process(self, exp: Experience) -> [Experience]:
    self.buffer.append(exp)
    if exp.done:
      exps = [exp]
      reward = 0
      for t in range(len(self.buffer) - 2, -1, -1):  # skip exp
        e = self.buffer[t]
        reward = e.reward + reward * self.gamma
        exps.append(e._replace(reward=reward, state_after=exp.state_after))
      self.buffer = []
      return exps
    elif len(self.buffer) >= self.n:
      reward = 0
      for t in range(self.n - 1):
        reward += (self.gamma ** t) * self.buffer[t].reward
      ret = self.buffer.pop()
      ret._replace(reward=reward, state_after=exp.state_after)
      return [ret]
    else:
      return []


def create_experience_buffer(n: int, gamma: float) -> ExperienceBuffer:
  if n == 1:
    return SingleStep()
  else:
    return MultiStepBuffer(n, gamma)
