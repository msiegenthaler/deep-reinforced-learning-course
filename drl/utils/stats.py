from typing import NamedTuple
import numpy as np


class FloatStat(NamedTuple):
  mean: float
  std: float
  min: float
  max: float
  count: int


class FloatStatCollector:
  def __init__(self):
    self.values = []

  def record(self, value: float) -> None:
    self.values.append(value)

  def get(self) -> FloatStat:
    if len(self.values) == 0:
      return FloatStat(np.nan, np.nan, np.nan, np.nan, 0)
    else:
      # noinspection PyTypeChecker
      return FloatStat(
        mean=np.mean(self.values),
        std=np.std(self.values),
        min=np.min(self.values),
        max=np.max(self.values),
        count=len(self.values)
      )
