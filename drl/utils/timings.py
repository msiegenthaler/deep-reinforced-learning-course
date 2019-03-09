from time import time


class Timer:
  def __init__(self, name):
    self.name = name
    self.total_duration = 0.
    self.n = 0

  @property
  def avg_duration(self):
    if self.n == 0:
      return 0
    return self.total_duration / self.n

  def __enter__(self):
    self.t0 = time()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.total_duration += (time() - self.t0) * 1000
    self.n += 1

  def __str__(self):
    return '%s: %.0fms (%0.3fms per call)' % (self.name, self.total_duration, self.avg_duration)

  def format(self, length):
    return ('%-' + str(length) + 's: %10.0fms (%10.1fms per call, %10d calls)') % (
      self.name, self.total_duration, self.avg_duration, self.n)


class Timings:
  def __init__(self):
    self.timers = {}

  def reset(self):
    self.timers = {}

  def __getitem__(self, name: str) -> Timer:
    if name in self.timers:
      return self.timers[name]
    else:
      timer = Timer(name)
      self.timers[name] = timer
      return timer

  def __str__(self):
    length = max(map(lambda name: len(name), self.timers))
    s = []
    for _, t in self.timers.items():
      s.append(t.format(length))
    return "\n".join(s)

  def __repr__(self):
    return str(self)
