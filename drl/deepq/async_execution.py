import abc
from typing import NamedTuple, Callable

import torch
from torch import nn
from torch.multiprocessing import Process, Queue

from drl.deepq.execution import EpisodeCompleted, GameExecutor
from drl.deepq.game import Experience


class AsyncGameExecutor(abc.ABC):
  @abc.abstractmethod
  def get_experiences(self) -> ([EpisodeCompleted, Experience]):
    pass

  @abc.abstractmethod
  def update_exploration_rate(self, exploration_rate):
    pass

  @abc.abstractmethod
  def close(self):
    pass

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()


GameFactory = Callable[[], GameExecutor]


class NotAyncGameExecutor(AsyncGameExecutor):
  def __init__(self, game_factory: GameFactory, network: nn.Module, device: torch.device, batch_size: int):
    self._game: GameExecutor = game_factory()
    self._network = network
    self._device = device
    self._batch_size = batch_size
    self.exploration_rate = 0.

  def get_experiences(self):
    return self._game.multi_step(self._network, self._device, self.exploration_rate, self._batch_size)

  def update_exploration_rate(self, exploration_rate):
    self.exploration_rate = exploration_rate

  def close(self):
    self._game.close()


class _RunGameRequest(NamedTuple):
  set_exploration_rate: float = None
  do_terminate: bool = False


def _run_game(process_id: int, game: GameExecutor, network: nn.Module, device: torch.device,
              request_queue: Queue, experience_queue: Queue, batch_size: int) -> None:
  exploration_rate = 1.
  # we use this to hold past (in-the-queue) experiences in memory until they get garbage collected
  while True:
    try:
      if not request_queue.empty():
        request: _RunGameRequest = request_queue.get(block=False)
        if request.do_terminate:
          print('* game worker %d terminated' % process_id)
          return
        if request.set_exploration_rate is not None:
          exploration_rate = request.set_exploration_rate

      response = game.multi_step(network, device, exploration_rate, batch_size)
      experience_queue.put(response, block=True)
    except Exception as e:
      print('error in worker %d: ' % process_id, e)


class MultiprocessAsyncGameExecutor(AsyncGameExecutor):
  def __init__(self, game_factory: GameFactory, network: nn.Module, device: torch.device,
               processes=1, steps_ahead=50, batch_size=1):
    self._experience_queue = Queue(maxsize=steps_ahead)
    print('* starting %d workers' % processes)
    self._processes = []
    self._request_queues = []
    for i in range(processes):
      request_queue = Queue(maxsize=10)
      p = Process(target=_run_game, args=(i, game_factory(), network, device, request_queue,
                                          self._experience_queue, batch_size,))
      p.start()
      self._request_queues.append(request_queue)
      self._processes.append(p)

  def _send_to_all(self, request, block=False):
    for request_queue in self._request_queues:
      request_queue.put(request, block=block)

  def get_experiences(self, block=True):
    return self._experience_queue.get(block=block)

  def update_exploration_rate(self, exploration_rate):
    self._send_to_all(_RunGameRequest(set_exploration_rate=exploration_rate))

  def close(self):
    print('* shutting down worker')
    self._send_to_all(_RunGameRequest(do_terminate=True))
    for p in self._processes:
      p.join()


def create_async_game_executor(game_factory: GameFactory, network: nn.Module, device: torch.device,
                               processes=0, steps_ahead=50, batch_size=1):
  if processes == 0:
    return NotAyncGameExecutor(game_factory, network, device, batch_size)
  else:
    return MultiprocessAsyncGameExecutor(game_factory, network, device, processes, steps_ahead, batch_size)