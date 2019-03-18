import abc
import queue
from typing import NamedTuple

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


class GameExecutorFactory(abc.ABC):
  @abc.abstractmethod
  def create(self) -> GameExecutor:
    pass


class NotAyncGameExecutor(AsyncGameExecutor):
  def __init__(self, game_factory: GameExecutorFactory, network: nn.Module, device: torch.device, batch_size: int,
               states_on_device: bool):
    self._game: GameExecutor = game_factory.create()
    self._network = network
    self._device = device
    self._batch_size = batch_size
    self.exploration_rate = 0.
    self._states_on_device = states_on_device

  def get_experiences(self):
    eps, exps = self._game.multi_step(self._network, self._device, self.exploration_rate, self._batch_size)
    if self._states_on_device:
      exps = [e.to_device(self._device) for e in exps]
    return eps, exps

  def update_exploration_rate(self, exploration_rate):
    self.exploration_rate = exploration_rate

  def close(self):
    self._game.close()


class _RunGameRequest(NamedTuple):
  set_exploration_rate: float = None
  do_terminate: bool = False


def _run_game(process_id: int, game_factory: GameExecutorFactory, network: nn.Module, device: torch.device,
              request_queue: Queue, experience_queue: Queue, batch_size: int, transfer_blocks: int,
              transfer_to_device: bool) -> None:
  exploration_rate = 1.
  game = game_factory.create()
  print('* worker %d started' % process_id)
  while True:
    try:
      if not request_queue.empty():
        request: _RunGameRequest = request_queue.get(block=False)
        if request.do_terminate:
          print('* game worker %d terminated' % process_id)
          experience_queue.close()
          request_queue.close()
          return
        if request.set_exploration_rate is not None:
          exploration_rate = request.set_exploration_rate

      block = []
      for _ in range(transfer_blocks):
        eps, exps = game.multi_step(network, device, exploration_rate, batch_size)
        if transfer_to_device:
          exps = [e.to_device(device, non_blocking=False) for e in exps]
        block.append((eps, exps))
      experience_queue.put(block, block=True)
    except Exception as e:
      print('error in worker %d: ' % process_id, e)


class MultiprocessAsyncGameExecutor(AsyncGameExecutor):
  def __init__(self, game_factory: GameExecutorFactory, network: nn.Module, device: torch.device,
               processes: int, batches_ahead: int, batch_size: int, states_on_device: bool):
    self._states_on_device = states_on_device
    self._device = device
    self._experience_queue = Queue(maxsize=processes + 1)
    block_size = max(1, batches_ahead - processes)
    self.block_buffer = []
    print('* starting %d workers (batch size: %d, block size: %d)' % (processes, batch_size, block_size))
    self._processes = []
    self._request_queues = []
    for i in range(processes):
      request_queue = Queue(maxsize=10)
      # Transfer to GPU in the other process does not work.. it does not throw an error, but training does not converge
      p = Process(target=_run_game, args=(i, game_factory, network, device, request_queue,
                                          self._experience_queue, batch_size, block_size, False,))
      p.start()
      self._request_queues.append(request_queue)
      self._processes.append(p)

  def _send_to_all(self, request, block=False):
    for request_queue in self._request_queues:
      request_queue.put(request, block=block)

  def get_experiences(self):
    if len(self.block_buffer) == 0:
      block_buffer = self._experience_queue.get(block=True)
      if self._states_on_device:
        for eps, exps in block_buffer:
          exps = [e.to_device(self._device) for e in exps]
          self.block_buffer.append((eps, exps))
      else:
        self.block_buffer.extend(block_buffer)
    return self.block_buffer.pop()

  def update_exploration_rate(self, exploration_rate):
    self._send_to_all(_RunGameRequest(set_exploration_rate=exploration_rate), block=True)

  def close(self):
    print('* shutting down workers')
    self._send_to_all(_RunGameRequest(do_terminate=True))
    # wake the workers
    try:
      while not self._experience_queue.empty():
        try:
          self._experience_queue.get(block=False)
        except queue.Empty:
          pass
    except ConnectionResetError:
      pass
    except FileNotFoundError:
      pass

    self._experience_queue.close()
    for p in self._processes:
      p.join(1000)
    for q in self._request_queues:
      q.close()
    self._experience_queue.close()


def create_async_game_executor(game_factory: GameExecutorFactory, network: nn.Module, device: torch.device,
                               processes=0, steps_ahead=50, batch_size=1, states_on_device=False):
  if processes == 0:
    return NotAyncGameExecutor(game_factory, network, device, batch_size, states_on_device)
  else:
    return MultiprocessAsyncGameExecutor(game_factory, network, device, processes, steps_ahead, batch_size,
                                         states_on_device)
