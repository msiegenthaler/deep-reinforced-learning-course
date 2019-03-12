from collections import deque
from typing import NamedTuple, List, Optional, Callable

from torch import nn
from torch.multiprocessing import Process, Queue

from drl.deepq.execution import EpisodeCompleted, GameExecutor
from drl.deepq.game import Experience


class _RunGameRequest(NamedTuple):
  set_exploration_rate: float = None
  do_terminate: bool = False


class RunGameResponse(NamedTuple):
  experiences: List[Experience]
  completed_episode: Optional[EpisodeCompleted]


def _run_game(id, game: GameExecutor, network: nn.Module, device: str,
              request_queue: Queue, experience_queue: Queue, keep_n: int) -> None:
  exploration_rate = 1.
  # we use this to hold past (in-the-queue) experiences in memory until they get garbage collected
  keep_buffer = deque(maxlen=keep_n)
  while True:
    try:
      if not request_queue.empty():
        request: _RunGameRequest = request_queue.get(block=False)
        if request.do_terminate:
          print('* game worker %d terminated' % id)
          return
        if request.set_exploration_rate is not None:
          exploration_rate = request.set_exploration_rate

      episodes, exps = game.step(network, device, exploration_rate)
      keep_buffer.extend(exps)
      response = RunGameResponse(exps, episodes)
      experience_queue.put(response, block=True)
    except Exception as e:
      print('error in worker %d: ' % id, e)


class AsyncGameExecutor:
  def __init__(self, game_factory: Callable[[], GameExecutor], network: nn.Module, device: str,
               processes=1, steps_ahead=50):
    self._experience_queue = Queue(maxsize=steps_ahead)
    print('* starting %d workers' % processes)
    self._processes = []
    self._request_queues = []
    for i in range(processes):
      request_queue = Queue(maxsize=10)
      p = Process(target=_run_game, args=(i, game_factory(), network, device, request_queue,
                                          self._experience_queue, steps_ahead + 1,))
      p.start()
      self._request_queues.append(request_queue)
      self._processes.append(p)

  def _send_to_all(self, request, block=False):
    for request_queue in self._request_queues:
      request_queue.put(request, block=block)

  def get_experience(self) -> RunGameResponse:
    return self._experience_queue.get(block=True)

  def update_exploration_rate(self, exploration_rate):
    self._send_to_all(_RunGameRequest(set_exploration_rate=exploration_rate))

  def close(self):
    print('shutting down worker')
    self._send_to_all(_RunGameRequest(do_terminate=True))
    for p in self._processes:
      p.join()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
