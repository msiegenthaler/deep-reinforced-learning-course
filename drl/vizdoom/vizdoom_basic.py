from drl.deepq.game import Action
from drl.vizdoom.game_vizdoom import VizdoomGame


class VizdoomBasicGame(VizdoomGame):
  """Vizdoom Basic => One room with a single non-moving opponent"""

  actions = [Action("left", [1, 0, 0], 0),
             Action("right", [0, 1, 0], 1),
             Action("fire", [0, 0, 1], 2)]

  def __init__(self, x: int, y: int, t: int, visible: bool = False):
    super().__init__('simpler_basic', x, y, t, visible)

  @property
  def name(self) -> str:
    return "vizdoom-basic"
