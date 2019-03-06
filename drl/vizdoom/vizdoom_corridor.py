from drl.deepq.game import Action
from drl.vizdoom.game_vizdoom import VizdoomGame


class VizdoomCorridorGame(VizdoomGame):
  """Vizdoom Corridor => One corridor with oponents"""

  actions = [Action("move_left",
                    [1, 0, 0, 0, 0, 0, 0], 0),
             Action("move_right",
                    [0, 1, 0, 0, 0, 0, 0], 1),
             Action("fire",
                    [0, 0, 1, 0, 0, 0, 0], 2),
             Action("move_forward",
                    [0, 0, 0, 1, 0, 0, 0], 3),
             Action("move_backward",
                    [0, 0, 0, 0, 1, 0, 0], 4),
             Action("turn_left",
                    [0, 0, 0, 0, 0, 1, 0], 5),
             Action("turn_right",
                    [0, 0, 0, 0, 0, 0, 1], 6),
             ]

  def __init__(self, x: int, y: int, t: int, visible: bool = False):
    super().__init__('deadly_corridor', x, y, t, visible)

  @property
  def name(self) -> str:
    return "vizdoom-corridor"
