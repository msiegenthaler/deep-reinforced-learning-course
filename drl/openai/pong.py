from drl.deepq.game import Action
from drl.openai.atari_wrappers import MaxAndSkipEnv
from drl.openai.game_openai import OpenAIGame
import torchvision.transforms as T


class Pong(OpenAIGame):
  actions = [Action('up', 2, 0),
             Action('down', 3, 1)]

  def __init__(self, x: int, y: int, t: int):
    self.transform = T.Compose([T.ToPILImage(), T.Resize((y, x)), T.Grayscale(), T.ToTensor()])
    super().__init__('Pong-v0', t)
    self.env = MaxAndSkipEnv(self.env)

  @property
  def name(self) -> str:
    return 'pong'

  def _get_frame(self, env_state):
    raw = env_state[..., :3]
    image = self.transform(raw)
    return image.squeeze(0)
