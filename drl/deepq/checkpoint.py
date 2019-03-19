import os

import torch

from drl.deepq.model import LearningModel
from drl.utils.timings import Timings


def save_checkpoint(model: LearningModel) -> str:
  """
  :return: filename the model was saved to
  """
  data = {
    'game_name': model.game_name,
    'strategy_name': model.strategy_name,
    'status_training_epochs': model.status.training_epochs,
    'status_training_episodes': model.status.training_episodes,
    'status_validation_episodes': model.status.validation_episodes,
    'steps': model.status.trained_for_steps,
    'optimizer_state_dict': model.optimizer.state_dict(),
    'model_state_dict': model.policy_net.state_dict(),
    'model_description': str(model.policy_net),
    'timings': model.status.timings
  }
  file = 'checkpoints/%s-%s-%04d.pt' % (model.game_name, model.strategy_name, model.status.trained_for_epochs)
  torch.save(data, file)
  last_file = 'checkpoints/%s-%s-last.pt' % (model.game_name, model.strategy_name)
  torch.save(data, last_file)
  print(' - saved checkpoint to', file)
  return file


def load_checkpoint(model: LearningModel, suffix: str = 'last') -> bool:
  file = 'checkpoints/%s-%s-%s.pt' % (model.game_name, model.strategy_name, suffix)
  if os.path.isfile(file):
    data = torch.load(file, map_location=model.device)
    assert (model.game_name == data['game_name'])
    if model.strategy_name != data['strategy_name']:
      print('Warning, loading checkpoint from strategy %s' % data['strategy_name'])
    model.status._training_epochs = data['status_training_epochs'] if 'status_training_epochs' in data else []
    model.status._training_episodes = data['status_training_episodes'] if 'status_training_episodes' in data else []
    model.status._validation_episodes = data['status_validation_episodes'] if 'status_validation_episodes' in data else []
    model.status.timings = data['timings'] if 'timings' in data else Timings()
    model.optimizer.load_state_dict(data['optimizer_state_dict'])
    model.policy_net.load_state_dict(data['model_state_dict'])
    model.target_net.load_state_dict(data['model_state_dict'])
    print('loaded checkpoint from', file)
    return True
  else:
    return False
