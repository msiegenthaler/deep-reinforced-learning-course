import os

import torch

from drl.deepq.model import LearningModel


def save_checkpoint(model: LearningModel) -> str:
  """
  :return: filename the model was save to
  """
  data = {
    'status_trainingLog': model.status.training_log,
    'status_validationLog': model.status.validation_log,
    'steps': model.status.trained_for_steps,
    'optimizer_state_dict': model.optimizer.state_dict(),
    'model_state_dict': model.policy_net.state_dict(),
    'model_description': str(model.policy_net)
  }
  file = 'checkpoints/%s-%s-%04d.pt' % (model.game.name, model.strategy_name, model.status.trained_for_epochs)
  torch.save(data, file)
  last_file = 'checkpoints/%s-%s-last.pt' % (model.game.name, model.strategy_name)
  torch.save(data, last_file)
  print(' - saved checkpoint to', file)
  return file


def load_checkpoint(model: LearningModel, suffix: str = 'last') -> bool:
  file = 'checkpoints/%s-%s-%s.pt' % (model.game.name, model.strategy_name, suffix)
  if os.path.isfile(file):
    data = torch.load(file, map_location=model.device)
    model.status.training_log = data['status_trainingLog']
    model.status.validation_log = data['status_validationLog']
    model.status.timings.reset()
    model.optimizer.load_state_dict(data['optimizer_state_dict'])
    model.policy_net.load_state_dict(data['model_state_dict'])
    model.target_net.load_state_dict(data['model_state_dict'])
    print('loaded checkpoint from', file)
    return True
  else:
    return False
