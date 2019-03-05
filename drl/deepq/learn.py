from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor
from torch.optim import Optimizer

from drl.deepq.game import Experience, Game
from drl.deepq.replay_memory import ReplayMemory
from drl.utils.timings import timings

class TrainingStatus:
  def __init__(self):
    self.trained_for_epochs = 0
    self.trained_for_steps = 0
    self.trained_for_episodes = 0

class LearningModel(NamedTuple):
  target_net: nn.Module
  policy_net: nn.Module
  optimizer: Optimizer
  game: Game
  memory: ReplayMemory
  device: object  # Device to train on
  strategy_name: str
  status: TrainingStatus = TrainingStatus()


def get_target_action_values(model: LearningModel, gamma: float, exps: [Experience]) -> Tensor:
  """
  :param gamma: discounting factor for future (next-step) rewards (e.g. 0.99)
  :param exps: Experiences to calculate target action values for
  """
  next_states = torch.stack([e.state_after for e in exps if not e.done]).to(model.device)
  non_final_mask = tensor(tuple(map(lambda e: not e.done, exps)), device=model.device, dtype=torch.uint8)
  next_state_values = torch.zeros(len(exps), device=model.device)
  next_state_values[non_final_mask] = model.target_net(next_states).max(1)[0].detach()

  rewards = tensor([e.reward for e in exps], device=model.device)
  target_action_values = (next_state_values * gamma) + rewards
  return target_action_values.unsqueeze(1)


def calculate_losses(model: LearningModel, gamma: float, exps: [Experience]) -> Tensor:
  """
  :param gamma: discounting factor for future (next-step) rewards (e.g. 0.99)
  :param exps: Experiences calculate losses for
  """
  target_action_values = get_target_action_values(model, gamma, exps).detach()

  states = torch.stack([e.state_before for e in exps])
  states = states.to(model.device)
  actions = torch.stack([tensor([e.action.index]) for e in exps])
  actions = actions.to(model.device)
  predicted_action_values = model.policy_net(states).gather(1, actions)

  return F.mse_loss(predicted_action_values, target_action_values, reduction='none')


def learn_from_memory(model: LearningModel, batch_size: int, gamma: float, beta: float) -> float:
  """
  :param gamma: discounting factor for future (next-step) rewards (e.g. 0.99)
  :param beta: weighting of priorized experiences [0=no correction, 1=uniform]
  :returns the (average) loss
  """
  if model.memory.size() < batch_size:
    raise ValueError('memory contains less than batch_size (%d) samples' % batch_size)

  with timings['sample from memory']:
    sample = model.memory.sample(batch_size)
    exps = [s.experience for s in sample]
    weights = tensor([s.weight(beta) for s in sample])
    weights = weights.to(model.device)

  with timings['forward loss']:
    losses = calculate_losses(model, gamma, exps)
    loss = torch.mean(losses * weights)

  with timings['backprop loss']:
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    if torch.cuda.is_available():
      torch.cuda.synchronize()  # for the timings

  with timings['update memory weights']:
    model.memory.update_weights(sample, losses.detach())

  return loss.item()
