import torch
import torch.nn.functional as F
from torch import Tensor, tensor

from drl.deepq.game import Experience
from drl.deepq.model import LearningModel


def _state_from_experiences(exps: [Experience], before: bool, dtype: torch.dtype, device: torch.device):
  frames = []
  for e in exps:
    if before:
      frames.append(e.state_before.as_tensor(dtype, device))
    elif not e.done:
      frames.append(e.state_after.as_tensor(dtype, device))
  return torch.stack(tuple(frames))


def get_target_action_values(model: LearningModel, gamma: float, exps: [Experience]) -> Tensor:
  non_final_mask = tensor(tuple(map(lambda e: not e.done, exps)), device=model.device, dtype=torch.uint8)
  next_states = _state_from_experiences(exps, False, dtype=model.input_dtype, device=model.device). \
    to(model.device, non_blocking=True)
  next_state_values = torch.zeros(len(exps), dtype=next_states.dtype, device=model.device)

  next_state_values[non_final_mask] = model.target_net(next_states).max(1)[0].detach()

  lengths = tensor([e.state_difference_in_steps for e in exps], dtype=next_states.dtype, device=model.device)
  gammas = torch.pow(tensor(gamma, dtype=next_states.dtype, device=model.device), lengths).detach()
  rewards = tensor([e.reward for e in exps], dtype=next_states.dtype, device=model.device)
  target_action_values = (next_state_values * gammas) + rewards
  return target_action_values.unsqueeze(1)


def calculate_losses(model: LearningModel, gamma: float, exps: [Experience]) -> Tensor:
  """
  :param model: the model
  :param timings: to record time taken
  :param gamma: discounting factor for future (next-step) rewards (e.g. 0.99)
  :param exps: Experiences calculate losses for
  """
  target_action_values = get_target_action_values(model, gamma, exps).detach()

  states = _state_from_experiences(exps, True, dtype=target_action_values.dtype, device=model.device)
  actions = torch.stack([tensor([e.action.index]) for e in exps]). \
    to(model.device, non_blocking=True)
  predicted_action_values = model.policy_net(states).gather(1, actions)

  return F.mse_loss(predicted_action_values, target_action_values, reduction='none')


def learn_from_memory(model: LearningModel, batch_size: int, gamma: float, beta: float) -> float:
  """
  :param model: the model
  :param batch_size: number of examples to process together
  :param gamma: discounting factor for future (next-step) rewards (e.g. 0.99)
  :param beta: weighting of priorized experiences [0=no correction, 1=uniform]
  :returns the (average) loss
  """
  if model.memory.size() < batch_size:
    raise ValueError('memory contains less than batch_size (%d) samples' % batch_size)

  with model.status.timings['  sample from memory']:
    sample = model.memory.sample(batch_size)
    exps = [s.experience for s in sample]
    weights = tensor([s.weight(beta) for s in sample], device=model.device)

  with model.status.timings['  forward loss']:
    losses = calculate_losses(model, gamma, exps)
    loss = torch.mean(losses * weights.type(losses.dtype))

  with model.status.timings['  backprop loss']:
    model.optimizer.zero_grad()
    loss.backward()

    model.optimizer.step()
    if torch.cuda.is_available():
      torch.cuda.synchronize()  # for the timings

  with model.status.timings['  update memory weights']:
    model.memory.update_weights(sample, losses.detach())

  return loss.item()
