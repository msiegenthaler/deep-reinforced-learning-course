import torch
import torch.nn.functional as F
from torch import Tensor, tensor

from drl.deepq.game import Experience
from drl.deepq.model import LearningModel
from drl.utils.timings import Timings


def _state_from_experiences(exps: [Experience], before: bool):
  frames = []
  frame_shape = exps[0].state_before.frames[0].shape
  count = 0
  for e in exps:
    if before:
      count += 1
      if e.state_before.on_device is not None:
        frames.append(e.state_before.on_device)
      else:
        frames.extend(e.state_before.frames)
    elif not e.done:
      count += 1
      if e.state_after.on_device is not None:
        frames.append(e.state_after.on_device)
      else:
        frames.extend(e.state_after.frames)
  state = torch.cat(tuple(frames))
  return state.reshape((count, -1, *frame_shape))


def get_target_action_values(model: LearningModel, timings: Timings, gamma: float, exps: [Experience]) -> Tensor:
  with timings['      transfer target states']:
    non_final_mask = tensor(tuple(map(lambda e: not e.done, exps)), device=model.device, dtype=torch.uint8)
    next_states = _state_from_experiences(exps, False).to(model.device, non_blocking=True)
    next_state_values = torch.zeros(len(exps), device=model.device)

  with timings['      calculate Q next']:
    next_state_values[non_final_mask] = model.target_net(next_states).max(1)[0].detach()

  with timings['      calculate Q target']:
    lengths = tensor([e.state_difference_in_steps for e in exps]).to(model.device, non_blocking=True)
    gammas = torch.pow(tensor(gamma).to(model.device, non_blocking=True), lengths.float()).detach()
    rewards = tensor([e.reward for e in exps], device=model.device)
    target_action_values = (next_state_values * gammas) + rewards
    return target_action_values.unsqueeze(1)


def calculate_losses(model: LearningModel, timings: Timings, gamma: float, exps: [Experience]) -> Tensor:
  """
  :param model: the model
  :param timings: to record time taken
  :param gamma: discounting factor for future (next-step) rewards (e.g. 0.99)
  :param exps: Experiences calculate losses for
  """
  with timings['    get target action value']:
    target_action_values = get_target_action_values(model, model.status.timings, gamma, exps).detach()

  with timings['    predicted action values']:
    with timings['      transfer states']:
      states = _state_from_experiences(exps, True). \
        to(model.device, non_blocking=True)
      actions = torch.stack([tensor([e.action.index]) for e in exps]).\
        to(model.device, non_blocking=True)
    with timings['      run network']:
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
    weights = tensor([s.weight(beta) for s in sample])
    weights = weights.to(model.device, non_blocking=True)

  with model.status.timings['  forward loss']:
    losses = calculate_losses(model, model.status.timings, gamma, exps)
    loss = torch.mean(losses * weights)

  with model.status.timings['  backprop loss']:
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    if torch.cuda.is_available():
      torch.cuda.synchronize()  # for the timings

  with model.status.timings['  update memory weights']:
    model.memory.update_weights(sample, losses.detach())

  return loss.item()
