import torch
import torch.nn as nn


def conv2d_size_out(size, kernel_size=4, stride=2):
  return (size - (kernel_size - 1) - 1) // stride + 1


class DQN(nn.Module):
  def __init__(self, w: int, h: int, t: int, action_count: int):
    """
    :param w: width of the image
    :param h: height of the image
    :param t: number of images in the frame stack (temporal history)
    :param action_count: number of actions/action combinations in the game
    """
    super(DQN, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(t, 32, kernel_size=8, stride=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=4, stride=2),
      nn.BatchNorm2d(128),
      nn.ReLU()
    )

    conv_out_w = conv2d_size_out(
      conv2d_size_out(conv2d_size_out(w, kernel_size=8)))
    conv_out_h = conv2d_size_out(
      conv2d_size_out(conv2d_size_out(h, kernel_size=8)))
    linear_in = conv_out_w * conv_out_h * 128

    self.action_count = action_count
    self.linear = nn.Sequential(
      nn.Linear(linear_in, 512),
      nn.ReLU(),
      nn.Linear(512, action_count))

  def forward(self, state):
    r = self.conv(state)
    r = r.view(r.size(0), -1)
    r = self.linear(r)
    return r


class DuelingDQN(nn.Module):
  def __init__(self, w: int, h: int, t: int, action_count: int):
    """
    :param w: width of the image
    :param h: height of the image
    :param t: number of images in the frame stack (temporal history)
    :param action_count: number of actions/action combinations in the game
    """
    super(DuelingDQN, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(t, 32, kernel_size=8, stride=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=4, stride=2),
      nn.BatchNorm2d(128),
      nn.ReLU()
    )

    conv_out_w = conv2d_size_out(
      conv2d_size_out(conv2d_size_out(w, kernel_size=8)))
    conv_out_h = conv2d_size_out(
      conv2d_size_out(conv2d_size_out(h, kernel_size=8)))
    linear_in = conv_out_w * conv_out_h * 128

    self.action_count = action_count

    self.state_value_linear = nn.Sequential(
      nn.Linear(linear_in, 512),
      nn.ReLU(),
      nn.Linear(512, 1)
    )
    self.action_value_linear = nn.Sequential(
      nn.Linear(linear_in, 512),
      nn.ReLU(),
      nn.Linear(512, action_count)
    )

  def forward(self, state):
    r = self.conv(state)
    r = r.view(r.size(0), -1)
    state_value = self.state_value_linear(r)
    action_value = self.action_value_linear(r)
    value = state_value + action_value - torch.mean(action_value, 1, keepdim=True)
    return value
