import math

import numpy as np


class SumTree:
  """SumTree with priorities. Allow for fast getting (sampling) at specific (priority) positions"""

  def __init__(self, capacity: int):
    self.capacity = capacity
     # [--------------Parent nodes (n-1)----------][-------leaves to recode priority (n) -----]
    self.tree = np.zeros(2 * capacity - 1)
    self.data = np.zeros(capacity, dtype=object)
    self.pointer = 0
    self.leaf_offset = capacity - 1
    self.at_max_capacity = False

  def add(self, data, priority: float) -> None:
    self.data[self.pointer] = data
    self.update(self.pointer + self.leaf_offset, priority)
    self.pointer += 1
    if self.pointer >= self.capacity:
      self.at_max_capacity = True
      self.pointer = 0 # overwrite oldest if full

  def total_priority(self) -> float:
    """sum of all item priorities"""
    return self.tree[0] # root node

  def max_prio(self) -> float:
    """biggest item priority"""
    if self.size() == 0: return 0.
    else: return np.max(self._valid_leafs())

  def min_prio(self) -> float:
    """smallest item priority"""
    if self.size() == 0: return 0.
    else: return np.min(self._valid_leafs())

  def size(self) -> int:
    """current number of items in the sumtree"""
    if self.at_max_capacity: return self.capacity
    else: return self.pointer

  def get(self, priority_position: float) -> (float, object, int):
    """
    :param priority_position value between 0 and total_priority
    :returns (item priority, item data, tree index of the item)
    """
    tree_index, data_index = self._get_leaf_position(priority_position)
    return self.tree[tree_index], self.data[data_index], tree_index

  def update(self, tree_index: int, new_priority: float) -> None:
    if math.isnan(new_priority) or math.isinf(new_priority): new_priority = 1.0
    delta = new_priority - self.tree[tree_index]
    self.tree[tree_index] = new_priority
    while tree_index != 0:
      tree_index = (tree_index - 1) // 2 # parent node
      self.tree[tree_index] += delta

  def _valid_leafs(self):
    if self.at_max_capacity: return self.tree[-self.capacity:]
    else: return self.tree[self.leaf_offset:self.leaf_offset+self.pointer]

  def _get_leaf_position(self, prio_position):
    tree_index = 0 #start at root node
    while tree_index < self.leaf_offset: # stop when we are at leaf level
      index_left_child = 2 * tree_index + 1
      index_right_child = index_left_child + 1

      prio_left_child = self.tree[index_left_child]
      if prio_position <= prio_left_child:
        tree_index = index_left_child
      else:
        prio_position -= prio_left_child
        tree_index = index_right_child
    data_index = tree_index - self.leaf_offset
    if (data_index >= self.size()):
      # basically the prio position was too large, just return the last element
      data_index = self.size() - 1
      tree_index = data_index + self.leaf_offset
    return tree_index, data_index
