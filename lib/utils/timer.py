# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import time
import numpy as np


class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.average_time = 0.
    self.all_times = []


  def tic(self):
    self.start_time = time.time()


  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.all_times.append(self.diff)
    self.total_time += self.diff
    self.calls += 1
    self.average_time = self.total_time / self.calls
    if average:
      return self.average_time
    else:
      return self.diff


  def get_avg(self, last_nbr_iter=None):
    if last_nbr_iter is None or self.calls <= last_nbr_iter:
      return self.average_time
    else:
      return np.mean(np.asarray(self.all_times[self.calls - last_nbr_iter:]))