import sys

from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right
from collections import OrderedDict


class CustomMultiStepLR(_LRScheduler):
	""" Set the learning rate of each parameter group to the given values once the number of epoch reaches one of the milestones.

	Args:
		optimizer (Optimizer): Wrapped optimizer.
		baselr (float): Initial value.
		milestones (dict): Dictionary in the form {n_epoch (int) : lr (float)}
		last_epoch (int): The index of last epoch. Default: -1.

	Example:
		>>> # Assuming optimizer uses lr = 0.05 for all groups
		>>> # lr = 0.001     if epoch < 1000
		>>> # lr = 0.0005    if 1000 <= epoch < 2000
		>>> # lr = 0.00001   if epoch >= 2000
		>>> milestones = {1000 : 0.0005, 2000 : 0.0001}
		>>> scheduler = CustomMultiStepLR(optimizer, baselr=0.001, milestones=milestones)
		>>> for epoch in range(3000):
		>>>     train(...)
		>>>     validate(...)
		>>>     scheduler.step()
	"""

	def __init__(self, optimizer, baselr, milestones, last_epoch=-1):
		if not list(milestones) == sorted(milestones):
			raise ValueError("Milestones should be a list of increasing integers. Got {}", milestones)
		self.milestones = milestones
		self.milestone_values = list(milestones.values())
		self.milestone_values.insert(0, baselr)
		self.milestone_keys = list(milestones.keys())
		super(CustomMultiStepLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		return [self.milestone_values[bisect_right(self.milestone_keys, self.last_epoch)] for base_lr in self.base_lrs]
