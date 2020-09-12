import numpy as np
import math
from IPython import embed

class Sampler(object):
	def __init__(self, dim, base, unit):
		self.dim = dim
		self.unit = unit
		self.base = base
		self.rewards_dense = []
		self.rewards_sparse = []
		self.bound = []

	def updateBound(self, bound):
		bound_prev = self.bound
		self.bound = bound
		
		diff = len(self.bound) - len(bound_prev)
		if diff > 0:
			for _ in range(diff):
				self.rewards_dense.append([0])
				self.rewards_sparse.append([0])
			for i in range(len(self.bound)):
				target = self.base + self.bound[i] * self.unit
		print("num new bound: ", len(self.bound))

	def randomSample(self):
		t = np.random.randrange(len(self.bound))
		idx = self.bound[t]
		target = self.base + (idx + np.random.uniform(0, 1, len(self.unit))) * self.unit

		return target

	def adaptiveSample(self):

		t_index = len(self.bound) - 1
		cur = 0
		ran = np.random.uniform(0.0, 1.0)
		
		for i in range(len(self.bound)):
			cur += self.e[i]
			if ran < cur:
				t_index = i
				break

		target = self.base + (self.bound[t_index] + np.random.uniform(0, 1, len(self.unit))) * self.unit
		self.target_idx = t_index

		return target

	def saveResults(self, rewards_sparse, rewards_dense):

		self.rewards_sparse[self.target_idx].append(rewards_sparse)
		self.rewards_dense[self.target_idx].append(rewards_dense)				

		if len(self.rewards_sparse[self.target_idx]) > 500:
			self.rewards_sparse[self.target_idx] = self.rewards_sparse[self.target_idx][100:]
			self.rewards_dense[self.target_idx] = self.rewards_dense[self.target_idx][100:]

	def allTrained(self):
		if self.min_s.mean() < 6.5 or self.min_d.mean() < 40:
			return False
		return True

	def resetCounter(self):
		self.rewards_dense = []
		self.rewards_sparse = []
		for _ in range(len(self.bound)):
			self.rewards_dense.append([0])
			self.rewards_sparse.append([0])
		self.updateStatus()

	def updateStatus(self):
		self.e_s = [np.array(self.rewards_sparse[i]).mean() for i in range(len(self.rewards_sparse))]
		self.e_d = [np.array(self.rewards_dense[i]).mean() for i in range(len(self.rewards_dense))]

		if len(self.e_s) > 10:
			n = round(len(self.e_s) * 0.1)
		else:
			n = len(self.e_s)
		self.min_s = np.sort(np.array(self.e_s))[:n]
		self.min_d = np.sort(np.array(self.e_d))[:n]

		self.e = np.array(self.e_s) + 0.08*np.array(self.e_d)

		# print("weighted reward per bin : ", e)

		self.e = np.array([(r - self.e.min()) for r in self.e])
		self.e = np.exp(-self.e)
		tot = self.e.sum()
		self.e = self.e / tot

		print("===========================================")
		print("mean sparse reward per bin : ", np.array(self.e_s).mean())
		print("mean dense reward per bin : ", np.array(self.e_d).mean())
		print("min sparse reward per bin : ", self.min_s)
		print("min dense reward per bin : ", self.min_d)
		print("probability of each bin : ", self.e)

		print("===========================================")
