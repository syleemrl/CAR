import numpy as np
import math
from IPython import embed

class Sampler(object):
	def __init__(self, dim, bin, boundary):
		self.dim = dim
		self.boundary = boundary
		
		self.nBin = bin
		self.interval = (self.boundary[1] - self.boundary[0]) / self.nBin
		self.count = [1] * self.nBin
		self.rewards_sparse = []
		self.rewards_dense = []
		for _ in range(self.nBin):
			self.rewards_sparse.append([0])
			self.rewards_dense.append([0])
	
		self.target = self.boundary[1]
		self.flag = 0
	def randomSample(self):
		t = np.random.rand(self.dim)
		target =  self.boundary[0] + (self.boundary[1] - self.boundary[0]) * t
		self.target = target
		return target

	def adaptiveSample(self):
		e = [np.array(self.rewards_sparse[i]).mean() for i in range(len(self.rewards_sparse))]
		e = np.array(e)

		print("sparse reward per bin : ", e)

		if self.flag == 0:
			flag_count = 0
			for i in range(len(self.count)):
				if self.count[i] < 50:
					flag_count += 1
					break
			if flag_count == 0:
				self.flag = 1
			e = np.ones(e.shape)
		else:
			e = np.array([(r - e.min()) for r in e])
			e = np.exp(-e*0.75)
		tot = e.sum()
		e = e / tot
		print("probability of each bin : ", e)

		t_index = self.nBin - 1
		cur = 0
		ran = np.random.uniform(0.0, 1.0)
		
		for i in range(self.nBin):
			cur += e[i]
			if ran < cur:
				t_index = i
				break

		target = self.boundary[0] + self.interval * (t_index + np.random.uniform(0.0, 1.0))
		self.target = target

		print("new target : ", target)

		return [target]

	def saveResults(self, rewards_sparse, rewards_dense):
		idx = (self.target - self.boundary[0]) / self.interval
		idx = math.floor(idx)
		if idx > self.nBin - 1:
			idx = self.nBin - 1
		self.count[idx] += 1
		self.rewards_sparse[idx].append(rewards_sparse)
		self.rewards_dense[idx].append(rewards_dense)

		if self.count[idx] > 100:
			self.rewards_sparse[idx] = self.rewards_sparse[idx][50:]
			self.rewards_dense[idx] = self.rewards_dense[idx][50:]
			self.count[idx] -= 50