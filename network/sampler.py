import numpy as np
import math
from IPython import embed

class Sampler(object):
	def __init__(self, dim, interval, boundary):
		self.dim = dim
		self.boundary = boundary
		
		self.interval = interval
		self.nBin = math.ceil((self.boundary[1] - self.boundary[0]) / self.interval)
		self.count = [1] * self.nBin
		self.count_update = [0] * self.nBin

		self.rewards_sparse = []
		self.rewards_dense = []
		for _ in range(self.nBin):
			self.rewards_sparse.append([0])
			self.rewards_dense.append([0])
	
		self.target = self.boundary[1]
		self.flag = 0

	def updateLowerBound(self, l):
		if self.boundary[0] == l:
			return

		nbin_prev = self.nBin

		self.boundary[0] = l
		self.nBin = math.ceil((self.boundary[1] - self.boundary[0]) / self.interval)

		diff = nbin_prev - self.nBin
		
		if diff > 0:
			self.rewards_sparse = self.rewards_sparse[diff:]
			self.rewards_dense = self.rewards_dense[diff:]
			self.count = self.count[diff:]
			self.count_update = self.count_update[diff:]

		elif diff < 0:
			for _ in range(-diff):
				self.rewards_sparse = [[0]] + self.rewards_sparse
				self.rewards_dense = [[0]] + self.rewards_dense
				self.count = [1] + self.count
				self.count_update = [0] + self.count_update

		print("under bound updated: ", self.boundary)
		print(self.rewards_sparse)

	def updateUpperBound(self, u):
		if self.boundary[1] == u:
			return

		nbin_prev = self.nBin

		self.boundary[1] = u
		self.nBin = math.ceil((self.boundary[1] - self.boundary[0]) / self.interval)
		diff = self.nBin - nbin_prev

		if diff < 0:
			self.rewards_sparse = self.rewards_sparse[:self.nBin]
			self.rewards_dense = self.rewards_dense[:self.nBin]
			self.count = self.count[:self.nBin]
			self.count_update = self.count_update[:self.nBin]
		elif diff > 0:
			for _ in range(diff):
				self.rewards_sparse = self.rewards_sparse + [[0]] 
				self.rewards_dense = self.rewards_dense + [[0]]
				self.count = self.count +  [1]
				self.count_update = self.count_update + [0] 

		print("upper bound updated: ", self.boundary)
		print(self.rewards_sparse)

	def updateBound(self, bound):
		self.updateUpperBound(bound[1])
		self.updateLowerBound(bound[0])

	def randomSample(self):
		t = np.random.rand(self.dim)
		target =  self.boundary[0] + (self.boundary[1] - self.boundary[0]) * t
		self.target = target
		return target

	def adaptiveSample(self):
		e_s = [np.array(self.rewards_sparse[i]).mean() for i in range(len(self.rewards_sparse))]
		e_d = [np.array(self.rewards_dense[i]).mean() for i in range(len(self.rewards_dense))]

		e = np.array(e_s) + 0.08*np.array(e_d)

		print("sparse reward per bin : ", e_s)
		print("sparse reward per bin : ", e_d)
		print("weighted reward per bin : ", e)

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
			e = np.exp(-e*0.5)
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
		idx = math.ceil(idx) - 1
		if idx > self.nBin - 1:
			idx = self.nBin - 1
		self.count[idx] += 1
		self.rewards_sparse[idx].append(rewards_sparse)
		self.rewards_dense[idx].append(rewards_dense)
		self.count_update[idx] = 1

		if self.count[idx] > 500:
			self.rewards_sparse[idx] = self.rewards_sparse[idx][100:]
			self.rewards_dense[idx] = self.rewards_dense[idx][100:]
			self.count[idx] -= 100

	def allTrained(self):
		for i in range(self.nBin):
			if self.count_update[i] == 0 or np.array(self.rewards_sparse[i]).mean() < 5.5 or np.array(self.rewards_dense[i]).mean() < 70:
				return False
		return True

	def resetCounter(self):
		for i in range(self.nBin):
			self.count_update[i] = 0
