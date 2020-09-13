import numpy as np
import math
from regression import Regression
from IPython import embed

class Sampler(object):
	def __init__(self, base, unit):
		self.unit = unit
		self.base = base
		self.rewards_dense = []
		self.rewards_sparse = []
		self.bound = []
		
		self.v_mean = 0
		self.n_samples = 0
		self.random = True

	def updateBound(self, bound):
		self.bound = bound

		print("num new bound: ", len(self.bound))

	def randomSample(self):
		t = np.random.randrange(len(self.bound))
		idx = self.bound[t]
		target = self.base + (idx + np.random.uniform(0, 1, len(self.unit))) * self.unit

		return target

	def prob(self, v_func, target):
		v = v_func.run(target)[0]
		return math.exp(-self.k*(v-self.v_mean)/self.v_mean)

	def update(self, v_func, m=5, N=100):
		n = 0
		self.pool = []
		for i in range(m):
			x_cur = self.randomSample()
			for j in range(int(N/m)):
				self.pool.append(x_cur)
				x_new = self.randomSample()
				alpha = min(1.0, self.prob(v_func, x_new)/self.prob(v_func, x_cur))
				if np.random.rand() <= alpha:          
					x_cur = x_new
	

	def adaptiveSample(self):
		if self.random:
			return self.randomSample()

		t = np.random.randrange(len(self.pool))
		target = self.pool[t] 

		return target

	def reset(self):
		self.v_mean = 0
		self.n_samples = 0
		self.random = True

	def isEnough(self, results):

		v_mean_cur = np.array(results).mean()
		n = len(results)

		self.v_mean = (n * v_mean_cur + self.n_samples * self.v_mean) / (n + self.n_samples)
		self.n_samples += n

		self.random = False

		print("===========================================")
		print("min reward per bin : ", self.v_mean)
		print("===========================================")

		if self.v_mean < 40:
			return False
		return True