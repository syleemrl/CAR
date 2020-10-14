import numpy as np
import math
from regression import Regression
from IPython import embed

class Sampler(object):
	def __init__(self, base, unit):
		self.unit = unit
		self.base = base
		self.dim = len(base)

		self.bound = []
		
		self.v_mean = 0
		self.n_samples = 0
		self.random = True

		self.k = 10
		self.n_iter = 0

	def updateBound(self, bound):
		self.bound = bound

		print("num new bound: ", len(self.bound))

	def randomSample(self):
		t = np.random.randint(len(self.bound))
		idx = self.bound[t]
		target = self.base + (idx + np.random.uniform(0, 1, len(self.unit))) * self.unit

		return target

	def prob(self, v_func, target):
		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)[0]
		return math.exp(-self.k*(v-self.v_mean)/self.v_mean)

	def update(self, v_func, m=10, N=1000):
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
	
		t = []
		count = []
		for i in range(len(self.bound)):
			t.append(self.base + self.bound[i] * self.unit)
			count.append(0)
		v = v_func.getValue(t)

		# for i in range(len(self.pool)):
		# 	x = self.pool[i]
		# 	idx = []
		# 	for j in range(self.dim):
		# 		idx.append(math.floor((x[j] - self.base[j]) / self.unit[j]))
		# 	for j in range(len(self.bound)):
		# 		if np.linalg.norm(idx - self.bound[j]) < 1e-2:
		# 			count[j] += 1
		# 			break

		print("prob: ", np.exp(-self.k*(v-self.v_mean)/self.v_mean))
		# print("sample: ",end=' ')
		# for i in range(len(v)):
		# 	print(v[i], count[i], end=', ')
		# print()

	def adaptiveSample(self):
		if self.random:
			return self.randomSample()

		t = np.random.randint(len(self.pool))
		target = self.pool[t] 

		return target

	def reset(self):
		self.n_iter = 0
		self.n_samples = 0
		self.random = True

	def isEnough(self, results):

		self.n_iter += 1
		self.random = False

		v_mean_cur = np.array(results).mean()
		n = len(results)

		self.v_mean = (n * v_mean_cur + self.n_samples * self.v_mean) / (n + self.n_samples)
		if self.v_mean == 0:
			embed()

		self.n_samples += n

		print("===========================================")
		print("mean reward : ", self.v_mean)
		print("===========================================")

		if self.n_iter < 2 or self.v_mean < 7.5:
			return False
		return True