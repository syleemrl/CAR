import numpy as np
import math
from regression import Regression
from IPython import embed

class Sampler(object):
	def __init__(self, sim_env, dim):
		self.sim_env = sim_env
		self.dim = dim
		
		self.v_mean = 0
		self.random = True

		self.k = 5
		self.n_iter = 0

	def randomSample(self):
		return self.sim_env.UniformSampleParam()

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

	def adaptiveSample(self):
		if self.random:
			return self.randomSample()

		t = np.random.randint(len(self.pool))
		target = self.pool[t] 

		return target

	def reset(self):
		self.n_iter = 0
		self.random = True
		self.v_mean == 0

	def isEnough(self, results):

		self.n_iter += 1
		self.random = False

		v_mean_cur = np.array(results).mean()
		n = len(results)

		self.v_mean = 0.6 * self.v_mean + 0.4 * v_mean_cur

		print("===========================================")
		print("mean reward : ", self.v_mean)
		print("===========================================")

		if self.n_iter < 3 or self.v_mean < 8.3:
			return False
		return True