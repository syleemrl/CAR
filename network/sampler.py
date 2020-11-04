import numpy as np
import math
from regression import Regression
from IPython import embed

class Sampler(object):
	def __init__(self, sim_env, dim, path):
		self.sim_env = sim_env
		self.dim = dim
		
		self.v_mean = 0
		self.random = True

		self.k = 10
		self.k_ex = 10
		self.n_iter = 0

		self.total_iter = 0
		self.path = path

	def randomSample(self):
		return self.sim_env.UniformSampleParam()

	def probAdaptive(self, v_func, target):
		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)[0]
		return math.exp(-self.k*(v-self.v_mean)/self.v_mean)

	def probForget(self, v_func, v_func_prev, target):
		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)[0]
		v_prev = v_func_prev.getValue(target)[0]
		return math.exp((v_prev-v)*self.k)

	def update(self, v_func, v_func_prev, m=10, N=1000):
		n = 0
		self.pool = []
		for i in range(m):
			x_cur = self.randomSample()
			for j in range(int(N/m)):
				self.pool.append(x_cur)
				x_new = self.randomSample()
				alpha = min(1.0, self.probAdaptive(v_func, x_new)/self.probAdaptive(v_func, x_cur))
				#alpha = min(1.0, self.probForget(v_func, v_func_prev, x_new)/self.probForget(v_func, v_func_prev, x_cur))

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

	def isEnough(self, v_func, results):

		self.n_iter += 1
		self.total_iter += 1
		
		if self.total_iter > 15:
			self.random = False

		v_mean_cur = np.array(results).mean()
		n = len(results)

		self.v_mean = 0.6 * self.v_mean + 0.4 * v_mean_cur

		print("===========================================")
		print("mean reward : ", self.v_mean)
		print("===========================================")

		if self.n_iter >= 15:
			self.printSummary(v_func)
			return True

		return False

	def printSummary(self, v_func):
		vs = []
		cs = []
		tuples_str = []
		for _ in range(200):
			c = self.randomSample()
			c = np.reshape(c, (-1, self.dim))
			v = v_func.getValue(c)[0]
			cs.append(c[0])
			vs.append(v)
			tuples_str.append(str(c)+" "+str(v)+", ")
		if self.path is not None:
			out = open(self.path+"curriculum", "a")
			out.write(str(self.total_iter)+'\n')
			for i in range(200):
				out.write(tuples_str[i])
			out.write('\n')
			vs = np.sort(np.array(vs))

			vs_mean = vs.mean()
			vs_mean_bottom = vs[:20].mean()
			vs_mean_top = vs[180:].mean()

			out.write(str(vs_mean)+'\n')
			out.write(str(vs_mean_bottom)+'\n')
			out.write(str(vs_mean_top)+'\n')
			out.close()


	def probEx(self, v, v_mean):
		return math.exp(-self.k_ex*(v-v_mean)/v_mean)
	
	def probExEasy(self, v, v_max):
		return math.exp(-self.k_ex*abs(v/v_max - 0.9))

	def selectExGoalParameter(self, li, v_func, m=10, N=200):

		#uniform
		# t = np.random.randint(len(li))
		# return li[t][0]

		#adaptive
		v_li = []
		v_max = 1e-8

		for i in range(len(li)):
			v = np.mean(v_func.getValue(li[i][1]))
			if v > v_max:
				v_max = v
			v_li.append(v)
		v_mean = np.mean(v_li)
		pool_ex = []
		for i in range(m):
			t = np.random.randint(len(li))
			x_cur = li[t][0]
			v_cur = v_li[t]
			for j in range(int(N/m)):
				pool_ex.append(x_cur)
				t = np.random.randint(len(li))
				x_new = li[t][0]	
				v_new = v_li[t]			
	#			alpha = min(1.0, self.probExEasy(v_new, v_max)/self.probExEasy(v_cur, v_max))
				alpha = min(1.0, self.probEx(v_new, v_mean)/self.probEx(v_cur, v_mean))
				if np.random.rand() <= alpha:          
					x_cur = x_new
					v_cur = v_new
		t = np.random.randint(len(pool_ex))
		target = pool_ex[t] 
		
		for i in range(len(li)):
			if np.linalg.norm(li[i][0] - target) < 1e-5:
				v = v_li[i]
		return [target, v]