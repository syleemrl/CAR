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
		self.n_learning = 0
		self.path = path

		self.start = 0

	def randomSample(self, visited=True):
		return self.sim_env.UniformSample(visited)
		
	def probAdaptive(self, v_func, target, hard=True):
		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)[0]
		if hard:
			return math.exp(- self.k * (v - self.v_mean) / self.v_mean) + 1e-10
		else:
			return math.exp(self.k * (v - self.v_mean) / self.v_mean) + 1e-10

	def probTS(self, v_func, v_func_prev, target, hard=True):
		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)[0]
		v_prev = v_func_prev.getValue(target)[0]
		if hard:
			slope = (v_prev - v) / v_prev * self.k * 2
		else:
			slope = (v - v_prev) / v_prev * self.k * 2
		if slope > 10:
			slope = 10
		return math.exp(slope) + 1e-10

	def probAdaptiveSampling(self, idx):
		v = self.v_sample[idx]
		return math.exp(self.k * (v - self.v_mean) / self.v_mean) + 1e-10

	def probTSSampling(self, idx):
		v = self.v_sample[idx]
		v_prev = self.v_prev_sample[idx]
		slope = (v - v_prev) / v_prev * self.k * 2
		if slope > 10:
			slope = 10
		return math.exp(slope) + 1e-10


	def updateGoalDistribution(self, v_func, v_func_prev, results, visited, 
		network=True, idxs=[], m=10, N=1000):
		self.start += 1
		self.v_mean_cur = np.array(results).mean()
		self.v_mean = 0.6 * self.v_mean + 0.4 * self.v_mean_cur

		if visited:
			self.pool = []
		else:
			self.pool_ex = []
			self.idx_ex = []
		if network:
			for i in range(m):
				x_cur = self.randomSample(visited)
				for j in range(int(N/m)):
					if visited:
						self.pool.append(x_cur)
					else:
						self.pool_ex.append(x_cur)

					x_new = self.randomSample(visited)
					if visited:
						alpha = min(1.0, self.probAdaptive(v_func, x_new, visited)/self.probAdaptive(v_func, x_cur, visited))
					else:
						alpha = min(1.0, self.probTS(v_func, v_func_prev, x_new, visited)/self.probTS(v_func, v_func_prev, x_cur, visited))

					if np.random.rand() <= alpha:          
						x_cur = x_new
		else:
			if visited:
				print('not supported for visited')
				return

			v_mean_sample_cur = [0] * len(self.pool_ex)
			count_sample_cur = [0] * len(self.pool_ex)
			for i in range(len(results)):
				v_mean_sample_cur[idxs[i]] += results[i]
				count_sample_cur[idxs[i]] += 1
			
			for i in range(len(self.pool_ex)):
				self.v_prev_sample[i] = self.v_sample[i]
				self.v_sample[i] = 0.6 * self.v_mean_sample[i] + 0.4 * v_mean_sample_cur[i] / count_sample_cur[i]
			print('v prev goals: ', self.v_prev_sample)
			print('v goals: ', self.v_sample)

			for i in range(m):
				t_cur = np.random.randint(len(self.sample))
				x_cur = self.sample[t_cur]
				for j in range(int(N/m)):
					self.pool_ex.append(x_cur)
					self.idx_ex.append(t_cur)

					t_new = np.random.randint(len(self.sample))
					x_new = self.sample[t_new]
					alpha = min(1.0, self.probAdaptiveSampling(t_new)/self.probAdaptiveSampling(t_cur))
					# alpha = min(1.0, self.probTSSampling(t_new)/self.probTSSampling(t_cur))

					if np.random.rand() <= alpha:          
						x_cur = x_new
						t_cur = t_new

	def sampleGoals(self, m=10):
		self.sample = []
		for i in range(m):
			self.sample.append(self.randomSample(False))
		print('new goals: ', self.sample)

	def adaptiveSample(self, visited, network=True):
		if not network:
			t = np.random.randint(len(self.pool_ex))	
			target = self.pool_ex[t]
			idx = idx_ex[t]
			return target, idx

		if self.random or self.start < 10:
			return self.randomSample(visited), -1
		if visited and (self.n_iter % 5) == 0:
			return self.randomSample(visited), -1

		if visited and len(self.pool) != 0:
			t = np.random.randint(len(self.pool))
			target = self.pool[t] 
		elif not visited and len(self.pool_ex) != 0:
			t = np.random.randint(len(self.pool_ex))
			target = self.pool_ex[t]
		else:
			return self.randomSample(visited), -1

		return target, t

	def reset(self):
		self.n_iter = 0
		self.random = True
		self.v_mean == 0
		self.n_learning += 1

	def isEnough(self, v_func):
		
		self.random = False

		print("===========================================")
		print("mean reward : ", self.v_mean)
		print("===========================================")

		if self.n_iter % 5 == 0:
			self.printSummary(v_func)

		if self.n_iter >= 5 and (self.v_mean >= 3.75 or self.v_mean_cur > 3.75):
			return True

		self.n_iter += 1
		self.total_iter += 1
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
			tuples_str.append(str(c[0])+" "+str(v)+", ")
		if self.path is not None:
			out = open(self.path+"curriculum"+str(self.n_learning), "a")
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
