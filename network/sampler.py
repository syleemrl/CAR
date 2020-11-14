import numpy as np
import math
from regression import Regression
from IPython import embed
from copy import copy
class Sampler(object):
	def __init__(self, sim_env, dim, path, explore, visit, egreedy):
		self.sim_env = sim_env
		self.dim = dim
		
		self.v_mean = 0
		self.random = True

		self.k = 15
		self.k_ex = 20

		self.total_iter = 0
		self.n_learning = 0
		self.path = path

		self.start = 0
		# 0: uniform 1: adaptive 2: ts
		self.type_visit = visit
		# 0: uniform 1 :adaptive(network) 2:adaptive(sampling) 3:ts(network) 4:ts(sampling) 5: uniform(sampling) 6: num sample slope(sampling) 7:num sample near goal(sampling)
		self.type_explore = explore
		if egreedy:
			self.epsilon_greedy = True
		else:
			self.epsilon_greedy = False
		self.epsilon = 0.2
		print('=======================================')
		print('curriculum option')
		print('type visit', self.type_visit)
		print('type explore', self.type_explore)
		print('e greedy', self.epsilon_greedy)
		print('=======================================')

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
		return math.exp(self.k_ex * (v - self.v_mean) / self.v_mean) + 1e-10

	def probTSSampling(self, idx):
		v = self.v_sample[idx]
		v_prev = self.v_prev_sample[idx]
		slope = (v - v_prev) / v_prev * self.k_ex * 2
		if slope > 10:
			slope = 10
		return math.exp(slope) + 1e-10

	def probTS2Sampling(self, idx):
		v = self.ns_slope_sample[idx]
		mean = np.array(self.ns_slope_sample).mean() + 1e-8
		return math.exp((v - mean) / mean) + 1e-10

	def updateNumSampleDelta(self, idx):
		if self.type_explore != 6 and self.type_explore != 7:
			return
		if self.type_explore == 6:
			slope = max(0.0, self.sim_env.GetNumSamples() - self.prev_ns)
		else:
			slope = self.sim_env.GetNewSamplesNearGoal()
		self.ns_slope_temp[idx] += slope
		self.ns_count_temp[idx] += 1

	def updateGoalDistribution(self, v_func, v_func_prev, results, idxs, visited, m=10, N=1000):
		self.start += 1
		if visited:
			self.n_visit += 1
		else:
			self.n_explore += 1
			
		self.v_mean_cur = np.array(results).mean()
		self.v_mean = 0.6 * self.v_mean + 0.4 * self.v_mean_cur

		if visited:
			if self.type_visit == 0:
				return
			self.pool = []
			for i in range(m):
				x_cur = self.randomSample(visited)
				for j in range(int(N/m)):
					self.pool.append(x_cur)
					x_new = self.randomSample(visited)
					if self.type_visit == 1:
						alpha = min(1.0, self.probAdaptive(v_func, x_new, True)/self.probAdaptive(v_func, x_cur, True))
					else:
						alpha = min(1.0, self.probTS(v_func, v_func_prev, x_new, True)/self.probTS(v_func, v_func_prev, x_cur, True))

					if np.random.rand() <= alpha:          
						x_cur = x_new
		else:
			if self.type_explore == 0 or self.type_explore == 5:
				return
			self.pool_ex = []
			self.idx_ex = []
			if self.type_explore == 1 or self.type_explore == 3:
				for i in range(m):
					x_cur = self.randomSample(visited)
					for j in range(int(N/m)):
						self.pool_ex.append(x_cur)
						x_new = self.randomSample(visited)
						if self.type_explore == 1:
							alpha = min(1.0, self.probAdaptive(v_func, x_new, False)/self.probAdaptive(v_func, x_cur, False))
						else:
							alpha = min(1.0, self.probTS(v_func, v_func_prev, x_new, False)/self.probTS(v_func, v_func_prev, x_cur, False))

						if np.random.rand() <= alpha:          
							x_cur = x_new
			else:
				if self.n_explore == 1:
					return
				v_mean_sample_cur = [0] * len(self.sample)
				count_sample_cur = [0] * len(self.sample)

				for i in range(len(results)):
					v_mean_sample_cur[idxs[i]] += results[i]
					count_sample_cur[idxs[i]] += 1
					
				for i in range(len(self.sample)):
					if count_sample_cur[i] != 0:
						self.v_prev_sample[i] = copy(self.v_sample[i])
						self.v_sample[i] = 0.6 * self.v_sample[i] + 0.4 * v_mean_sample_cur[i] / count_sample_cur[i]

				print('v prev goals: ', self.v_prev_sample)
				print('v goals: ', self.v_sample)
				if self.type_explore == 6 or self.type_explore == 7:
					for i in range(len(self.sample)):
						if self.ns_count_temp[i] != 0:
							w = min(1.0, 0.1 * self.ns_count_temp[i])
							self.ns_slope_sample[i] = (1-w) * self.ns_slope_sample[i] + w * self.ns_slope_temp[i] / self.ns_count_temp[i]
					print('ns slope goals current: ', self.ns_slope_temp)
					print('ns slope goals: ', self.ns_slope_sample)

					for i in range(len(self.sample)):
						self.ns_slope_temp[i] = 0
						self.ns_count_temp[i] = 0

				self.prob = []

				for i in range(len(self.sample)):
					if self.type_explore == 2:
						self.prob.append(self.probAdaptiveSampling(i))
					elif self.type_explore == 4:
						self.prob.append(self.probTSSampling(i))
					else:
						self.prob.append(self.probTS2Sampling(i))
				prob_mean = np.array(self.prob).mean() * len(self.sample)
				self.bound_sample = []
				for i in range(len(self.sample)):
					if i == 0:
						self.bound_sample.append(self.prob[i] / prob_mean)
					else:
						self.bound_sample.append(self.bound_sample[-1] + self.prob[i] / prob_mean)
				print(self.bound_sample)
	def adaptiveSample(self, visited):
		if visited:
			if self.n_visit < 1 or self.n_visit % 5 == 4:
				return self.randomSample(visited), -1

			if self.type_visit == 0:
				return self.randomSample(visited), -1
			else:
				t = np.random.randint(len(self.pool))
				target = self.pool[t] 
			return target, t
		else:
			if self.type_explore == 0:
				return self.randomSample(visited), -1
			elif self.type_explore == 5:
				t = np.random.randint(len(self.sample))	
				target = self.sample[t]
				idx = t
				return target, t
			elif self.type_explore == 1 or self.type_explore == 3:
				if self.start < 2:
					return self.randomSample(visited), -1
				t = np.random.randint(len(self.pool_ex))
				target = self.pool_ex[t]
				return target, t 
			else:
				if self.n_explore <= 2:
					t = np.random.randint(len(self.sample))	
					target = self.sample[t]
					idx = t
				else:
					t = np.random.rand()
					if self.epsilon_greedy:
						if t < self.epsilon:
							t = np.random.randint(len(self.sample))	
							target = self.sample[t]
							idx = t
						else:
							idx = 0
							max_prob = 0
							for i in range(len(self.prob)):
								if max_prob < self.prob[i]:
									max_prob = self.prob[i]
									idx = i
							target = self.sample[idx]
					else:
						idx = -1	
						for i in range(len(self.bound_sample)):
							if t <= self.bound_sample[i]:
								target = self.sample[i]
								idx = i
								break
						if idx == -1:
							idx = len(self.bound_sample) - 1
							target = self.sample[idx]
				if self.type_explore == 6:
					self.prev_ns = self.sim_env.GetNumSamples()

				return target, idx

	def reset_visit(self):
		self.random_start = True
		self.v_mean = 0
		self.n_visit = 0
		self.n_learning += 1

	def sampleGoals(self, m=10):
		self.sample = []
		self.v_sample = []
		self.ns_slope_sample = []
		self.ns_slope_temp = []
		self.ns_count_temp = []
		for i in range(m):
			self.sample.append(self.randomSample(False))
			self.v_sample.append(0.1)
			self.ns_slope_sample.append(0)
			self.ns_slope_temp.append(0)
			self.ns_count_temp.append(0)

		self.v_prev_sample = copy(self.v_sample)
		# print('new goals: ', self.sample)

	def reset_explore(self):
		if self.type_explore != 0 and self.type_explore != 1 and self.type_explore != 3:
			self.sampleGoals()
		self.n_explore = 0

	def isEnough(self, v_func):
		
		self.random_start = False

		print("===========================================")
		print("mean reward : ", self.v_mean)
		print("===========================================")

		if self.n_visit % 5 == 4:
			self.printSummary(v_func)
			if self.n_visit > 10 and self.v_mean_cur > 0.9:
				return True
		if self.n_visit > 10 and self.v_mean > 0.9:
				return True

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
		return vs_mean, vs_mean_bottom, vs_mean_top