import numpy as np
import math
from regression import Regression
from IPython import embed
from copy import copy
class Sampler(object):
	def __init__(self, sim_env, dim, path):
		self.sim_env = sim_env
		self.dim = dim
		
		self.v_mean = 1.0
		self.random = True

		self.k = 10
		self.k_ex = 20

		self.total_iter = 0
		self.path = path

		# 0: uniform 1: adaptive 2: ts
		self.type_exploit = 1
		# 0: uniform 1 :adaptive(network) 2:adaptive(sampling) 3:ts(network) 4:ts(sampling) 5: uniform(sampling)
		# 6: num sample slope(sampling) 7:num sample near goal(sampling) 8: num sample slope (network)
		self.type_explore = 0
		
		self.prev_action = 0
		self.prev_nsample = 0
		self.ns_mean = 0

		self.done = False

		# value, progress, updated
		self.vp_table = [[1.0, 10, 0]]
		self.eval_target_v = 0

		self.progress_queue_evaluation = []
		self.progress_queue_exploit = [10.0]
		self.progress_queue_explore = [0]

		self.progress_cur = 0
		self.evaluation_counter = 0
		self.evaluation_done = False
		self.eval_frequency = 0
		self.scale = 1.0 / 0.05
		print('=======================================')
		print('curriculum option')
		print('type exploit', self.type_exploit)
		print('type explore', self.type_explore)
		print('=======================================')

	def randomSample(self, visited=True):
		return self.sim_env.UniformSample(visited)
		
	def probAdaptive(self, v_func, target, hard=True):
		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)[0]
		if hard:
			return math.exp(- self.k * (v - self.v_mean) / self.v_mean) + 1e-10
		else:
			return math.exp(self.k_ex * (v - self.v_mean) / self.v_mean) + 1e-10

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
		if self.hard:
			return math.exp(self.k_ex * -(v - self.v_mean) / self.v_mean) + 1e-10
		else:
			return math.exp(self.k_ex * (v - self.v_mean) / self.v_mean) + 1e-10

	def probTSSampling(self, idx):
		v = self.v_sample[idx]
		v_prev = self.v_prev_sample[idx]
		slope = (v - v_prev) / v_prev * self.k_ex * 3

		if self.hard:
			slope = -slope
		if slope > 10:
			slope = 10
		return math.exp(slope) + 1e-10

	def probTS2Sampling(self, idx):
		v = self.ns_slope_sample[idx]
		mean = np.array(self.ns_slope_sample).mean() + 1e-8
		if self.hard:
			return math.exp(-(v - mean) / mean) + 1e-10
		else:
			return math.exp((v - mean) / mean) + 1e-10

	def updateGoalDistribution(self, mode, v_func, m=2, N=400):
		if mode == 0:
			self.n_explore += 1
		elif mode == 1:
			self.n_exploit += 1
		else:
			self.n_evaluation += 1
		self.total_iter += 1
		

		it = 0
		while it < len(self.vp_table):
			if mode == 0:
				self.vp_table[it][2] += 4
			else:
				self.vp_table[it][2] += 1
			if self.vp_table[it][2] >= 30:
				self.vp_table = self.vp_table[:it] + self.vp_table[(it+1):]  
			else:
				it += 1

		if mode == 0:
			if self.type_explore == 0 or self.total_iter < 5:
				return
			self.pool_ex = []
			if self.type_explore == 1:
				for i in range(m):
					x_cur = self.randomSample(mode)
					for j in range(int(N/m)):
						self.pool_ex.append(x_cur)
						x_new = self.randomSample(mode)
						alpha = min(1.0, self.probAdaptive(v_func, x_new, False)/self.probAdaptive(v_func, x_cur, False))
					
						if np.random.rand() <= alpha:          
							x_cur = x_new	
		elif mode == 1:
			if self.type_exploit == 0:
				return
			self.pool = []
			for i in range(m):
				x_cur = self.randomSample(mode)
				for j in range(int(N/m)):
					self.pool.append(x_cur)
					x_new = self.randomSample(mode)
					if self.type_exploit == 1:
						alpha = min(1.0, self.probAdaptive(v_func, x_new, True)/self.probAdaptive(v_func, x_cur, True))
					else:
						alpha = min(1.0, self.probTS(v_func, v_func_prev, x_new, True)/self.probTS(v_func, v_func_prev, x_cur, True))

					if np.random.rand() <= alpha:          
						x_cur = x_new
		else:
			if self.n_evaluation == 3:
				vp_table_tmp = []
				vp_table_max = 0
				for i in range(len(self.sample_progress)):
					vp_table_tmp.append([self.sample_progress[i][1], self.sample_progress[i][0] / 2.0, 0])
					if self.sample_progress[i][1] > vp_table_max:
						vp_table_max = self.sample_progress[i][1]
				for i in range(len(self.vp_table)):
					if vp_table_max < self.vp_table[i][0]:
						vp_table_tmp.append(self.vp_table[i])

				self.vp_table = copy(vp_table_tmp)
				print(self.vp_table)
				self.evaluation_done = True


	def saveProgress(self, mode):
		if mode == 0:
			p = self.sim_env.GetProgressGoal()
			self.progress_cur += p
			if self.eval_target_v < 0.7:
				return

			if self.total_iter >= 5:
				flag = False
				for i in range(len(self.vp_table)):
					if abs(self.vp_table[i][0] - self.eval_target_v) < 1e-2:
						rate = 0.1 + 0.02 * self.vp_table[i][2]
						self.vp_table[i][1] = (1 - rate) * self.vp_table[i][1] + rate * p * 10
						self.vp_table[i][2] = 0
						flag = True
				if not flag:
					self.vp_table.append([self.eval_target_v, 0.5 * p * 10, 0])

		elif mode == 2:
			t = self.evaluation_counter % len(self.sample)
			lb = self.sample[t][1]
			for i in range(len(self.sample_progress)):
				if abs(self.sample_progress[i][1] - lb) < 1e-2:
					self.sample_progress[i][0] += self.sim_env.GetProgressGoal()
					break 
			self.evaluation_counter += 1
		else:
			self.progress_cur += self.sim_env.GetProgressGoal()

	def updateProgress(self, mode):
		if mode == 0:
			if len(self.progress_queue_explore) >= 5:
				self.progress_queue_explore = self.progress_queue_explore[1:]
			self.progress_queue_explore.append(self.progress_cur)
		elif mode == 1:		
			if len(self.progress_queue_exploit) >= 10:
				self.progress_queue_exploit = self.progress_queue_exploit[1:]
			self.progress_queue_exploit.append(self.progress_cur)
		else:
			self.progress_queue_evaluation.append(self.progress_cur)		

		self.progress_cur = 0

	def adaptiveSample(self, mode, v_func):
		if mode == 0:
			if self.total_iter < 5:
				target = self.randomSample(mode)
				t = -1
			elif self.type_explore == 0:
				li = self.sim_env.UniformSampleWithNearestParams() 
				params = li[1]
				vs = v_func.getValue(params)
				v = np.array(vs).mean()
				target = li[0]
				self.eval_target_v = math.floor(v * self.scale) / self.scale
				t = -1
			else:
				t = np.random.randint(len(self.pool_ex)) 
				target = self.pool_ex[t] 
				target_np = np.array(target, dtype=np.float32) 
				params = self.sim_env.GetNearestParams(target_np) 
				vs = v_func.getValue(params)
				v = np.array(vs).mean()
				self.eval_target_v = math.floor(v * self.scale) / self.scale
			return target, t

		elif mode == 1:
			if self.n_exploit < 1 or self.n_exploit % 5 == 4:
				target = self.randomSample(mode)
				t = -1
			elif self.type_exploit == 0:
				target = self.randomSample(mode)
				t = -1
			else:
				t = np.random.randint(len(self.pool)) 
				target = self.pool[t] 

			return target, t
		else:
			t = self.evaluation_counter % len(self.sample)
			target = self.sample[t][0]

			return target, t

	def resetExploit(self):
		self.n_exploit = 0
		self.prev_progress = np.array(self.progress_queue_exploit).mean()
		self.progress_queue_exploit = []

	def resetExplore(self):
		self.n_explore = 0
		self.prev_progress = np.array(self.progress_queue_explore).mean()
		self.progress_queue_explore = []

	def resetEvaluation(self, v_func):
		self.n_evaluation = 0
		self.eval_target_v = max(0.75, math.floor((self.v_mean - 0.1) * self.scale) / self.scale)
		self.sample = []
		self.sample_progress = []
		self.evaluation_done = False
		self.evaluation_counter = 0

		while True:
			it = 0
			add = 0
			flag = True
			while add < 10:
				it += 1

				li = self.sim_env.UniformSampleWithNearestParams() 
				params = li[1]
				vs = v_func.getValue(params)
				v = np.array(vs).mean()
				t = li[0]
				# t = self.randomSample(False)
				# v = v_func.getValue([t])[0]
				if v >= self.eval_target_v  and v < self.eval_target_v + 0.05:
					self.sample.append([t, self.eval_target_v ])
					add += 1
					it = 0

				if it > 1000:
					flag = False
					break

			if flag:
				self.sample_progress.append([0, self.eval_target_v])
			# if not flag:
			# 	break

			self.eval_target_v += 0.05
			if self.eval_target_v > self.v_mean + 0.15:
				break
		self.eval_frequency = len(self.sample) + 1
		print(self.sample)
		print(self.eval_frequency)

	def isEnough(self, results):
		self.v_mean_cur = np.array(results).mean()
		if self.v_mean == 0:
			self.v_mean = self.v_mean_cur
		else:
			self.v_mean = 0.6 * self.v_mean + 0.4 * self.v_mean_cur

		p_mean = np.array(self.progress_queue_exploit).mean()

		print(self.progress_queue_exploit)
		print("===========================================")
		print("mean reward : ", self.v_mean)
		print("exploration rate : ", p_mean)
		print("===========================================")


		if self.n_exploit < 5:
			return False

		v = math.floor(self.v_mean * self.scale) / self.scale
		for i in reversed(range(len(self.vp_table))):
			if self.vp_table[i][0] <= v - 1 / self.scale and p_mean < self.vp_table[i][1]:
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