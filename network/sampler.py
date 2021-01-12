import numpy as np
import math
from regression import Regression
from IPython import embed
from copy import copy

class Sampler(object):
	def __init__(self, sim_env, dim):
		self.sim_env = sim_env
		self.dim = dim
		
		self.v_mean = 0.0
		self.v_mean_boundary = 0.0
		self.random = True

		self.k = 5
		self.k_ex = 20

		self.total_iter = 0

		# 0: uniform 1: adaptive
		self.type_exploit = 0
		# 0: uniform 1 :adaptive
		self.type_explore = 0
	
		self.done = False

		# value, progress, updated
		self.v_list_explore = []
		self.p_list_explore = []
		self.progress_cur_list = []
		self.distance = 2
		self.unit = 1.5
		# keep if added recently or data is rare ( <= 5)
		self.vp_dict = dict()

		self.eval_target_v = 0

		self.progress_queue_exploit = [10.0]
		self.progress_queue_explore = [0]

		self.progress_cur = 0
		self.evaluation_counter = 0
		self.evaluation_done = False
		self.eval_frequency = 0

		self.progress_cur_new = 0
		self.progress_cur_update = 0

		self.use_table = True
		self.delta = 0
		print('=======================================')
		print('curriculum option')
		print('type exploit', self.type_exploit)
		print('type explore', self.type_explore)
		print('use table', self.use_table)

		print('=======================================')


	def randomSample(self, visited=1):
		return self.sim_env.UniformSample(visited)
		
	def probAdaptive(self, v_func, target, hard=True):
		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)[0]
		if hard:
			return math.exp(-self.k * (v - self.v_mean) / self.v_mean) + 1e-10
		else:
			return math.exp(self.k_ex * (v - self.v_mean) / self.v_mean) + 1e-10

	def updateGoalDistribution(self, mode, v_func, m=2, N=400):
		if mode == 0:
			self.n_explore += 1
		elif mode == 1:
			self.n_exploit += 1
		else:
			self.n_evaluation += 1
		self.total_iter += 1
		
		if mode == 0:
			if self.total_iter < 3:
				return
			if self.type_explore == 0:
				self.sampleBatch(self.type_explore)
			elif self.type_explore == 1:
				self.pool_ex = []
				for i in range(m):
					x_cur = self.randomSample(mode)
					for j in range(int(N/m)):
						self.pool_ex.append(x_cur)
						x_new = self.randomSample(mode)
						alpha = min(1.0, self.probAdaptive(v_func, x_new, False)/self.probAdaptive(v_func, x_cur, False))
						
						if np.random.rand() <= alpha:          
							x_cur = x_new
				self.sampleBatch(self.type_explore)
	
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
		elif mode == 2:
			self.sample = []
			self.evaluation_counter = 0

			while len(self.sample) < 10:
				t = self.randomSample(False) 
				self.sample.append(t)
	def saveProgress(self, mode):
		plist = self.sim_env.GetExplorationRate()
		p = plist[0] + plist[1]
		self.progress_cur += p
		self.progress_cur_new += plist[0]
		self.progress_cur_update += plist[1]
		if mode == 0 or mode == 2:
			self.progress_cur_list.append(p * 10)

	def updateCurrentStatus(self, mode, results, info):
		if mode == 0 or mode == 2:
			if len(self.progress_queue_explore) >= 10:
				self.progress_queue_explore = self.progress_queue_explore[1:]
			self.progress_queue_explore.append(self.progress_cur)

			self.value_cur_list = []
			self.count = []
			for i in range(len(self.progress_cur_list)):
				self.value_cur_list.append(0)
				self.count.append(0)
			for i in range(len(results)):
				self.value_cur_list[info[i]] += results[i]
				self.count[info[i]] += 1

			for i in range(len(self.progress_cur_list)):
				if self.count[i] != 0:	
					v = self.value_cur_list[i] / self.count[i]
					v_key = math.floor(v * 1 / self.unit)
					if v_key in self.vp_dict:
						while len( self.vp_dict[v_key]) >= 5 and (self.vp_dict[v_key][-1][2] - self.vp_dict[v_key][0][2]) > 3:
							 self.vp_dict[v_key] = self.vp_dict[v_key][1:]
						self.vp_dict[v_key].append([v, self.progress_cur_list[i], self.total_iter])
					else:
						self.vp_dict[v_key] = [[v, self.progress_cur_list[i], self.total_iter]] 

			self.progress_cur_list = []
			self.sample_counter = 0

			if mode == 2 and self.n_evaluation == 3:
				self.evaluation_done = True
				self.printExplorationRateData()
			if len(self.progress_queue_explore) < 5 and self.v_mean != 0:
				if len(self.progress_queue_explore) == 1:
					self.delta = self.v_mean - np.array(results).mean()
					print('new delta :', self.delta)
				else:
					self.delta = 0.9 * self.delta + 0.1 * (self.v_mean - np.array(results).mean())
					print('delta updated:', self.delta)

		elif mode == 1:		
			if len(self.progress_queue_exploit) >= 5:
				self.progress_queue_exploit = self.progress_queue_exploit[1:]
			self.progress_queue_exploit.append(self.progress_cur)		

			if len(results) != 0:
				self.v_mean_cur = np.array(results).mean()
				if self.v_mean == 0:
					self.v_mean = self.v_mean_cur
				else:
					self.v_mean = 0.6 * self.v_mean + 0.4 * self.v_mean_cur

				p_mean = np.array(self.progress_queue_exploit).mean()

				v_mean_boundary_cur = 0
				count = 0
				for i in range(len(results)):
					if info[i] > 1.0 and info[i] < 1.15:
						count += 1
						v_mean_boundary_cur += results[i]
				rate = 0		
				if count != 0:
					v_mean_boundary_cur /= count
					if self.v_mean_boundary == 0:
						self.v_mean_boundary = v_mean_boundary_cur
					else:
						rate = min(0.2, count * 0.0006)
						self.v_mean_boundary = (1- rate) * self.v_mean_boundary + rate * v_mean_boundary_cur
			

				print(self.progress_queue_exploit)
				print("===========================================")
				print("mean reward : ", self.v_mean)
				print("current mean boundary reward: ", v_mean_boundary_cur, rate)
				print("mean boundary reward : ", self.v_mean_boundary)
				print("exploration rate : ", p_mean)
				print("===========================================")

		print(self.progress_cur_new, self.progress_cur_update)
		self.progress_cur_new = 0
		self.progress_cur_update = 0
		self.progress_cur = 0

	def adaptiveSample(self, mode, v_func):
		if mode == 0:
			if self.total_iter < 3:
				target = self.randomSample(mode)

				self.sample_counter += 1
				t = self.sample_counter-1

			elif self.type_explore == 0:
				target = self.sample[self.sample_counter % len(self.sample)]
				self.sample_counter += 1
				t = self.sample_counter-1
			else:
				target = self.sample[self.sample_counter % len(self.sample)]
				self.sample_counter += 1
				t = self.sample_counter-1

			return target, t

		elif mode == 1:
			if self.n_exploit % 5 == 4:
				target = self.randomSample(mode)
				target_np = np.array(target, dtype=np.float32) 
				t = self.sim_env.GetDensity(target_np)				
			elif self.type_exploit == 0:
				target = self.randomSample(mode)
				target_np = np.array(target, dtype=np.float32) 
				t = self.sim_env.GetDensity(target_np)
			else:
				t = np.random.randint(len(self.pool)) 
				target = self.pool[t] 
				target_np = np.array(target, dtype=np.float32) 
				t = self.sim_env.GetDensity(target_np)
			return target, t
		else:
			t = self.evaluation_counter % len(self.sample)
			target = self.sample[t]
			self.evaluation_counter += 1

			return target, t

	def resetExploit(self):
		self.n_exploit = 0
		self.prev_progress_ex = np.array(self.progress_queue_exploit).mean()
		self.prev_queue_exploit = copy(self.progress_queue_exploit)
		self.progress_queue_exploit = []
		self.v_mean = 0
		self.printExplorationRateData()

	def resetExplore(self):
		self.n_explore = 0
		self.prev_progress = np.array(self.progress_queue_explore).mean()
		self.progress_queue_explore = []
		self.v_list_explore = []
		self.p_list_explore = []
		self.sample_counter = 0 
		self.delta = 0

	def sampleBatch(self, type_explore):
		self.sample = []
		self.sample_counter = 0
		for _ in range(20):
			if type_explore == 0:
				target = self.randomSample(False) 	
			else:
				t = np.random.randint(len(self.pool_ex)) 
				target = self.pool_ex[t] 
			self.sample.append(target)


	def resetEvaluation(self):
		self.n_evaluation = 0
		self.sample = []
		self.v_list_explore = []
		self.p_list_explore = []
		self.progress_queue_explore = []
		self.evaluation_done = False
		self.evaluation_counter = 0

		while len(self.sample) < 10:
			t = self.randomSample(False) 
			self.sample.append(t)

		self.eval_frequency = len(self.sample)

	def printExplorationRateData(self):
		for k, v in zip(self.vp_dict.keys(), self.vp_dict.values()):
			print('(', k * self.unit, ',', (k+1)*self.unit,') : ',v, ' ', np.array(v).mean(axis=0)[1])

	def predictWindow(self, v, scale):
		v_min = v - scale
		v_max = v + scale
		mean = 0
		count = 0

		for v, p in zip(self.v_list_explore, self.p_list_explore):
			if v >= v_min and v <= v_max:
				mean += p
				count += 1
		return mean, count

	def isEnough(self):

		p_mean = np.array(self.progress_queue_exploit).mean()
		p_mean_prev = np.array(self.progress_queue_explore).mean()

		if self.n_exploit < 10:
			return False

		if self.use_table:
			max_mean = 0
			v_key = math.floor((self.v_mean - self.delta) * 1 / self.unit)
			it = 0
			while v_key in self.vp_dict:
				count = 0
				mean = 0
				for i in range(len(self.vp_dict[v_key])):
					if self.total_iter - self.vp_dict[v_key][i][2] < 50:
						mean += self.vp_dict[v_key][i][1]
						count += 1
					
				if count != 0:
					mean /= count	

				if max_mean < mean:
					max_mean = mean
				v_key -= 1
				it += 1
				if it >= 3:
					break

			print((self.v_mean - self.delta), p_mean, max_mean)
			if p_mean <= max_mean or p_mean <= 0.2:
				return True
		else:
			print(p_mean, p_mean_prev, (p_mean + 1e-3) - p_mean_prev * 0.9 < 0)
			if (p_mean + 1e-3) - p_mean_prev * 0.9 < 0:
				return True

		return False