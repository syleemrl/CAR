import numpy as np
import math
from regression import Regression
from IPython import embed
from copy import copy
from sklearn.neighbors import KNeighborsRegressor

class Sampler(object):
	def __init__(self, sim_env, dim, path):
		self.sim_env = sim_env
		self.dim = dim
		
		self.k = 15
		self.k_ex = 20

		self.path = path

		# 0: uniform 1: adaptive 2: adaptive2
		self.type_exploit = 0
		# 0: uniform 1 :adaptive(network)
		self.type_explore = 0

		self.sample_counter = 0
		self.regressor = KNeighborsRegressor(n_neighbors=20, weights="uniform")
		self.regression_x = []
		self.regression_y = []

		self.iter = 0
		self.v_mean = 0
		self.v_ratio = 0
		self.exploration_ratio = 1
		self.uniform_mean = 0
		self.uniform_std = 0
		
		self.done = False
		self.reg_init = False
		self.mode = 0

		if self.type_exploit == 2:
			self.sampleGoals(True, 10)
		else:
			self.sampleGoals(False, 10)

		print('=======================================')
		print('curriculum option')
		print('type visit', self.type_exploit)
		print('type explore', self.type_explore)
		print('=======================================')


	def randomSample(self, exploit=True):
		return self.sim_env.UniformSample(exploit)
		
	def probAdaptive(self, v_func, target, hard=True):
		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)[0]
		if hard:
			return math.exp(-self.k * (v - self.v_mean) / self.v_mean) + 1e-10
		else:
			return math.exp(self.k * (v - self.v_mean) / self.v_mean) + 1e-10

	def probAdaptive2(self, v_func, target):
		if not self.reg_init:
			return 1

		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)
		v = v.reshape(-1, 1)
		p_predict =	self.regressor.predict(v)[0]

		return math.exp(p_predict * 100)

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
		slope = (v - v_prev) / v_prev * self.k_ex
		if slope > 10:
			slope = 10
		return math.exp(slope) + 1e-10

	def updateGoalDistribution(self, v_func, v_func_prev, results, idxs, m=4, N=200):
		self.iter += 1
		self.sample_counter = 0
		self.v_mean_cur = np.array(results).mean()
		
		if self.v_mean == 0:
			self.v_mean = self.v_mean_cur
		else:
			self.v_mean = 0.6 * self.v_mean + 0.4 * self.v_mean_cur
		print("===========================================")
		print("mean reward : ", self.v_mean)
		print("===========================================")
		if self.iter % 10 == 9:
			results = np.sort(np.array(results))
			mean_bottom = results[:20].mean()
			mean_top = results[180:].mean()

			print("uniform mean reward : ", self.v_mean_cur)
			print("uniform bottom reward : ", mean_bottom)
			print("uniform top reward : ", mean_top)
			print("===========================================")
		########update exploitation curriculum#########

		if self.type_exploit == 1:	
			self.pool = []
			for i in range(m):
				x_cur = self.randomSample(True)
				for j in range(int(N/m)):
					self.pool.append(x_cur)
					x_new = self.randomSample(True)
					alpha = min(1.0, self.probAdaptive(v_func, x_new, True)/self.probAdaptive(v_func, x_cur, True))
					if np.random.rand() <= alpha:          
						x_cur = x_new

		if self.type_exploit == 2:
			if self.iter % 2 == 1:
				v_mean_sample_cur = [0] * len(self.sample)
				count_sample_cur = [0] * len(self.sample)

				for i in range(len(results)):
					if idxs[i] != -1:
						v_mean_sample_cur[idxs[i]] += results[i]
						count_sample_cur[idxs[i]] += 1
				
				self.v_prev_sample = []
				for i in range(len(self.sample)):
					if count_sample_cur[i] != 0:
						self.v_prev_sample.append(v_mean_sample_cur[i] / count_sample_cur[i])
					else:
						self.v_prev_sample.append(-1)

			elif self.iter % 2 == 0:
				v_mean_sample_cur = [0] * len(self.sample)
				count_sample_cur = [0] * len(self.sample)

				for i in range(len(results)):
					if idxs[i] != -1:
						v_mean_sample_cur[idxs[i]] += results[i]
						count_sample_cur[idxs[i]] += 1
						
				self.v_sample = []
				v_diff = []
				for i in range(len(self.sample)):
					if count_sample_cur[i] == 0:
						self.v_sample.append(-1)
					else:
						self.v_sample.append(v_mean_sample_cur[i] / count_sample_cur[i])
						v_diff = self.v_sample[i] - self.v_prev_sample[i]
						if self.v_sample[i] != -1 and self.v_prev_sample[i] != -1:
							self.regression_x.append(self.v_prev_sample[i])
							self.regression_y.append(v_diff)
				
				if self.iter % 10 == 8:
					self.sample = []
					for _ in range(10):
						t = self.randomSample(True)
						self.sample.append(t)
				else:
					if len(self.regression_x) >= 50:
						out = open(self.path+'_'+str(self.iter), "w")
						for i in range(len(self.regression_x)):
							out.write(str(self.regression_x[i])+', '+str(self.regression_y[i])+'\n')
						out.close()

						x =  np.array(self.regression_x).reshape(-1, 1)
						self.regressor.fit(x, self.regression_y)

						x = np.linspace(0.8, 1.2, num=10)
						x_ = np.array(x).reshape(-1, 1)
						y = self.regressor.predict(x_)
						print('progress by regressor: ', y)

						self.regression_x = []
						self.regression_y = []
						self.reg_init = True

					self.pool = []
					for i in range(m):
						x_cur = self.randomSample(True)
						for j in range(int(N/m)):
							self.pool.append(x_cur)
							x_new = self.randomSample(True)
							alpha = min(1.0, self.probAdaptive2(v_func, x_new)/self.probAdaptive2(v_func, x_cur))
							if np.random.rand() <= alpha:          
								x_cur = x_new		

					self.sample = []
					for _ in range(10):
						t = np.random.randint(len(self.pool))
						self.sample.append(self.pool[t])
		########update exploration curriculum#########

		if self.type_explore == 0:
			return

		self.pool_ex = []
		self.idx_ex = []
		if self.type_explore == 1:
			for i in range(m):
				x_cur = self.randomSample(False)
				for j in range(int(N/m)):
					self.pool_ex.append(x_cur)
					x_new = self.randomSample(False)
					if self.type_explore == 1:
						alpha = min(1.0, self.probAdaptive(v_func, x_new, False)/self.probAdaptive(v_func, x_cur, False))
					if np.random.rand() <= alpha:          
						x_cur = x_new
	
		# else:
		# 	v_mean_sample_cur = [0] * len(self.sample)
		# 	count_sample_cur = [0] * len(self.sample)

		# 	for i in range(len(results)):
		# 		v_mean_sample_cur[idxs[i]] += results[i]
		# 		count_sample_cur[idxs[i]] += 1
			
		# 	for i in range(len(self.sample)):
		# 		d = self.sim_env.GetDensity(self.sample[i]) 
		# 		if d > 0.3:
		# 			print(self.sample[i], d)
		# 			self.sample[i] = self.randomSample(False)
		# 			self.v_prev_sample[i] = 0.8
		# 			self.v_sample[i] = 1.0
		# 		else:
		# 			if count_sample_cur[i] != 0:
		# 				self.v_prev_sample[i] = copy(self.v_sample[i])
		# 				self.v_sample[i] = v_mean_sample_cur[i] / count_sample_cur[i]

		# 	for i in range(len(self.sample)):
		# 		if count_sample_cur[i] != 0:
		# 			self.v_prev_sample[i] = copy(self.v_sample[i])
		# 			self.v_sample[i] = 0.6 * self.v_sample[i] + 0.4 * v_mean_sample_cur[i] / count_sample_cur[i]

		# 	print('v prev goals: ', self.v_prev_sample)
		# 	print('v goals: ', self.v_sample)

		# 	prob = []
		# 	for i in range(len(self.sample)):
		# 		if self.type_explore == 2:
		# 			prob.append(self.probAdaptiveSampling(i))
		# 		else:
		# 			prob.append(self.probTSSampling(i))
		# 	prob_mean = np.array(prob).mean() * len(self.sample)
				
		# 	self.bound_sample = []
		# 	for i in range(len(self.sample)):
		# 		if i == 0:
		# 			self.bound_sample.append(prob[i] / prob_mean)
		# 		else:
		# 			self.bound_sample.append(self.bound_sample[-1] + prob[i] / prob_mean)
		# 	print(self.bound_sample)

	def adaptiveSample(self):
		r = np.random.rand()
		if r < self.exploration_ratio:
			exploit = False
		else:
			exploit = True

		###evaluation###	
		if self.iter % 10 == 8:
			if self.type_exploit != 2:
				return self.randomSample(True), -1, True
			else:
				t = math.floor(self.sample_counter / 2.0) % len(self.sample)
				target = self.sample[t]
				self.sample_counter += 1
				return target, t, True

		###else###
		if exploit:
			if self.type_exploit == 0:
				return self.randomSample(exploit), -1, True
			elif self.type_exploit == 1:
				t = np.random.randint(len(self.pool))
				target = self.pool[t]
			elif self.type_exploit == 2:
				t = math.floor(self.sample_counter / 2.0) % len(self.sample)
				target = self.sample[t]
				self.sample_counter += 1
			return target, t, True
		else:
			if self.iter < 5 or self.type_explore == 0:
				return self.randomSample(exploit), -1, False
			elif self.type_explore == 1:
				t = np.random.randint(len(self.pool_ex))
				target = self.pool_ex[t]
				return target, -1, False
			# else:
			# 	t = np.random.rand()
			# 	idx = -1	
			# 	for i in range(len(self.bound_sample)):
			# 		if t <= self.bound_sample[i]:
			# 			target = self.sample_ex[i]
			# 			idx = i
			# 			break
			# 	if idx == -1:
			# 		idx = len(self.bound_sample) - 1
			# 		target = self.sample_ex[idx]
			# 	return target, idx


	def sampleGoals(self, exploit, m=10):
		if exploit:
			self.sample = []
			self.v_sample = []
			for i in range(m):
				self.sample.append(self.randomSample(exploit))
				self.v_sample.append(1.0)
			self.v_prev_sample = copy(self.v_sample)
		else:
			self.sample_ex = []
			self.v_sample_ex = []
			for i in range(m):
				self.sample_ex.append(self.randomSample(exploit))
				self.v_sample_ex.append(1.0)
			self.v_prev_sample_ex = copy(self.v_sample_ex)

	def isEnough(self):
		
		if self.iter % 10 == 9:
			self.exploration_ratio = max(0, math.exp((self.v_mean_cur - 1.05)*2) - 1)
			print('exploration ratio: ', self.exploration_ratio)
		else:
			exploration_ratio_temp = max(0, math.exp((self.v_mean_cur - 1.05)*2) - 1)
			# if abs(exploration_ratio_temp - self.exploration_ratio) > 0.15:
			# 	self.exploration_ratio = exploration_ratio_temp
			# 	print('exploration ratio: ', self.exploration_ratio)

		if self.done:
			self.exploration_ratio = 0

		if self.v_mean > 1.4 and self.done:
			return True	

		return False
