import numpy as np
import math
from regression import Regression
from IPython import embed
from copy import copy
from sklearn.neighbors import KNeighborsRegressor
import os
class Sampler(object):
	def __init__(self, sim_env, dim, path, explore, visit, egreedy, hard, print_verbose):
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
		# 0: uniform 1: adaptive 2: min(borderline, mean) 3: adaptive2 
		self.type_visit = visit
		# 0: uniform 1 :adaptive(network) 2:adaptive(sampling) 3:adaptive2(network) 4:ts(sampling) 5: uniform(sampling)
		# 6: num sample slope(sampling) 7:num sample near goal(sampling) 8: num sample slope (network)
		self.type_explore = explore

		# if egreedy:
		# 	self.epsilon_greedy = True
		# else:
		# 	self.epsilon_greedy = False
		# if hard:
		# 	self.hard = True
		# else:
		# 	self.hard = False
		self.epsilon = 0.25

		self.state_batch = []
		self.progress_batch = []

		self.prev_action = 0
		self.prev_nsample = 0
		self.ns_mean = 0

		self.done = False
		self.exploration_test_print = print_verbose
		
		self.progress = []
		
		self.v_mean_uniform = 1.0
		self.v_std_uniform = 0.1

		# 0 : exploitation 1: exploration 2: evaluation
		self.mode = 2
		self.sample_counter = 0

		if hard == 0:
			self.regressor = KNeighborsRegressor(n_neighbors=8)
		else:
			self.regressor = KNeighborsRegressor(n_neighbors=10, weights="distance", metric='minkowski')

		self.iter = egreedy

		self.regression_x = []
		self.regression_y = []

		print('=======================================')
		print('curriculum option')
		print('type visit', self.type_visit)
		print('type explore', self.type_explore)
	#	print('e greedy', self.epsilon_greedy)
	#	print('hard', self.hard)
		print('=======================================')

	def randomSample(self, visited=True):
		return self.sim_env.UniformSample(visited, False)
		
	def probAdaptive(self, v_func, target, hard=True):
		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)[0]
		if hard:
			return math.exp(- self.k * (v - self.v_mean) / self.v_mean) + 1e-10
		else:
			if self.type_explore == 8:
				return math.exp(3 *  (v- self.ns_mean) / self.ns_mean) + 1e-10
			else:
				return math.exp(5 * self.k * (v - self.v_mean) / self.v_mean) + 1e-10

	def probAdaptive2(self, v_func, target):
		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)
		v = v.reshape(-1, 1)
		p_predict =	self.regressor.predict(v)[0]

		return math.exp(p_predict * 100)

	def probTS(self, v_func, v_func_prev, target, hard=True):
		target = np.reshape(target, (-1, self.dim))
		v = v_func.getValue(target)[0]
		slope = v - min(self.v_mean, 1.1)
		if slope < 0:
			slope = 0.5 * abs(slope)

		return math.exp(-slope * 20) + 1e-10


	def probAdaptiveSampling(self, idx):
		v = self.v_sample[idx]
		if self.hard:
			return math.exp(self.k * -(v - self.v_mean) / self.v_mean) + 1e-10
		else:
			return math.exp(self.k_ex * (v - self.v_mean) / self.v_mean) + 1e-10

	def probTSSampling(self, idx):
		v = self.v_sample[idx]
		v_prev = self.v_prev_sample[idx]
		v = v - min(1.1, 0.3 * self.v_std_uniform + self.v_mean_uniform)
		if v < 0:
			v = 0.5 * abs(v)
		else:
			v = abs(v)
		return math.exp(- v * 25) + 1e-10
		

	def probTS2Sampling(self, idx):
		v = self.ns_slope_sample[idx]
		mean = np.array(self.ns_slope_sample).mean() + 1e-8
		if self.hard:
			return math.exp(-(v - mean)) + 1e-10
		else:
			return math.exp(0.3 * abs(v - 0.85)) + 1e-10

	def updateNumSampleDelta(self, idx):
		if self.type_explore < 6:
			return
		if self.type_explore == 7:
			slope = self.sim_env.GetNewSamplesNearGoal()
		else:
			slope =  max(0, self.sim_env.GetNumSamples() - self.prev_nsample)
		if self.type_explore == 6 or self.type_explore == 7:
			self.ns_slope_temp[idx] += slope
			self.ns_count_temp[idx] += 1
		else:
			# print(slope)
			# print(self.sim_env.GetNumSamples() - self.prev_nsample)
			self.ns_slope_sample.append(self.prev_action)
			self.ns_slope_temp.append(2 * slope)
			# self.state_batch.append(self.prev_action)
			# self.progress_batch.append(slope * 2)
			self.ns_mean += 0.05 * 2 * slope

	def UpdateTrainingDataProgress(self):
		to_delete = []
		state_temp = []
		progress_temp = []
		for j in range(len(self.state_batch)):
			to_delete.append(False)
		for i in range(len(self.ns_slope_sample)):
			state_temp.append(self.ns_slope_sample[i])
			progress_temp.append(self.ns_slope_temp[i])
			for j in range(len(self.state_batch)):
				if not to_delete[j] and np.linalg.norm(self.state_batch[j] - self.ns_slope_sample[i]) < 2*1e-1:
					to_delete[j] = True

		for i in range(len(self.state_batch)):
			if not to_delete[i]:
				state_temp.append(self.state_batch[i])
				progress_temp.append(self.progress_batch[i])

		self.state_batch = copy(state_temp)
		self.progress_batch = copy(progress_temp)
		
		self.ns_slope_sample = []
		self.ns_slope_temp = []

	def GetTrainingDataProgress(self, augment=False):
		if not augment:
			return np.array(copy(self.state_batch)), np.array(copy(self.progress_batch))
		else:
			aug_x = []
			aug_y = []
			for _ in range(200):
				tp = self.sim_env.UniformSample(False, True)
				aug_x.append(tp)
				aug_y.append(0)

			for _ in range(200):
				tp = self.sim_env.UniformSample(True, False)
				aug_x.append(tp)
				aug_y.append(0)
			if len(self.state_batch) == 0:
				return np.array(aug_x), np.array(aug_y)

			aug_x = np.concatenate((self.state_batch, aug_x))
			aug_y = np.concatenate((self.progress_batch, aug_y))
			aug_x = np.concatenate((self.state_batch, aug_x))
			aug_y = np.concatenate((self.progress_batch, aug_y))

			return aug_x, aug_y

	def ClearTrainingDataProgress(self, all):
		if not all and len(self.state_batch) >= 60:
			self.state_batch = self.state_batch[-60:]
			self.progress_batch = self.progress_batch[-60:]
		elif all:
			self.state_batch = []
			self.progress_batch = []

	def updateGoalDistribution(self, v_func, v_func_prev, results, idxs, visited, m=4, N=400):

		self.start += 1
		if visited:
			self.n_visit += 1
		else:
			self.n_explore += 1
			
		self.v_mean_cur = np.array(results).mean()
		if self.v_mean == 0:
			self.v_mean = self.v_mean_cur
		else:
			self.v_mean = 0.5 * self.v_mean + 0.5 * self.v_mean_cur


		if self.start <= 2 and self.type_explore == 3:
			v_mean_sample_cur = [0] * len(self.sample)
			count_sample_cur = [0] * len(self.sample)

			for i in range(len(results)):
				v_mean_sample_cur[idxs[i]] += results[i]
				count_sample_cur[idxs[i]] += 1
				
			for i in range(len(self.sample)):
				if count_sample_cur[i] != 0:
					self.v_prev_sample[i] = copy(self.v_sample[i])
					self.v_sample[i] = v_mean_sample_cur[i] / count_sample_cur[i]
				else:
					self.v_prev_sample[i] = copy(self.v_sample[i])
					self.v_sample[i] = -1

			if self.start == 2:
				self.regression_x = []
				self.regression_y = []
				for i in range(len(self.sample)):
					progress = self.v_sample[i] - self.v_prev_sample[i]
					if self.v_sample[i] != -1 and self.v_prev_sample[i] != -1:
				#	if self.v_prev_sample[i] > self.v_mean_uniform - 0.5 * self.v_std_uniform:
						self.regression_x.append(self.v_prev_sample[i])
						self.regression_y.append(progress)
				
				x = copy(self.regression_x)
				y = copy(self.regression_y)
				x.append(self.v_mean_uniform - self.v_std_uniform)
				y.append(-0.1)
				x =  np.array(x).reshape(-1, 1)
				self.regressor.fit(x, y)
				self.mode = 0

				x = np.linspace(0.8, 1.2, num=10)
				x_ = np.array(x).reshape(-1, 1)
				y = self.regressor.predict(x_)
				print('progress by regressor: ', y)

				self.pool_ex = []
				self.idx_ex = []
				for i in range(m):
					x_cur = self.randomSample(visited)
					for j in range(int(N/m)):
						self.pool_ex.append(x_cur)
						x_new = self.randomSample(visited)
						alpha = min(1.0, self.probAdaptive2(v_func, x_new)/self.probAdaptive2(v_func, x_cur))

						if np.random.rand() <= alpha:          
							x_cur = x_new
				# for i in range(len(self.pool_ex)):
				# 	v = v_func.getValue([self.pool_ex[i]])
				# 	v = v.reshape(-1, 1)
				# 	print(v[0], self.regressor.predict(v)[0], self.probAdaptive2(v_func, self.pool_ex[i]))
				self.regression_x = []
				self.regression_y = []

		elif self.start % 5 == 0 and self.type_explore == 3:
			x = copy(self.regression_x)
			y = copy(self.regression_y)
			# x.append(self.v_mean_uniform - self.v_std_uniform)
			# y.append(-0.1)
			x =  np.array(x).reshape(-1, 1)
			self.regressor.fit(x, y)

			x = np.linspace(0.8, 1.2, num=10)
			x_ = np.array(x).reshape(-1, 1)
			y = self.regressor.predict(x_)
			print('progress by regressor: ', y)
			self.regression_x = []
			self.regression_y = []
		if self.start == 1:
			self.v_mean_uniform = self.v_mean_cur
			self.v_std_uniform = np.std(np.array(results))
			print(self.v_mean_uniform, self.v_std_uniform)
			return

		if self.start % 20 == 0:
			self.mode = 2
		elif self.start % 20 == 1 and self.start != 1:
			self.v_mean_uniform = self.v_mean_cur
			self.v_std_uniform = np.std(np.array(results))
			print(self.v_mean_uniform, self.v_std_uniform)

			li = copy(results)
			li = np.sort(np.array(li))
			size = len(li)
			mean_bottom = li[:int(size*0.2)].mean()
			mean_top = li[int(size*0.8):].mean()
			if not os.path.isfile(self.exploration_test_print):
				out = open(self.exploration_test_print, "w")
				out.write('ratio: '+str(self.sim_env.GetVisitedRatio())+'\n')
				out.write('v goals: '+str(self.v_mean_cur)+', '+str(mean_bottom)+','+str(mean_top)+','+str(self.v_std_uniform)+'\n')
				out.close()
			else:
				out = open(self.exploration_test_print, "a")
				out.write('ratio: '+str(self.sim_env.GetVisitedRatio())+'\n')
				out.write('v goals: '+str(self.v_mean_cur)+', '+str(mean_bottom)+','+str(mean_top)+','+str(self.v_std_uniform)+'\n')
				out.close()	
			self.mode = 0
			if self.type_explore == 3:
				return
		if self.start == 2:
			self.mode = 0

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
			visited = True

			if self.type_explore == 0:
				return

			if self.type_explore == 3:
				if self.n_count % 2 == 1:
					v_mean_sample_cur = [0] * len(self.sample)
					count_sample_cur = [0] * len(self.sample)

					for i in range(len(results)):
						v_mean_sample_cur[idxs[i]] += results[i]
						count_sample_cur[idxs[i]] += 1
				
					self.v_prev_sample = []
					for i in range(len(self.sample)):
						if count_sample_cur[i] == 0:
							self.v_prev_sample.append(-1)
						else:
							self.v_prev_sample.append(v_mean_sample_cur[i] / count_sample_cur[i])
					if self.start > 1 and self.start % 20 != 0:
						self.n_count += 1
					return

				elif self.n_count % 2 == 0:
					if self.start != 2:
						v_mean_sample_cur = [0] * len(self.sample)
						count_sample_cur = [0] * len(self.sample)

						for i in range(len(results)):
							v_mean_sample_cur[idxs[i]] += results[i]
							count_sample_cur[idxs[i]] += 1
					
						self.v_sample = []
						v_diff = []
						for i in range(len(self.sample)):
							if count_sample_cur[i] != 0:
								self.v_sample.append(v_mean_sample_cur[i] / count_sample_cur[i])
								v_diff.append(self.v_sample[i] - self.v_prev_sample[i])
							#	if self.v_prev_sample[i] > self.v_mean_uniform -  0.5 * self.v_std_uniform:
								if self.v_prev_sample[i] != -1:
									self.regression_x.append(self.v_prev_sample[i])
									self.regression_y.append(v_diff[i])
						print(self.v_prev_sample)
						print(v_diff)
					
					self.pool_ex = []
					for i in range(4):
						x_cur = self.randomSample(visited)
						for j in range(50):
							self.pool_ex.append(x_cur)
							x_new = self.randomSample(visited)
							alpha = min(1.0, self.probAdaptive2(v_func, x_new)/self.probAdaptive2(v_func, x_cur))

							if np.random.rand() <= alpha:          
								x_cur = x_new
					# for i in range(len(self.pool_ex)):
					# 	v = v_func.getValue([self.pool_ex[i]])
					# 	v = v.reshape(-1, 1)
					# 	print(v[0], self.regressor.predict(v)[0], self.probAdaptive2(v_func, self.pool_ex[i]))

					self.sample = []
					m = int(20 / self.iter)
					for _ in range(m):
						t = np.random.randint(len(self.pool_ex))	
						self.sample.append(self.pool_ex[t])
					self.sample_counter = 0
					if self.start > 1 and self.start % 20 != 0:
						self.n_count += 1
					return

			self.pool_ex = []
			self.idx_ex = []

			if self.type_explore == 1 or self.type_explore == 8:
				for i in range(m):
					x_cur = self.randomSample(visited)
					for j in range(int(N/m)):
						self.pool_ex.append(x_cur)
						x_new = self.randomSample(visited)
						if self.type_explore == 1:
							alpha = min(1.0, self.probAdaptive(v_func, x_new, visited)/self.probAdaptive(v_func, x_cur, visited))
						else:
							alpha = min(1.0, self.probTS(v_func, v_func_prev, x_new, visited)/self.probTS(v_func, v_func_prev, x_cur, visited))

						if np.random.rand() <= alpha:          
							x_cur = x_new
				self.ns_mean = 0
			else:
				if self.type_explore == 9:
					if self.start == 1:
						self.init_mean = self.v_mean_cur
					elif self.start == 2:
						progress_str = ''
						for i in range(len(self.progress) - 1):
							progress_str += str(self.progress[i+1] - self.progress[i]) +' '
						if not os.path.isfile(self.exploration_test_print):
							out = open(self.exploration_test_print, "w")
							out.write(str(self.init_mean)+' / '+str(self.v_mean)+' / '+str(self.v_mean-self.init_mean)+'\n')
							out.write(progress_str+'\n')
							out.close()
						else:
							out = open(self.exploration_test_print, "a")
							out.write(str(self.init_mean)+' / '+str(self.v_mean)+' / '+str(self.v_mean-self.init_mean)+'\n')
							out.write(progress_str+'\n')
							out.close()	
			
				v_mean_sample_cur = [0] * len(self.sample)
				count_sample_cur = [0] * len(self.sample)

				for i in range(len(results)):
					v_mean_sample_cur[idxs[i]] += results[i]
					count_sample_cur[idxs[i]] += 1
				
				v_prev_sample_print = ""
				v_sample_print = ""	
				for i in range(len(self.sample)):
					# d = self.sim_env.GetDensity(self.sample[i]) 
					# if d > 0.3:
					# 	print(self.sample[i], d)
					# 	self.sample[i] = self.randomSample(visited)
					# 	if self.type_explore == 2:
					# 		self.v_sample[i] = 0.0
					# 		self.v_prev_sample[i] = 0.0
					# 	elif self.type_explore == 4:
					# 		self.v_sample[i] = 0.0
					# 		self.v_prev_sample[i] = 0.3
					# 	else:
					# 		self.v_sample[i] = 0.0
					# 		self.v_prev_sample[i] = 0.8
					# else:
					if count_sample_cur[i] != 0:
						self.v_prev_sample[i] = copy(self.v_sample[i])
						self.v_sample[i] = v_mean_sample_cur[i] / count_sample_cur[i]
					v_prev_sample_print += str(self.v_prev_sample[i]) +', '
					v_sample_print += str(self.v_sample[i]) +', '

				print('v prev goals: ', self.v_prev_sample)
				print('v goals: ', self.v_sample)
				if self.n_explore == 1:
					return
				if self.type_explore == 5:
					return

				if self.type_explore == 6 or self.type_explore == 7:
					for i in range(len(self.sample)):
						if self.ns_count_temp[i] != 0:
							w = min(1.0, 0.2 * self.ns_count_temp[i])
							self.ns_slope_sample[i] = (1-w) * self.ns_slope_sample[i] + w * ( 2 * self.ns_slope_temp[i] / self.ns_count_temp[i])
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
				target = self.randomSample(visited)
				t = -1

			if self.type_visit == 0:
				target = self.randomSample(visited)
				t = -1
			else:
				t = np.random.randint(len(self.pool))
				target = self.pool[t] 

			if self.type_explore == 8:
				self.prev_nsample = self.sim_env.GetNumSamples()
				self.prev_action = target

			return target, t
		else:
			####TO DELETE
			visited = True
			if self.start <= 1:
				t = math.floor(self.sample_counter /self.iter) % len(self.sample)
				target = self.sample[t]
				self.sample_counter += 1
				return target, t
			elif self.mode == 2:
				return self.randomSample(visited), -1
			if self.type_explore == 3:
				t = math.floor(self.sample_counter /self.iter) % len(self.sample)
				target = self.sample[t]
				self.sample_counter += 1
				return target, t
			if self.type_explore == 9:
				t = 0
				target = self.sample[0]
				return target, t
			if self.type_explore == 0:
				return self.randomSample(visited), -1
			elif self.type_explore == 5:
				t = np.random.randint(len(self.sample))	
				target = self.sample[t]
				idx = t
				return target, t
			elif self.type_explore == 1 or self.type_explore == 8:
				t = np.random.randint(len(self.pool_ex))
				target = self.pool_ex[t]
				return target, t 
			else:
				if self.n_explore <= 2:
					t = np.random.randint(len(self.sample))	
					target = self.sample[t]
					idx = target
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

	def sampleGoals(self, m=30):
		self.sample = []
		self.v_sample = []
		self.v_prev_sample = []
		self.ns_count_temp = []

		while len(self.sample) < m:
			r = self.randomSample(True)
			flag = True
			for s in self.sample:
				if np.linalg.norm(s - r) < 0.1:
					flag = False
					break
			if flag:
				self.sample.append(r)
		for i in range(m):
			####TO MODIFY
			#self.sample.append(self.randomSample(True))
			if self.type_explore == 2:
				self.v_sample.append(0.0)
				self.v_prev_sample.append(0.0)
			elif self.type_explore == 4:
				self.v_sample.append(0.8)
				self.v_prev_sample.append(0.8)
			else:
				self.v_sample.append(0.0)
				self.v_prev_sample.append(0.8)

			self.ns_slope_sample.append(0)
			self.ns_slope_temp.append(0)
			self.ns_count_temp.append(0)

		# print('new goals: ', self.sample)

	def reset_explore(self):
		self.ns_slope_sample = []
		self.ns_slope_temp = []
		self.sampleGoals()
		self.n_explore = 0
		self.n_count = 0

	def isEnough(self, v_func):
		
		self.random_start = False

		print("===========================================")
		print("mean reward : ", self.v_mean)
		print("===========================================")

		if self.n_visit % 5 == 4:
			self.printSummary(v_func)
			if self.n_visit > 5 and self.v_mean_cur > 0.9 and not self.done:
				return True
			if self.v_mean_cur > 1.1 and self.done:
				return True

		# if self.n_visit > 20 and not self.done:
		# 	return True
		
		if self.n_visit > 5 and self.v_mean > 0.9 and not self.done:
			return True

		if self.v_mean > 1.1 and self.done:
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