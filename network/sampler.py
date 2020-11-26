import numpy as np
import math
from regression import Regression
from IPython import embed
from copy import copy
class Sampler(object):
	def __init__(self, sim_env, dim, path):
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
		self.type_visit = 0
		# 0: uniform 1 :adaptive(network) 2:adaptive(sampling) 3:ts(network) 4:ts(sampling) 5: uniform(sampling)
		# 6: num sample slope(sampling) 7:num sample near goal(sampling) 8: num sample slope (network)
		self.type_explore = 0
		if 1:
			self.epsilon_greedy = True
		else:
			self.epsilon_greedy = False
		if 1:
			self.hard = True
		else:
			self.hard = False
		self.epsilon = 0.25

		self.state_batch = []
		self.progress_batch = []

		self.prev_action = 0
		self.prev_nsample = 0
		self.ns_mean = 0

		self.done = False

		self.vr_diff_explore = 0
		self.vr_diff_exploit = 0

		self.vr_prev_explore = 0
		self.vr_prev_exploit = 0

		self.test = True
		self.test_counter = 0
		print('=======================================')
		print('curriculum option')
		print('type visit', self.type_visit)
		print('type explore', self.type_explore)
		print('e greedy', self.epsilon_greedy)
		print('hard', self.hard)
		print('=======================================')

	def randomSample(self, visited=True):
		return self.sim_env.UniformSample(visited)
		
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
				tp = self.sim_env.UniformSample(False)
				aug_x.append(tp)
				aug_y.append(0)

			for _ in range(200):
				tp = self.sim_env.UniformSample(True)
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

	def updateGoalDistribution(self, v_func, v_func_prev, results, idxs, visited, m=2, N=400):
		self.start += 1
		if visited:
			self.n_visit += 1
		else:
			self.n_explore += 1
			
		self.v_mean_cur = np.array(results).mean()
		if self.v_mean == 0:
			self.v_mean = self.v_mean_cur
		else:
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
			if self.type_explore == 1 or self.type_explore == 3 or self.type_explore == 8:
				for i in range(m):
					x_cur = self.randomSample(visited)

					# while 1:
					# 	x_cur = self.randomSample(visited)
					# 	if v_func.getValue([x_cur])[0] > 0.5:
					# 		break
					for j in range(int(N/m)):
						self.pool_ex.append(x_cur)
						x_new = self.randomSample(visited)
						if self.type_explore == 1 or self.type_explore == 8:
							alpha = min(1.0, self.probAdaptive(v_func, x_new, False)/self.probAdaptive(v_func, x_cur, False))
						else:
							alpha = min(1.0, self.probTS(v_func, v_func_prev, x_new, False)/self.probTS(v_func, v_func_prev, x_cur, False))

						if np.random.rand() <= alpha:          
							x_cur = x_new
				# for i in range(len(self.pool_ex)):
				# 	print(self.pool_ex[i], v_func.getValue([self.pool_ex[i]])[0], self.probAdaptive(v_func, self.pool_ex[i], False))
				self.ns_mean = 0
			else:
				if self.n_explore == 1:
					return
				v_mean_sample_cur = [0] * len(self.sample)
				count_sample_cur = [0] * len(self.sample)

				for i in range(len(results)):
					v_mean_sample_cur[idxs[i]] += results[i]
					count_sample_cur[idxs[i]] += 1
					
				for i in range(len(self.sample)):
					d = self.sim_env.GetDensity(self.sample[i]) 
					if d > 0.3:
						print(self.sample[i], d)
						self.sample[i] = self.randomSample(visited)
						self.v_prev_sample[i] = 0.8
						self.v_sample[i] = 1.0
					else:
						if count_sample_cur[i] != 0:
							self.v_prev_sample[i] = copy(self.v_sample[i])
							self.v_sample[i] = v_mean_sample_cur[i] / count_sample_cur[i]

				print('v prev goals: ', self.v_prev_sample)
				print('v goals: ', self.v_sample)
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

	def saveProgress(self):
		t = self.test_counter % 10
		self.progress_sample[t] += self.sim_env.GetProgressGoal()
		self.test_counter += 1
	def adaptiveSample(self, visited):
		if self.test:
			t = self.test_counter % 10
			target = self.sample[t]
			return target, t

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
			if self.type_explore == 0:
				return self.randomSample(visited), -1
			elif self.type_explore == 5:
				t = np.random.randint(len(self.sample))	
				target = self.sample[t]
				idx = t
				return target, t
			elif self.type_explore == 1 or self.type_explore == 3 or self.type_explore == 8:
				if self.n_explore <= 2:
					target = self.randomSample(visited)
					t = -1
				else:
					t = np.random.randint(len(self.pool_ex))
					target = self.pool_ex[t]
				if self.type_explore == 8:
					self.prev_nsample = self.sim_env.GetNumSamples()
					self.prev_action = target
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
		self.v_prev_sample = []
		self.ns_count_temp = []
		for i in range(m):
			self.sample.append(self.randomSample(False))

			self.v_sample.append(1.0)
			self.v_prev_sample.append(0.8)

			self.ns_slope_sample.append(0)
			self.ns_slope_temp.append(0)
			self.ns_count_temp.append(0)

		# print('new goals: ', self.sample)

	def reset_explore(self):
		self.ns_slope_sample = []
		self.ns_slope_temp = []
		if self.type_explore != 0 and self.type_explore != 1 and self.type_explore != 3 and self.type_explore != 8:
			self.sampleGoals()
		self.n_explore = 0

	def sample_evaluation_points(self, v_func, lower_bound):
		self.sample = []
		self.progress_sample = []
		while len(self.sample) < 10:
			li = self.sim_env.UniformSampleWithNearestParams() 
			params = li[1]
			vs = v_func.getValue(params)
			mean = np.array(vs).mean()
			if mean > lower_bound and mean <= lower_bound + 0.05:
				self.sample.append(li[0])
				self.progress_sample.append(0)
				print(li[0], vs)


	def isEnough(self, v_func):
		
		self.random_start = False

		print("===========================================")
		print("mean reward : ", self.v_mean)
		print("===========================================")

		if self.n_visit % 5 == 4:
			self.printSummary(v_func)
			if self.n_visit > 5 and self.v_mean_cur > 0.85 and not self.done:
				return True
			if self.v_mean_cur > 1.3 and self.done:
				return True

		# if self.n_visit > 20 and not self.done:
		# 	return True
		
		if self.n_visit > 5 and self.v_mean > 0.85 and not self.done:
			return True

		if self.v_mean > 1.3 and self.done:
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