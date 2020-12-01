import numpy as np
import math
from regression import Regression
from IPython import embed
from copy import copy
from sklearn.linear_model import LinearRegression

class Sampler(object):
	def __init__(self, sim_env, dim, path):
		self.sim_env = sim_env
		self.dim = dim
		
		self.v_mean = 1.0
		self.v_mean_boundary = 0.0
		self.d_cur = 0
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
		self.v_list_explore = []
		self.p_list_explore = []
		self.scale = 0.8

		self.eval_target_v = 0

		self.progress_queue_evaluation = []
		self.progress_queue_exploit = [15.0]
		self.progress_queue_explore = [0]

		self.progress_cur = 0
		self.evaluation_counter = 0
		self.evaluation_done = False
		self.eval_frequency = 0

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

	def updateGoalDistribution(self, mode, v_func, m=2, N=400):
		if mode == 0:
			self.n_explore += 1
		elif mode == 1:
			self.n_exploit += 1
		else:
			self.n_evaluation += 1
		self.total_iter += 1
		

		# it = 0
		# while it < len(self.vp_table):
		# 	if mode == 0:
		# 		self.vp_table[it][2] += 3
		# 		self.vp_table[it][1] *= 0.95
		# 	else:
		# 		self.vp_table[it][2] += 1
		# 	if self.vp_table[it][2] >= 30:
		# 		self.vp_table = self.vp_table[:it] + self.vp_table[(it+1):]  
		# 	else:
		# 		it += 1

		if mode == 0:
			if self.total_iter < 3:
				return
			if self.type_explore == 0 and self.n_explore % 2 == 1:
				self.sampleBatch(v_func, self.type_explore)
			elif self.type_explore == 1 and self.n_explore % 2 == 1:
				self.pool_ex = []
				for i in range(m):
					x_cur = self.randomSample(mode)
					for j in range(int(N/m)):
						self.pool_ex.append(x_cur)
						x_new = self.randomSample(mode)
						alpha = min(1.0, self.probAdaptive(v_func, x_new, False)/self.probAdaptive(v_func, x_cur, False))
						
						if np.random.rand() <= alpha:          
							x_cur = x_new
				self.sampleBatch(v_func, self.type_explore)
	
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
				self.evaluation_done = True

				self.printExplorationRateData()

	def saveProgress(self, mode):
		if mode == 0 or mode == 2:
			p = self.sim_env.GetProgressGoal()
			self.progress_cur += p
			if self.eval_target_v < 0.7:
				return

			if self.total_iter >= 3:
				flag = False

				if mode != 2 and len(self.v_list_explore) >= 80:
					self.v_list_explore = self.v_list_explore[5:]
					self.p_list_explore = self.p_list_explore[5:]

				self.v_list_explore.append(self.eval_target_v)
				self.p_list_explore.append(p * 10)
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
			if self.total_iter < 3:
				target = self.randomSample(mode)
				t = -1
			elif self.type_explore == 0:
				# li = self.sim_env.UniformSampleWithNearestParams() 
				# params = li[1]
				# vs = v_func.getValue(params)
				# v = np.array(vs).mean()
				# target = li[0]
				# self.eval_target_v = round(v * self.scale) / self.scale
				target = self.sample[self.sample_counter % len(self.sample)]
				self.eval_target_v = self.v_sample[self.sample_counter % len(self.sample)]
				self.sample_counter += 1
				t = -1
			else:
				# t = np.random.randint(len(self.pool_ex)) 
				# target = self.pool_ex[t] 
				# target_np = np.array(target, dtype=np.float32) 
				# params = self.sim_env.GetNearestParams(target_np) 
				# vs = v_func.getValue(params)
				# v = np.array(vs).mean()
				# self.eval_target_v = round(v * self.scale) / self.scale
				target = self.sample[self.sample_counter % len(self.sample)]
				self.eval_target_v = self.v_sample[self.sample_counter % len(self.sample)]
				self.sample_counter += 1
				t = -1

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
			self.eval_target_v = self.v_sample[t]
			self.evaluation_counter += 1

			return target, t

	def resetExploit(self):
		self.n_exploit = 0
		self.prev_progress = np.array(self.progress_queue_exploit).mean()
		self.progress_queue_exploit = []

		self.printExplorationRateData()

	def resetExplore(self):
		self.n_explore = 0
		self.prev_progress = np.array(self.progress_queue_explore).mean()
		self.progress_queue_explore = []
		self.v_list_explore = []
		self.p_list_explore = []

	def sampleBatch(self, v_func, type_explore):
		self.sample = []
		self.v_sample = []
		self.sample_counter = 0
		for _ in range(20):
			if type_explore == 0:
				li = self.sim_env.UniformSampleWithNearestParams() 
				params = li[1]
				vs = v_func.getValue(params)
				v = np.array(vs).mean()
				target = li[0]
			else:
				t = np.random.randint(len(self.pool_ex)) 
				target = self.pool_ex[t] 
				target_np = np.array(target, dtype=np.float32) 
				params = self.sim_env.GetNearestParams(target_np) 
				vs = v_func.getValue(params)
				v = np.array(vs).mean()
			self.v_sample.append(v)
			self.sample.append(target)


	def resetEvaluation(self, v_func):
		self.n_evaluation = 0
		self.sample = []
		self.v_sample  = []
		self.v_list_explore = []
		self.p_list_explore = []
		self.evaluation_done = False
		self.evaluation_counter = 0

		while len(self.sample) < 40:
			li = self.sim_env.UniformSampleWithNearestParams() 
			params = li[1]
			vs = v_func.getValue(params)
			v = np.array(vs).mean()
			t = li[0]
			self.sample.append(t)
			self.v_sample.append(v)


		self.eval_frequency = len(self.sample) + 1

	def printExplorationRateData(self):
		self.v_list_explore, self.p_list_explore = (list(t) for t in zip(*sorted(zip(self.v_list_explore, self.p_list_explore))))
		# for v, p in zip(self.v_list_explore, self.p_list_explore): 
		# 	print(v, p, end=' ')
		# print()

		v_min = np.array(self.v_list_explore).min() + self.scale
		v_max = np.array(self.v_list_explore).max() - self.scale

		v_predict = np.linspace(v_min, v_max, 10)
		for v in v_predict:
			mean, count = self.predictWindow(v, self.scale)
			if count < 5:
				mean, count = self.predictWindow(v, self.scale * 2)

				if count < 5:
					mean = -1
				else:
					mean /= count		
			else:
				mean /= count	
			print(v, ':', mean, end='; ')
		print()
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

	def isEnough(self, results, density):
		self.v_mean_cur = np.array(results).mean()
		if self.v_mean == 0:
			self.v_mean = self.v_mean_cur
		else:
			self.v_mean = 0.6 * self.v_mean + 0.4 * self.v_mean_cur

		p_mean = np.array(self.progress_queue_exploit).mean()

		v_mean_boundary_cur = 0
		count = 0
		for i in range(len(results)):
			if density[i] > 0.6 and density[i] < 0.75:
				count += 1
				v_mean_boundary_cur += results[i]
				
		if count != 0:
			v_mean_boundary_cur /= count
			if self.v_mean_boundary == 0:
				self.v_mean_boundary = v_mean_boundary_cur
			else:
				rate = min(1, count * 0.001)
				self.v_mean_boundary = (1- rate) * self.v_mean_boundary + rate * v_mean_boundary_cur


		print(self.progress_queue_exploit)
		print("===========================================")
		print("mean reward : ", self.v_mean)
		print("current mean boundary reward: ", v_mean_boundary_cur, count)
		print("mean boundary reward : ", self.v_mean_boundary)
		print("exploration rate : ", p_mean)
		print("===========================================")


		if self.n_exploit < 10:
			return False

		mean, count = self.predictWindow(self.v_mean_boundary, self.scale)
		if count < 5:
			mean, count = self.predictWindow(self.v_mean_boundary,  self.scale * 2)

			if count < 5:
				mean = np.array(self.p_list_explore).mean()
			else:
				mean /= count		
		else:
			mean /= count
		print(p_mean, mean)
		if p_mean < mean * 0.8:
			return True
	
		# for i in reversed(range(len(self.vp_table))):
		# 	if self.n_exploit % 5 == 4 and self.vp_table[i][0] <= v - 1 / self.scale and p_mean < self.vp_table[i][1] * 0.8:
		# 		return True
		# 	if self.n_exploit % 5 != 4 and self.vp_table[i][0] <= v and p_mean < self.vp_table[i][1] * 0.8:
		# 		return True
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