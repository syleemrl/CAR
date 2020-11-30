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
		self.vp_table = [[1.0, 5, 0]]
		self.vp_list_explore = []
		self.eval_target_v = 0

		self.progress_queue_evaluation = []
		self.progress_queue_exploit = [5.0]
		self.progress_queue_explore = [0]

		self.progress_cur = 0
		self.evaluation_counter = 0
		self.evaluation_done = False
		self.eval_frequency = 0
		self.scale = 1.0 / 0.05

		self.reg = LinearRegression()
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
		

		it = 0
		while it < len(self.vp_table):
			if mode == 0:
				self.vp_table[it][2] += 3
				self.vp_table[it][1] *= 0.95
			else:
				self.vp_table[it][2] += 1
			if self.vp_table[it][2] >= 30:
				self.vp_table = self.vp_table[:it] + self.vp_table[(it+1):]  
			else:
				it += 1

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
				self.evaluation_done = True

				x = np.array(self.vp_list_explore)[:,0]
				y = np.array(self.vp_list_explore)[:,1]

				x = x.reshape((-1, 1))
				y = np.array(y)
				self.reg.fit(x, y)

				x_predict = np.array([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
				x_predict = x_predict.reshape((-1, 1))
				y_predict = self.reg.predict(x_predict)

				print(y_predict)
	def saveProgress(self, mode):
		if mode == 0:
			p = self.sim_env.GetProgressGoal()
			self.progress_cur += p
			if self.eval_target_v < 0.7:
				return

			if self.total_iter >= 3:
				flag = False
				for i in range(len(self.vp_table)):
					if abs(self.vp_table[i][0] - self.eval_target_v) < 1e-2:
						rate = 0.1 + 0.02 * self.vp_table[i][2]
						self.vp_table[i][1] = (1 - rate) * self.vp_table[i][1] + rate * p * 10
						self.vp_table[i][2] = 0
						flag = True
				if not flag:
					self.vp_table.append([self.eval_target_v, 0.5 * p * 10, 0])

				if len(self.vp_list_explore) >= 50:
					self.vp_list_explore = self.vp_list_explore[5:]
				self.vp_list_explore.append([self.eval_target_v, p * 10])
		elif mode == 2:
			p = self.sim_env.GetProgressGoal()
			t = self.evaluation_counter % len(self.sample)
			lb = self.sample[t][1]
			for i in range(len(self.sample_progress)):
				if abs(self.sample_progress[i][1] - lb) < 1e-2:
					self.sample_progress[i][0] += p
					break 
			self.evaluation_counter += 1
			self.vp_list_explore.append([self.eval_target_v, p * 10])
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

		x = np.array(self.vp_list_explore)[:,0]
		y = np.array(self.vp_list_explore)[:,1]

		x = x.reshape((-1, 1))
		y = np.array(y)
		self.reg.fit(x, y)

		x_predict = np.array([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
		x_predict = x_predict.reshape((-1, 1))
		y_predict = self.reg.predict(x_predict)

		print(y_predict)

	def resetExplore(self):
		self.n_explore = 0
		self.prev_progress = np.array(self.progress_queue_explore).mean()
		self.progress_queue_explore = []
		self.vp_list_explore = []
	
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
			self.v_sample.append(round(v * self.scale) / self.scale)
			self.sample.append(target)


	def resetEvaluation(self, v_func):
		self.n_evaluation = 0
		self.eval_target_v = max(0.75, round((self.v_mean - 0.05) * self.scale) / self.scale)
		self.sample = []
		self.sample_progress = []
		self.vp_list_explore = []
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
			if self.eval_target_v > self.v_mean + 0.1:
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


		if self.n_exploit < 10:
			return False

		v = round(self.v_mean_cur * self.scale) / self.scale
		y0 = self.reg.predict([[v]])
		y1 = self.reg.predict([[v -  1 / self.scale]])
		print(v, y0, v -  1 / self.scale, y1)
		if y1 > y0:
			if p_mean < np.array(self.progress_queue_explore).mean() * 0.8:
				return True
		else:
			if self.n_exploit % 5 == 4 and p_mean < y1 * 0.8:
				return True
			if self.n_exploit % 5 != 4 and p_mean < y0 * 0.8:
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