from env import Env
from sampler import Sampler
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy
from utils import RunningMeanStd
from IPython import embed
import os
def vector_to_str(vec):
	s = ''
	for v in vec:
		s += str(v) + ' '
	return s
def matrix_to_str(mat):
	s = ''
	for m in mat:
		s += '(' + vector_to_str(m) + ')'
	return s
class Monitor(object):
	def __init__(self, ref, num_slaves, directory, adaptive, parametric, plot=True, verbose=True):
		self.env = Env(ref, directory, adaptive, parametric, num_slaves)
		self.num_slaves = self.env.num_slaves
		self.sim_env = self.env.sim_env
		
		self.num_state = self.env.num_state
		self.num_action = self.env.num_action
		self.RMS = RunningMeanStd(shape=(self.num_state))	
		self.verbose = verbose
		self.plot = plot
		self.directory = directory
		self.adaptive = adaptive
		self.parametric = parametric

		self.start_time = time.time()		
		self.num_evaluation = 0
		self.num_episodes = 0
		self.num_transitions = 0
		self.total_frames_elapsed = 0
		self.total_rewards = []
		self.max_episode_length = 0

		self.reward_label = self.sim_env.GetRewardLabels()
		self.total_rewards_by_parts = np.array([[]]*len(self.reward_label))
		self.transition_per_episodes = []
		self.num_nan_per_iteration = 0
		self.num_episodes_per_iteration = 0
		self.num_transitions_per_iteration = 0
		self.rewards_per_iteration = 0
		self.rewards_by_part_per_iteration = []

		self.terminated = [False]*self.num_slaves
		self.states = [0]*self.num_slaves
		self.prevframes = [0]*self.num_slaves
		
		self.rewards_dense_phase = [0]*self.num_slaves
		self.rewards_sparse_phase = [0]*self.num_slaves

		self.phaselength = self.sim_env.GetPhaseLength()
		self.dim_param = len(self.sim_env.GetParamGoal())
		self.sampler = Sampler(self.sim_env, self.dim_param)

		self.mode_counter = 0
		self.v_ratio = 0
		self.mode = 0
		self.mode_eval = False

		if self.plot:
			plt.ion()

	def getStates(self):
		return np.array(self.states).astype('float32') 

	def setTerminated(self, idx):
		self.terminated[idx] = True
	
	def getTerminated(self, idx):
		return self.terminated[idx]
	
	def getAllTerminated(self):
		for i in range(self.num_slaves):
			if not self.terminated[i]:
				return False
		return True 
	
	def reset(self, i, b=True):
		self.env.reset(i, b)
		state = np.array([self.sim_env.GetState(i)])
		self.states[i] = self.RMS.apply(state)[0]
		self.terminated[i] = False
		self.prevframes[i] = 0

	def step(self, actions, record=True):
		self.states, rewards, dones, times, frames, terminal_reason, nan_count =  self.env.step(actions)

		if self.adaptive and self.parametric:
			params = np.array(self.states)[:,-self.dim_param:]
			curframes = np.array(self.states)[:,-(self.dim_param+1)]
		else:
			params = np.zeros(self.num_slaves)
			curframes = np.array(self.states)[:,-1]

		states_updated = self.RMS.apply(self.states[~np.array(self.terminated)])
		self.states[~np.array(self.terminated)] = states_updated
		if record:
			self.num_nan_per_iteration += nan_count
			for i in range(self.num_slaves):
		
				if not self.terminated[i] and rewards[i][0] is not None:
					self.rewards_per_iteration += rewards[i][0]
					self.rewards_by_part_per_iteration.append(rewards[i])
					self.num_transitions_per_iteration += 1

					if dones[i]:
						self.num_episodes_per_iteration += 1
						self.total_frames_elapsed += frames[i]

						if frames[i] > self.max_episode_length:
							self.max_episode_length = frames[i]
			self.prevframes = curframes

		if self.adaptive:
			rewards = [[rewards[i][0], rewards[i][1]] for i in range(len(rewards))]
		else:	
			rewards = [rewards[i][0] for i in range(len(rewards))]
				
		return rewards, dones, curframes, params

	def updateReference(self):
		self.num_evaluation += 1
		if self.num_evaluation % 2 == 0:
			self.sim_env.UpdateParamState()
		if self.num_evaluation % 10 == 0:
			self.sim_env.SaveParamSpace(-1)
			self.env.sim_env.UpdateReference()
			self.sim_env.TrainRegressionNetwork()
	
	def saveEvaluation(self, li):
		mean = np.array(li).mean()
		self.sampler.v_mean_boundary = 0.6 * self.sampler.v_mean_boundary + 0.4 * mean
		print('evaluation mean: ', mean)
		if self.mode == 0:
			if len(self.sampler.progress_queue_explore) == 0:
				update_rate = np.array(self.sampler.progress_queue_exploit).mean()
			else:
				update_rate = np.array(self.sampler.progress_queue_explore).mean()
		else:
			if len(self.sampler.progress_queue_exploit) == 0:
				update_rate = np.array(self.sampler.progress_queue_explore).mean()
			else:
				update_rate = np.array(self.sampler.progress_queue_exploit).mean()

		self.v_ratio = self.sim_env.GetVisitedRatio()
		f_mean = self.sim_env.GetFitnessMean()

		if not os.path.isfile(self.directory+"curriculum_info") :
			out = open(self.directory+"curriculum_info", "w")
			out.write(str(self.num_evaluation)+':'+str(self.num_episodes)+':'+str(self.mode)+'\n')
			out.write(str(mean)+':'+str(update_rate)+':'+str(self.v_ratio)+':'+str(f_mean)+'\n')
			out.close()
		else:
			out = open(self.directory+"curriculum_info", "a")
			out.write(str(self.num_evaluation)+':'+str(self.num_episodes)+':'+str(self.mode)+'\n')
			out.write(str(mean)+':'+str(update_rate)+':'+str(self.v_ratio)+':'+str(f_mean)+'\n')
			out.close()			

	def saveVPtable(self):
		if not os.path.isfile(self.directory+"vp_table") :
			out = open(self.directory+"vp_table", "w")
			out.write(str(self.num_evaluation)+':'+str(self.num_episodes)+':'+str(self.mode)+'\n')
			for k, v in zip(self.sampler.vp_dict.keys(), self.sampler.vp_dict.values()):
				out.write('(' + str(k * self.sampler.unit)+ ','+ str((k+1)*self.sampler.unit) + ') : ' + matrix_to_str(v)+ ' / '+str(np.array(v).mean(axis=0)[1]))
				out.write('\n')
			out.write('\n')

			#out.write(matrix_to_str(self.sampler.vp_table)+'\n')
			out.close()
		else:
			out = open(self.directory+"vp_table", "a")
			out.write(str(self.num_evaluation)+':'+str(self.num_episodes)+':'+str(self.mode)+'\n')
			for k, v in zip(self.sampler.vp_dict.keys(), self.sampler.vp_dict.values()):
				out.write('(' + str(k * self.sampler.unit)+ ','+ str((k+1)*self.sampler.unit) + ') : ' + matrix_to_str(v)+ ' / '+str(np.array(v).mean(axis=0)[1]))
				out.write('\n')
			out.write('\n')

		#	out.write(matrix_to_str(self.sampler.vp_table)+'\n')
			out.close()	

	def saveParamSpaceSummary(self, v_func):
		self.sim_env.SaveParamSpace(self.num_evaluation)
		li = self.sim_env.GetParamSpaceSummary()
		grids = li[0]
		grids_norm = li[1]
		fitness = li[2]
		density = li[3]
		v_values = v_func.getValue(grids)
		
		out = open(self.directory+"param_summary"+str(self.num_evaluation), "w")
		for i in range(len(grids)):
			out.write(vector_to_str(grids_norm[i])+' '+str(v_values[i])+' '+str(density[i])+' '+str(fitness[i])+'\n')
		out.close()		

		self.v_ratio = self.sim_env.GetVisitedRatio()
		
		if not os.path.isfile(self.directory+"v_ratio") :
			out = open(self.directory+"v_ratio", "w")
			out.write(str(self.num_episodes)+':'+str(self.mode)+':'+str(self.v_ratio)+'\n')
			out.close()
		else:
			out = open(self.directory+"v_ratio", "a")
			out.write(str(self.num_episodes)+':'+str(self.mode)+':'+str(self.v_ratio)+'\n')
			out.close()		

	def updateCurriculum(self, v_func, results, info):
		self.sampler.updateCurrentStatus(self.mode, results, info)

		mode_change = 0
		if not self.mode_eval:
			self.mode_counter += 1
		
		#self.sim_env.UpdateParamState()
			# self.saveParamSpaceSummary(v_func)
		if self.num_evaluation % 30 == 29:
			self.sim_env.SaveParamSpace(-1)

		if self.mode == 0:
			# if self.mode_counter % 5 == 0 and self.num_evaluation > 3:
			# 	self.saveVPtable()
			print(self.sampler.progress_queue_explore)
			print(np.array(self.sampler.progress_queue_explore).mean(), np.array(self.sampler.progress_queue_exploit).mean())
			if self.num_evaluation >= 20 and self.sampler.n_explore >= 10 and \
			   np.array(self.sampler.progress_queue_explore).mean() <= np.array(self.sampler.progress_queue_exploit).mean() * 0.9:
				self.mode = 1
				self.mode_counter = 0
				self.sampler.resetExploit()
				# if self.mode_counter % 5 != 0:
				# 	self.saveVPtable()
				mode_change = 1
			elif self.mode_counter % 30 == 29:
				self.mode = 1
				self.mode_eval = True
				self.sampler.resetExploit()
		elif self.mode == 1:
			if self.mode_eval and len(self.sampler.progress_queue_exploit) >= 2:
				self.mode = 0
				self.mode_eval = False
				if self.sampler.prev_progress_ex > np.array(self.sampler.progress_queue_exploit).mean():
					self.sampler.progress_queue_exploit = self.sampler.prev_queue_exploit
			elif not self.mode_eval and self.sampler.isEnough():
				self.mode = 0
				self.mode_counter = 0
				self.sampler.resetExplore()
				mode_change = 1
			elif not self.mode_eval and self.mode_counter % 30 == 29:
				self.sampler.resetEvaluation()
				self.mode = 2
				self.mode_eval = True
				mode_change = 1
		elif self.mode == 2 and self.sampler.evaluation_done:
			self.mode_eval = False
			self.mode = 1
			# self.saveVPtable()

		if self.v_ratio == 1:
			mode_change = -1	

		self.sampler.updateGoalDistribution(self.mode, v_func)

		return mode_change

	def needEvaluation(self):
		if self.num_evaluation > 5 and self.num_evaluation % 20 == 0:
			return True

	def updateGoal(self, v_func, record=True):
		if record:
			# t = self.sampler.randomSample(-1)
			# idx = -1
			t, idx = self.sampler.adaptiveSample(self.mode, v_func)
		else:
			t = self.sampler.randomSample(1)
			idx = -1
		
		t = np.array(t, dtype=np.float32) 
		self.sim_env.SetGoalParameters(t, self.mode)
		
		if record:		
			t = np.reshape(t, (-1, self.dim_param))
			v = v_func.getValue(t)[0]

			print(t[0], v)
		return idx

	def plotFig(self, y_list, title, num_fig=1, ylim=True, path=None):
		if self.plot:
			plt.figure(num_fig, clear=True, figsize=(5.5, 4))
		else:
			plt.figure(num_fig, figsize=(5.5, 4))
		plt.title(title)

		i = 0
		for y in y_list:
			plt.plot(y[0], label=y[1])
			i+= 1
		plt.legend(loc=2)
		if self.plot:
			plt.show()
			if ylim:
				plt.ylim([0,1])
			plt.pause(0.001)
		if path is not None:
			plt.savefig(path, format="png")

	def printSummary(self, save=True):
		t_per_e = self.num_transitions_per_iteration / self.num_episodes_per_iteration
		r_per_e = self.rewards_per_iteration/self.num_episodes_per_iteration
		rp_per_i = np.array(self.rewards_by_part_per_iteration).sum(axis=0) / self.num_transitions_per_iteration

		if save:

			self.total_rewards.append(r_per_e)
		
			self.num_transitions += self.num_transitions_per_iteration
			self.num_episodes += self.num_episodes_per_iteration
			self.num_evaluation += 1
			self.total_rewards_by_parts = np.insert(self.total_rewards_by_parts, self.total_rewards_by_parts.shape[1], 
				np.asarray(self.rewards_by_part_per_iteration).sum(axis=0)/self.num_episodes_per_iteration, axis=1)
			print_list = []
			print_list.append('===============================================================')
			print_list.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
			print_list.append("Elapsed time : {:.2f}s".format(time.time() - self.start_time))
			print_list.append('Num eval : {}'.format(self.num_evaluation))
			print_list.append('total episode count : {}'.format(self.num_episodes))
			print_list.append('total transition count : {}'.format(self.num_transitions))
			t_per_e = 0
			if self.num_episodes is not 0:
				t_per_e = self.num_transitions / self.num_episodes

			print_list.append('total transition per episodes : {:.2f}'.format(t_per_e))

			print_list.append('episode count : {}'.format(self.num_episodes_per_iteration))
			print_list.append('transition count : {}'.format(self.num_transitions_per_iteration))
			
			t_per_e = 0
			if self.num_episodes_per_iteration is not 0:
				t_per_e = self.num_transitions_per_iteration / self.num_episodes_per_iteration
			self.transition_per_episodes.append(t_per_e)

			print_list.append('transition per episodes : {:.2f}'.format(t_per_e))
			print_list.append('rewards per episodes : {:.2f}'.format(self.total_rewards[-1]))
			print_list.append('max episode length : {}'.format(self.max_episode_length))

			te_per_t  = 0
			if self.num_transitions_per_iteration is not 0:
				te_per_t = self.total_frames_elapsed / self.num_transitions_per_iteration;
			print_list.append('frame elapsed per transition : {:.2f}'.format(te_per_t))
			if self.adaptive:
				print_list.append('param goal: ' + ' '.join(['%f' % p for p in self.sim_env.GetParamGoal()]))			
			if self.num_nan_per_iteration != 0:
				print_list.append('nan count : {}'.format(self.num_nan_per_iteration))
			print_list.append('===============================================================')

			for s in print_list:
				print(s)
			
			if self.directory is not None:
				out = open(self.directory+"results", "a")
				for s in print_list:
					out.write(s+'\n')
				out.close()
	# y_list에 타겟 추가하는거 고치기 transition_per_episodes로 또 나눠져서 겁나 작아짐
			if self.plot:
				y_list = [[np.asarray(self.transition_per_episodes), 'steps']]
				for i in range(len(self.total_rewards_by_parts)):
					y_list.append([np.asarray(self.total_rewards_by_parts[i]), self.reward_label[i]])

				self.plotFig(y_list, "rewards" , 1, False, path=self.directory+"result.png")

				y_list = y_list[1:]
				for i in range(len(y_list)):
					y_list[i][0] = np.array(y_list[i][0])/np.array(self.transition_per_episodes)

				self.plotFig(y_list, "rewards_per_step", 2, False, path=self.directory+"result_per_step.png")

		self.num_nan_per_iteration = 0
		self.num_episodes_per_iteration = 0
		self.num_transitions_per_iteration = 0
		self.rewards_per_iteration = 0
		self.rewards_by_part_per_iteration = []
		self.total_frames_elapsed = 0
		if self.parametric:
			self.sim_env.SaveParamSpaceLog(self.num_evaluation)
		summary = dict()
		summary['r_per_e'] = r_per_e
		summary['rp_per_i'] = rp_per_i
		summary['t_per_e'] = t_per_e
		return summary
