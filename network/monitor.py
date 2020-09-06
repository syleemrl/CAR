from env import Env
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils import RunningMeanStd
from IPython import embed

class Monitor(object):
	def __init__(self, ref, num_slaves, directory, adaptive, load=False, plot=True, verbose=True):
		self.env = Env(ref, directory, adaptive, num_slaves)
		self.num_slaves = self.env.num_slaves
		self.sim_env = self.env.sim_env
		
		self.num_state = self.env.num_state
		self.num_action = self.env.num_action
		self.RMS = RunningMeanStd(shape=(self.num_state))	
		self.verbose = verbose
		self.plot = plot
		self.directory = directory
		self.adaptive = adaptive

		self.start_time = time.time()		
		self.num_evaluation = 0
		self.num_episodes = 0
		self.num_transitions = 0
		self.total_frames_elapsed = 0
		self.total_rewards = []
		self.total_rewards_target = []
		self.max_episode_length = 0

		self.reward_label = self.sim_env.GetRewardLabels()
		self.total_rewards_by_parts = np.array([[]]*len(self.reward_label))
		self.transition_per_episodes = []
		self.num_nan_per_iteration = 0
		self.num_episodes_per_iteration = 0
		self.num_transitions_per_iteration = 0
		self.rewards_per_iteration = 0
		self.rewards_target_per_iteration = 0
		self.rewards_by_part_per_iteration = []
		self.target_update_count = 0
		self.r_target_avg_old = 0

		self.terminated = [False]*self.num_slaves
		self.states = [0]*self.num_slaves
		self.prevframes = [0]*self.num_slaves

		self.rewards_by_part_per_opt = []
		self.num_transitions_opt = 0
		self.num_episodes_opt = 0

		self.phaselength = self.sim_env.GetPhaseLength()
		self.dof = self.sim_env.GetDOF()
		self.nrewards = 0

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

	def stepForEval(self, action, i):
		s, r, t =  self.env.stepForEval(action, i)
		states_updated = self.RMS.apply(s.reshape(1, -1))
		return states_updated, r, t

	def step(self, actions, record=True):
		self.states, rewards, dones, times, frames, nan_count =  self.env.step(actions)
		curframes = np.array(self.states)[:,-1]
		states_updated = self.RMS.apply(self.states[~np.array(self.terminated)])
		self.states[~np.array(self.terminated)] = states_updated
		if record:
			self.num_nan_per_iteration += nan_count
			for i in range(self.num_slaves):
				if self.adaptive and rewards[i][1] != 0 and rewards[i][1] is not None:
					self.rewards_target_per_iteration += rewards[i][1]
					self.nrewards += 1

				if not self.terminated[i] and rewards[i][0] is not None:
					self.rewards_per_iteration += rewards[i][0]

					self.rewards_by_part_per_iteration.append(rewards[i])
					
					self.num_transitions_per_iteration += 1
					if self.adaptive:
						self.num_transitions_opt += 1
						self.rewards_by_part_per_opt.append(rewards[i][2:])

					if dones[i]:
						self.num_episodes_per_iteration += 1
						if self.adaptive:
							self.num_episodes_opt += 1

						self.total_frames_elapsed += frames[i]

						if frames[i] > self.max_episode_length:
							self.max_episode_length = frames[i]

			if self.adaptive:
				rewards = [[rewards[i][0], rewards[i][1]] for i in range(len(rewards))]
			else:	
				rewards = [rewards[i][0] for i in range(len(rewards))]
				
			self.prevframes = curframes

		return rewards, dones, curframes

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
			print_list.append("Target update : {}".format(self.target_update_count))
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
		self.rewards_target_per_iteration = 0

		summary = dict()
		summary['r_per_e'] = r_per_e
		summary['rp_per_i'] = rp_per_i
		summary['t_per_e'] = t_per_e
		return summary
