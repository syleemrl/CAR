import argparse
import random
import pickle
import datetime
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from monitor import Monitor
from pendulum import PendulumEnv
from IPython import embed
from tensorflow import keras
from utils import RunningMeanStd

activ = tf.nn.relu
kernel_initialize_func = tf.contrib.layers.xavier_initializer()
actor_layer_size = 1024
critic_layer_size = 512
initial_state_layer_size = 512
l2_regularizer_scale = 0.0
regularizer = tf.contrib.layers.l2_regularizer(l2_regularizer_scale)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, next_obs, act, rew, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(state=self.obs1_buf[idxs],
                    next_state=self.obs2_buf[idxs],
                    action=self.acts_buf[idxs],
                    reward=self.rews_buf[idxs],
                    terminal=self.done_buf[idxs])

class SquashBijector(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name="tanh"):
        super(SquashBijector, self).__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            name=name)

    def _forward(self, x):
        return tf.nn.tanh(x)

    def _inverse(self, y):
        return tf.atanh(y)

    def _forward_log_det_jacobian(self, x):
        return 2. * (np.log(2.) - x - tf.nn.softplus(-2. * x))

class Actor(object):
    def __init__(self, scope, state, num_actions, batch_size=256, reuse=False):
        self.scope = scope
        self.mean_action, self.action, self.log_pi = self.buildNetwork(state, num_actions, batch_size, reuse)

    def buildNetwork(self, state, num_actions, batch_size, reuse):
        with tf.variable_scope(self.scope, reuse=reuse):
            L1 = tf.layers.dense(state,actor_layer_size,activation=activ,name='L1',
                kernel_initializer=kernel_initialize_func,
                kernel_regularizer=regularizer
            )

            L2 = tf.layers.dense(L1,actor_layer_size,activation=activ,name='L2',
                kernel_initializer=kernel_initialize_func,
                kernel_regularizer=regularizer
            )

            L3 = tf.layers.dense(L2,actor_layer_size,activation=activ,name='L3',
                kernel_initializer=kernel_initialize_func,
                kernel_regularizer=regularizer
            )

            L4 = tf.layers.dense(L3,actor_layer_size,activation=activ,name='L4',
                kernel_initializer=kernel_initialize_func,
                kernel_regularizer=regularizer
            )

            out = tf.layers.dense(L4,num_actions*2,name='out',
                kernel_initializer=kernel_initialize_func,
                kernel_regularizer=regularizer
            )

            shift, logstd = tf.split(out, num_or_size_splits=2, axis=-1) 
            logstd =  tf.clip_by_value(logstd, -20, 2)
            
            base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(num_actions),             
            scale_diag=tf.ones(num_actions))

            latents = base_distribution.sample(batch_size)
            
            bijector = tfp.bijectors.Affine(shift=shift, scale_diag=tf.exp(logstd))
            actions = bijector.forward(latents)

            squash_bijector = SquashBijector()
            actions = squash_bijector.forward(actions)
            mean_actions = squash_bijector.forward(shift)

            bijector = tfp.bijectors.Chain((squash_bijector, bijector))
            distribution = (
                tfp.distributions.ConditionalTransformedDistribution(
                distribution=base_distribution,
                bijector=bijector)
            )

            logpi = distribution.log_prob(actions)[:, None]

            return mean_actions, actions, logpi

    def getVariable(self, trainable_only=False):
        if trainable_only:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

class Critic(object):
    def __init__(self, state, action, scope, reuse=False):
        self.scope = scope
        self.Q = self.buildNetwork(state, action, reuse)

    def buildNetwork(self, state, action, reuse): 
        with tf.variable_scope(self.scope, reuse=reuse):
            input = tf.concat([state, action], axis=1)

            L1 = tf.layers.dense(input,critic_layer_size,activation=activ,name='L1',
                kernel_initializer=kernel_initialize_func,
                kernel_regularizer=regularizer
            )

            L2 = tf.layers.dense(L1,critic_layer_size,activation=activ,name='L2',
                kernel_initializer=kernel_initialize_func,
                kernel_regularizer=regularizer
            )

            out = tf.layers.dense(L2,1,name='out',
                kernel_initializer=kernel_initialize_func,
                kernel_regularizer=regularizer
            )
            return out

    def getVariable(self, trainable_only=False):
        if trainable_only:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

class SAC(object):
    def __init__(self, lr=3e-4, reward_scale=1.0,
            discount=0.99, tau=5e-3, target_update_interval=1):

        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        tf.set_random_seed(int(time.time()))

        self.policy_lr = lr
        self.Q_lr = lr

        self.reward_scale = reward_scale
        self.discount = discount
        self.tau = tau
        self.target_update_interval = target_update_interval

    def initTrain(self, name, env, pretrain="",
        directory=None, batch_size=256, steps_per_iteration=200):
        self.name = name
        self.directory = directory
        self.steps_per_iteration = steps_per_iteration
        self.batch_size = batch_size
        self.pretrain = pretrain

        self.env = env
        self.num_slaves = self.env.num_slaves
        self.num_action = self.env.num_action
        self.num_state = self.env.num_state

        self.target_entropy = -np.prod(self.num_action)
        self.memory = ReplayBuffer(self.num_state, self.num_action, 1000000)

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = self.num_slaves
        config.inter_op_parallelism_threads = self.num_slaves
        self.sess = tf.Session(config=config)

        #build network and optimizer
        self.buildOptimize()

        self.soft_copy_ops = []
        self.hard_copy_ops = []
        
        for critic, critic_target in zip(self.critics, self.critic_targets):
            source_params = critic.getVariable(True)
            target_params = critic_target.getVariable(True)
            for src, target in zip(source_params, target_params):
                self.soft_copy_ops.append(target.assign((1. - self.tau) * target.value() + self.tau * src.value()))
                self.hard_copy_ops.append(target.assign(src.value()))

        self.sess.run(self.hard_copy_ops)
        writer = tf.summary.FileWriter('logs', self.sess.graph)
        summary_merged = tf.summary.merge_all()  
        save_list = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=save_list,max_to_keep=1)
            
        # load pretrained network
        if self.pretrain is not "":
            self.load(self.pretrain)

        # self.printSetting()

    def initRun(self, pretrain, num_state, num_action, num_slaves=4):
        self.pretrain = pretrain

        self.batch_size = 1
        self.num_slaves = num_slaves
        self.num_action = num_action
        self.num_state = num_state

        self.target_entropy = -np.prod(self.num_action)

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = self.num_slaves
        config.inter_op_parallelism_threads = self.num_slaves
        self.sess = tf.Session(config=config)

        #build network and optimizer
        self.buildOptimize()
        
        save_list = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=save_list,max_to_keep=1)
        
        if self.pretrain is not "":
            self.load(self.pretrain +"/network-0")
            self.RMS = RunningMeanStd(shape=(self.num_state))
            self.RMS.load(self.pretrain+'/rms')

    def buildOptimize(self):
        self.state_ph = tf.placeholder(tf.float32, shape=[None, self.num_state], name='state')
        self.next_state_ph = tf.placeholder(tf.float32, shape=[None, self.num_state], name='next_state')
        self.action_ph = tf.placeholder(tf.float32, shape=[None,self.num_action], name='action')
        self.terminals_ph = tf.placeholder(tf.float32, shape=[None], name='terminals')
        self.rewards_ph = tf.placeholder(tf.float32, shape=[None], name='rewards')

        self.actor = Actor('Actor', self.state_ph, self.num_action, 1)
        self.critics = [Critic(self.state_ph, self.action_ph, 'Critic'+str(i)) for i in range(2)]

        self.critic_targets = [Critic(self.state_ph, self.action_ph, 'CriticTarget'+str(i)) for i in range(2)]

        self.training_ops = []

        self.buildUpdateActor()
        self.buildUpdateCritic()
        self.sess.run(tf.global_variables_initializer())

    def getQtarget(self):
        actor_for_update = Actor('Actor', self.next_state_ph, self.num_action, self.batch_size, True)
        next_action = actor_for_update.action
        next_log_pis = actor_for_update.log_pi
        critic_targets_for_update = [Critic(self.next_state_ph, next_action, 'CriticTarget'+str(i), True) for i in range(2)]
        next_Qs_values = [critic_targets_for_update[i].Q for i in range(2)]

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_values = min_next_Q - self.alpha * next_log_pis

        Q_target = self.reward_scale * self.rewards_ph + self.discount * (1 - self.terminals_ph) * next_values

        return tf.stop_gradient(Q_target)

    def buildUpdateCritic(self):
 
        Q_target = self.getQtarget()
    #    Q_values = [self.critics[i].getQ(self.state_ph, self.action_ph) for i in range(2)]
        Q_values = [self.critics[i].Q for i in range(2)]

        Q_losses = [tf.compat.v1.losses.mean_squared_error(labels=Q_target, predictions=Q_value, weights=0.5) for Q_value in Q_values]

        self.critic_optimizers = [tf.compat.v1.train.AdamOptimizer(learning_rate=self.Q_lr, name='critic_{}_optimizer'.format(i)) for i in range(2)]

        self.critic_train_ops = [Q_optimizer.minimize(loss=Q_loss, var_list=Q.getVariable(True))
            for i, (Q, Q_loss, Q_optimizer) in enumerate(zip(self.critics, Q_losses, self.critic_optimizers))]

        self.training_ops.append(self.critic_train_ops)

    def buildUpdateActor(self):

        action = self.actor.action
        log_pis = self.actor.log_pi

        log_alpha = tf.compat.v1.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
       
        alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(log_pis + self.target_entropy))

        self.alpha_optimizer = tf.compat.v1.train.AdamOptimizer(self.policy_lr, name='alpha_optimizer')
        self.alpha_train_ops = self.alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])
        self.training_ops.append(self.alpha_train_ops)

        self.alpha = tf.exp(log_alpha)

        critics_for_update = [Critic(self.state_ph, action, 'Critic'+str(i), True) for i in range(2)]

        Q_log_targets = [critics_for_update[i].Q for i in range(2)]
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        policy_kl_losses = (self.alpha * log_pis - min_Q_log_target)
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.policy_lr, name="policy_optimizer")
        self.actor_train_ops = self.actor_optimizer.minimize(loss=policy_loss, var_list=self.actor.getVariable(True))
        self.training_ops.append(self.actor_train_ops)

    def update(self):

        batch = self.memory.sample_batch(self.batch_size)
        self.sess.run(self.training_ops, 
            feed_dict={self.state_ph: batch['state'], self.next_state_ph: batch['next_state'], self.action_ph: batch['action'], self.rewards_ph: batch['reward'], self.terminals_ph: batch['terminal']})

        self.sess.run(self.soft_copy_ops)

    def printSetting(self):
        
        print_list = []
        print_list.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print_list.append("test_name : {}".format(self.name))
        print_list.append("motion : {}".format(self.env.motion))
        print_list.append("num_slaves : {}".format(self.num_slaves))
        print_list.append("num state : {}".format(self.num_state))
        print_list.append("num action : {}".format(self.num_action))
        print_list.append("learning_rate : {}".format(self.policy_lr))
        print_list.append("batch_size : {}".format(self.batch_size))
        print_list.append("steps_per_iteration : {}".format(self.steps_per_iteration))
        print_list.append("pretrain : {}".format(self.pretrain))

        for s in print_list:
            print(s)

        if self.directory is not None:
            out = open(self.directory+"parameters", "w")
            for s in print_list:
                out.write(s + "\n")
            out.close()

            out = open(self.directory+"results", "w")
            out.close()

    def train(self, num_iteration):
        for it in range(num_iteration):
            # for i in range(self.num_slaves):
            #     self.env.reset(i)

            self.env.reset()
            states = self.env.getStates()
                
            local_step = 0
            last_print = 0
            
            t = time.time()
            while True:
                # set action
                if it < 5:
                    actions = np.array([np.random.rand(self.num_action) for _ in range(4)])[0]
                else:
                    actions = self.sess.run(self.actor.action, feed_dict={self.state_ph:[states]})[0]
                # rewards, dones = self.env.step(actions)
                # next_states = self.env.getStates()

                # for j in range(self.num_slaves):
                #     if not self.env.getTerminated(j):
                #         if rewards[j] is not None:
                #             self.memory.store(states[j], next_states[j], actions[j], rewards[j], dones[j])
                #             local_step += 1

                #         if dones[j]:                           
                #             if local_step < self.steps_per_iteration:
                #                 self.env.reset(j)
                #             else:
                #                 self.env.setTerminated(j)
                
                next_states, rewards, dones, _ = self.env.step(actions)
                self.memory.store(states, next_states, actions, rewards, dones)
                if dones:
                    self.env.reset()
                local_step += 1

                if local_step >= self.steps_per_iteration:
                    print('iter {} : {}/{}'.format(it+1, local_step, self.steps_per_iteration),end='\r')
                    break
                    # if self.env.getAllTerminated():
                    #     print('iter {} : {}/{}'.format(it+1, local_step, self.steps_per_iteration),end='\r')
                    #     break
                if last_print + 100 < local_step: 
                    print('iter {} : {}/{}'.format(it+1, local_step, self.steps_per_iteration),end='\r')
                    last_print = local_step

                states = self.env.getStates()
            
            if it >= 4:
                for _ in range(self.steps_per_iteration):
                    self.update() 

            print('')
            print(time.time() - t)
            if it % 10 == 9:
                if self.directory is not None:
                    self.save()
                self.env.printSummary()
    
    def save(self):
        self.saver.save(self.sess, self.directory + "network", global_step = 0)

    def load(self, path):
        self.saver.restore(self.sess, path)
    
    def run(self, state):
        state = np.reshape(state, (1, self.num_state))
        state = self.RMS.apply(state)
        return self.sess.run(self.actor.action, feed_dict={self.state_ph:state})

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntimesteps", type=int, default = 10000000)
    parser.add_argument("--motion", type=str, default=None)
    parser.add_argument("--test_name", type=str, default="")
    parser.add_argument("--pretrain", type=str, default="")
    parser.add_argument("--evaluation", type=bool, default=False)
    parser.add_argument("--nslaves", type=int, default=4)
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--no-plot", dest='plot', action='store_false')
    parser.set_defaults(plot=True)
    args = parser.parse_args()

    directory = None
    if args.save:
        if not os.path.exists("./output/"):
            os.mkdir("./output/")

        directory = "./output/" + args.test_name + "/"
        if not os.path.exists(directory):
            os.mkdir(directory)
    
    if args.pretrain != "":
        env = Monitor(motion=args.motion, num_slaves=args.nslaves, load=True, directory=directory, plot=args.plot)
    else:
        env = Monitor(motion=args.motion, num_slaves=args.nslaves, directory=directory, plot=args.plot)
    sac = SAC()
    sac.initTrain(env=PendulumEnv(), name=args.test_name, directory=directory, pretrain=args.pretrain)
    sac.train(args.ntimesteps)
