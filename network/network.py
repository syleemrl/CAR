import tensorflow as tf
import numpy as np

activ = tf.nn.relu
kernel_initialize_func = tf.contrib.layers.xavier_initializer()
actor_layer_size = 1024
critic_layer_size = 512
regression_layer_size = 512
initial_state_layer_size = 512
l2_regularizer_scale = 0.0
regularizer = tf.contrib.layers.l2_regularizer(l2_regularizer_scale)

class Actor(object):
	def __init__(self, sess, scope, state, num_actions):
		self.sess = sess
		self.name = scope
		self.scope = scope + '_Actor'

		self.mean, self.logstd, self.std = self.createNetwork(state, num_actions, False, None)
		self.policy = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
		self.neglogprob = self.neglogp(self.policy)

		self.state = state
	def neglogp(self, x):
		return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) + tf.reduce_sum(self.logstd, axis=-1)

	def createNetwork(self, state, num_actions, reuse, is_training):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
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
			mean = tf.layers.dense(L4,num_actions,name='mean',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)
			self.logstdvar = logstd = tf.get_variable(name='std', 
				shape=[num_actions], initializer=tf.constant_initializer(0)
			)
			sigma = tf.exp(logstd)

			return mean, logstd, sigma

	def getAction(self, states):
		with tf.variable_scope(self.scope):
			action, neglogprob = self.sess.run([self.policy, self.neglogprob], feed_dict={self.state: states})
			return action, neglogprob

	def getMeanAction(self, states):
		with tf.variable_scope(self.scope):
			action = self.sess.run([self.mean], feed_dict={self.state:states})
			return action[0]

	def getVariable(self, trainable_only=False):
		if trainable_only:
			return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
		else:
			return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)


class Critic(object):
	def __init__(self, sess, scope, state, postfix='',critic_layer_size=critic_layer_size):
		self.sess = sess
		self.name = scope
		self.scope = scope + '_Critic' + postfix
		self.value = self.createNetwork(state, False, None, critic_layer_size)

		self.state = state
	def createNetwork(self, state, reuse, is_training, critic_layer_size):	
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			L1 = tf.layers.dense(state,critic_layer_size,activation=activ,name='L1',
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

			return out[:,0]

	def getValue(self, states):
		with tf.variable_scope(self.scope):
			return self.sess.run(self.value, feed_dict={self.state:states})

	def getVariable(self, trainable_only=False):
		if trainable_only:
			return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
		else:
			return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
class RegressionNet(object):
	def __init__(self, sess, scope, input, output, postfix=''):
		self.sess = sess
		self.name = scope
		self.scope = scope + '_Regression' + postfix
		self.value = self.createNetwork(input, output, False)

		self.input = input
	def createNetwork(self, input, output, reuse):	
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			L1 = tf.layers.dense(input, regression_layer_size, activation=activ,name='L1',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			L2 = tf.layers.dense(L1, regression_layer_size, activation=activ,name='L2',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			out = tf.layers.dense(L2, output, name='out',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			return out
	def getValue(self, states):
		with tf.variable_scope(self.scope):
			return self.sess.run(self.value, feed_dict={self.input :states})

	def getVariable(self, trainable_only=False):
		if trainable_only:
			return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
		else:
			return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
