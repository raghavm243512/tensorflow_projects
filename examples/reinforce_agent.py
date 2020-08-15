from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import tensorflow as tf
import os

from tf_agents.agents.reinforce import reinforce_agent # tf_agents provides a multitude of RL agents to use
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# display = pyvirtualdisplay.Display(visible=True, size=(1400, 900)).start() # create display

env_name = "CartPole-v0" # selected sample environment
num_iterations = 250 # how many times the agent is trained
collect_episodes_per_iteration = 2 # 
replay_buffer_capacity = 2000 # 

fc_layer_params = (100,) # vararg of hidden layers

learning_rate = 1e-3 
log_interval = 25 # how often progress is logged
num_eval_episodes = 10 # amount of evaluations
eval_interval = 50 # logging of evaluation runs

env = suite_gym.load(env_name) # choose sample environment
env.reset()

print('Observation Spec:')
print(env.time_step_spec().observation) # observation is an array of 4 floats: the position and velocity of the cart, and the angular position and velocity of the pole
print('Action Spec:')
print(env.action_spec()) # scalar where 0 means "move left" and 1 means "move right."

time_step = env.reset()
print('Time step:')
print(time_step)

action = np.array(1, dtype=np.int32) # single value array of 32 bit int

next_time_step = env.step(action) #
print('Next time step:')
print(next_time_step)

train_py_env = suite_gym.load(env_name) # gives environments written in python, interacts as arrays
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env) # allows tensors to interact directly with environment
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

actor_net = actor_distribution_network.ActorDistributionNetwork( # actornetwork predicts actions based on environment, generate a 
                                                                 # distribution which a policy can then sample to generate actions
    train_env.observation_spec(), # Tensor of observations provided by the environment
    train_env.action_spec(), # Tensor of potential actions
    fc_layer_params=fc_layer_params) # number of neurons for each hidden layer

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.compat.v2.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

def collect_episode(environment, policy, num_episodes):

  episode_counter = 0
  environment.reset()

  while episode_counter < num_episodes:
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

    if traj.is_boundary():
      episode_counter += 1
# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few episodes using collect_policy and save to the replay buffer.
  collect_episode(
      train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

  # Use data from the buffer and update the agent's network.
  experience = replay_buffer.gather_all()
  train_loss = tf_agent.train(experience)
  replay_buffer.clear()

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=250)
plt.show()