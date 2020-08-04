import gym # sample environments
import numpy as np
import time
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
STATES = env.observation_space.n # number of states
ACTIONS = env.action_space.n # number of actions

Q = np.zeros((STATES, ACTIONS)) # table of all possible states & actions

EPISODES = 1500 # how many times to run the enviornment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of enviornment
LEARNING_RATE = 0.81  # learning rate
GAMMA = 0.96 # higher value gives more weight to future rewards

RENDER = False # if you want to see training set to true
epsilon = 0.9 # chance of a random action

rewards = [] # list of rewards for display
for episode in range(EPISODES):

  state = env.reset() # reset environment
  for _ in range(MAX_STEPS): # run a total of MAX_STEPS times
    
    if RENDER:
      env.render()

    if np.random.uniform(0, 1) < epsilon:
      action = env.action_space.sample()  # take potentially random action. Actions would be functions written by us
    else:
      action = np.argmax(Q[state, :]) # take action from Q table

    next_state, reward, done, _ = env.step(action) # perform action and get relevant information

    Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action]) # update Q table

    state = next_state # update state 

    if done: 
      rewards.append(reward) # add for display
      epsilon -= 0.001 # reduce randomness
      break  # reached goal

print(Q)
print(f"Average reward: {sum(rewards)/len(rewards)}:")


# Display average reward over 100 steps
def get_average(values):
  return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
  avg_rewards.append(get_average(rewards[i:i+100])) 

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()