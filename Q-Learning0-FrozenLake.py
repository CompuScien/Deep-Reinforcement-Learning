


import numpy as np
import pandas as pd
import gym
import random



#-------------create environment for RL-------------#
environment = gym.make("FrozenLake-v0")

#-------------initialization for Qtable-------------#
action_size = environment.action_space.n
state_size = environment.observation_space.n
# print(action_size, state_size)
# Qtable = np.zeros((state_size, action_size)) + 0.00000001
Qtable = np.random.random((state_size, action_size))/1000000

# print(Qtable)


#--------------hyper parameters set----------------#
episodes = 10000
lr = 0.8
max_steps = 100
gamma = 0.95


epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01


#------------implement Q-Learning------------#
# List of rewards
rewards = []

# 2 For life or until learning is stopped
for episode in range(episodes):
    # Reset the environment
    state = environment.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Qtable[state, :])

        # Else doing a random choice --> exploration
        else:
            action = environment.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = environment.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        Qtable[state, action] = Qtable[state, action] + lr * (reward + gamma * np.max(Qtable[new_state, :]) - Qtable[state, action])

        total_rewards += reward

        # Our new state is state
        state = new_state

        # If done (if we're dead) : finish episode
        if done == True:
            break

    episode += 1
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)






print("Score over time: " + str(sum(rewards) / episodes))
print(Qtable)


#--------------Run program after learning Q-Table------------#

environment.reset()

for episode in range(5):
    state = environment.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        environment.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Qtable[state, :])

        new_state, reward, done, info = environment.step(action)

        if done:
            break
        state = new_state
environment.close()













