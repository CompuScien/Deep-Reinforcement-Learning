
import numpy as np
import pandas as pd
import gym
import random
import matplotlib.pyplot as plt
import sys


env = gym.make("MountainCar-v0")


lr = 0.1
gamma = 0.95
episodes = 20000

SHOW_EVERY = 100


# DISCRETE_OS_SIZE = [20, 20]
DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = episodes//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
ep_rewards = []
agg_ep_reward = {'ep': [], 'avg': [], 'min': [], 'max': []}



def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(episodes):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    # print(discrete_state)
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        action = np.argmax(q_table[discrete_state])  # always go right!
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        # If simulation did not end yet after last step - update Q table
        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])
            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]
            # And here's our equation for a new Q value for current state and action
            new_q = (1 - lr) * current_q + lr * (reward + gamma * max_future_q)
            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"we made it on episode{episode}")
            # q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0
        discrete_state = new_discrete_state
    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
         epsilon -= epsilon_decay_value
    ep_rewards.append(episode_reward)
    if episode % 10 == 0:
        np.save(f"qtables/{episode}-qtable.npy", q_table)
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/SHOW_EVERY
        agg_ep_reward['ep'].append(episode)
        agg_ep_reward['avg'].append(average_reward)
        agg_ep_reward['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        agg_ep_reward['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')


env.close()



# plot the rewards
plt.figure()
plt.plot(agg_ep_reward['ep'], agg_ep_reward['avg'], label="average rewards")
plt.plot(agg_ep_reward['ep'], agg_ep_reward['max'], label="max rewards")
plt.plot(agg_ep_reward['ep'], agg_ep_reward['min'], label="min rewards")
plt.legend(loc=4)
plt.show()



# Thanks to https://pythonprogramming.net/ (Sentdex)


