"""
COMP 3625 ASG 3

Authors: Alex Pham and Ethan Ai
Prof: Eric Chalmers
Date: Dec/6/2024

cartpole_solution.py

Purpose: using reinforcement learning, with linear function approximation to
balance a pole on a cart in the CartPole-v1 environment to max episode lengths
throughtout the training

Details: reinforment learning agent using linear function approximation to solve the 
cart pole problem. The agent learns how to balance a pole on a cart using Q-values as 
a linear combination of state features and weights. The agent balances between exploration and 
exploitation. Using temporal difference, it updates values estimates incremntally based
on the difference between predicted and actual rewards.

"""
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
N_EPISODES = 500         
N_ACTIONS = 2             
ALPHA = 0.01              # Learning rate
GAMMA = 0.9               # Discount factor
EPSILON = 1.0             # Initial epsilon for ε-greedy policy
EPSILON_DECAY = 0.9       # Decay factor for epsilon
EPSILON_MIN = 0.01        # Minimum epsilon value
CRITICAL_THRESHOLD = 50   # Minimum step threshold for evaluation
RESET_WINDOW = 25         # Number of episodes to monitor for resetting

# Initialize environment
env = gym.make("CartPole-v1", render_mode="none")
n_features = env.observation_space.shape[0]  #number of states
state_scaler = np.array([2.4, 3, 0.2095, 3.5])  #normalizes state values (approx. max range for stable learning)

# Initialize weights for each action
weights = np.zeros((N_ACTIONS, n_features))

# track episode lengths and steps
episode_lengths = []


"""
scales the input state by dividing it by a predefined state_scaler, 
standardizing its values for consistent processing.
This improves numerical stability, and speeds up convergence
"""
def normalize_state(state):
    return state / state_scaler

""" 
calculates the q value for a given state by taking the dot product
of the weight assiocated to an action and a state which is the expected
value of the given state
"""
def q_value(state, action):
    return np.dot(weights[action], state)

"""
an epsilon-greedy policy for action selection

Exploration: With a probability of EPSILON, it chooses a random action 
which makes it explore a new or less familiar actions

Exploitation: it computes the Q-values for all possible actions in the given 
state for a in range(N_ACTIONS)) and selects the action with the highest 
q value (np.argmax(q_values))
"""
def select_action(state):
    if np.random.rand() < EPSILON:
        return np.random.randint(N_ACTIONS)  # Explore
    else:
        q_values = [q_value(state, a) for a in range(N_ACTIONS)]
        return np.argmax(q_values)  # Exploit

""" 
linear function approximation, models q values for a given state and action
as a linear combination of state features 

state: Current state vector
action: Action taken in the current state
reward: Reward received for taking the action
next_state: The resulting state after taking the action
done: Boolean indicating if the episode has ended

temporal difference error is the difference between the predicted q value 
and the target value which measures how much the current q value deviates 
from the expected future reward then the weights are updated from it

"""
def update_weights(state, action, reward, next_state, done):
    global weights
    q_current = q_value(state, action)
    q_next = max([q_value(next_state, a) for a in range(N_ACTIONS)]) if not done else 0
    td_error = reward + GAMMA * q_next - q_current  # Temporal difference error
    weights[action] += ALPHA * td_error * state     # Update weights

""" 
resets the training process by initializing the weights to zero and setting EPSILON to 
1.0 to encourage maximum exploration
"""
def reset_training():
    global weights, EPSILON
    weights = np.zeros((N_ACTIONS, n_features))
    EPSILON = 1.0  # Reset epsilon to encourage exploration

# Training loop
for episode in range(N_EPISODES):
    state, _ = env.reset()
    state = normalize_state(np.array(state))  # Normalize state
    episode_steps = 0

    for step in range(1000):  # max steps per ep
        # Select action using ε-greedy policy
        action = select_action(state)
        
        # Execute the action
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = normalize_state(np.array(next_state))  # Normalize next_state

        # Assign penalty if the episode terminates
        reward = -1 if terminated else 1

        # Update weights using Q-learning
        update_weights(state, action, reward, next_state, terminated)

        # Update state and step counter
        state = next_state
        episode_steps += 1

        if terminated:
            break

    # Track episode lengths
    episode_lengths.append(episode_steps)

    # Check for reset conditions
    if len(episode_lengths) >= RESET_WINDOW:
        last_episodes = episode_lengths[-RESET_WINDOW:]
        below_threshold = all(steps < CRITICAL_THRESHOLD for steps in last_episodes)
        significant_improvement = any(
            last_episodes[i] > 1.5 * last_episodes[i - 1]
            for i in range(1, len(last_episodes))
        )
        if below_threshold and not significant_improvement:
            print(f"Resetting training at episode {episode + 1} due to poor performance.")
            reset_training()
            continue

    # Decay epsilon
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)


    print(f"Episode {episode + 1}: {episode_steps} steps")

env.close()

# Save episode lengths
pd.Series(name='episode_length', data=episode_lengths).to_csv('episode_lengths.csv')

# Plot episode lengths
plt.plot(episode_lengths)
plt.xlabel('Episode')
plt.ylabel('Length (steps)')
plt.title('Episode Lengths Over Time')
plt.show()

