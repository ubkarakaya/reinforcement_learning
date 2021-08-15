"""
Deep Learning and Neural Networks Course
the 3rd Assignment
It is encoded by Ufuk Baran Karakaya
"""

import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import math
from torch import nn
from tqdm import tqdm

from Utilities import *

# Define the replay memory
replay_mem = ReplayMemory(capacity=3)

# Push some samples
replay_mem.push(1, 1, 1, 1)
replay_mem.push(2, 2, 2, 2)
replay_mem.push(3, 3, 3, 3)
replay_mem.push(4, 4, 4, 4)
replay_mem.push(5, 5, 5, 5)

# Check the content of the memory
print('\nCONTENT OF THE MEMORY')
print(replay_mem.memory)

# Random sample
print('\nRANDOM SAMPLING')
for i in range(5):
    print(replay_mem.sample(2))  # Select 2 samples randomly from the memory


class DQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(state_space_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_space_dim)
        )

    def forward(self, x):
        return self.linear(x)


net = DQN(state_space_dim=4, action_space_dim=2)

# PARAMETERS
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
# gamma parameter for the long term reward
gamma = 0.9
replay_memory_capacity = 10000
lr = 1e-2
target_net_update_steps = 10
batch_size = 128
bad_state_penalty = 0
min_samples_for_training = 1000


def choose_action_epsilon_greedy(net, state, epsilon):
    global steps_done
    steps_done += 1
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')
    # To reduce the training time, a threshold value was added to the process
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)

    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32)  # Convert the state to tensor
        net_out = net(state)

    # Get the best action (argmax of the network output)
    best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if random.random() > eps_threshold:
        # List of non-optimal actions
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        # Select randomly
        action = random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action

    return action, net_out.numpy()


# Test if it works as expected
steps_done = 0
state = (0, 0, 0, 0)
epsilon = 0.05
chosen_action, q_values = choose_action_epsilon_greedy(net, state, epsilon)


def choose_action_softmax(net, state, temperature):
    if temperature < 0:
        raise Exception('The temperature value must be greater than or equal to 0 ')

    # If the temperature is 0, just select the best action using the eps-greedy policy with epsilon = 0
    if temperature == 0:
        return choose_action_epsilon_greedy(net, state, 0)

    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32)
        net_out = net(state)

    # Apply softmax with temp
    temperature = max(temperature, 1e-3)  # set a minimum to the temperature for numerical stability
    softmax_out = nn.functional.softmax(net_out / temperature, dim=0).numpy()

    # Sample the action using softmax output as mass pdf
    all_possible_actions = np.arange(0, softmax_out.shape[-1])
    action = np.random.choice(all_possible_actions,
                              p=softmax_out)

    return action, net_out.numpy()


state = (0, 0, 0, 0)
temperature = 1
chosen_action, q_values = choose_action_softmax(net, state, temperature)

print(f"ACTION: {chosen_action}")
print(f"Q-VALUES: {q_values}")

# Define exploration profile
initial_value = 5
num_iterations = 200
exp_decay = np.exp(-np.log(
    initial_value) / num_iterations * 6)
exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]

# Plot exploration profile
plt.figure(figsize=(12, 8))
plt.plot(exploration_profile)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Exploration profile (Softmax temperature)')

# Create environment
env = gym.make('CartPole-v1')  # Initialize the Gym environment
env.seed(0)  # Set a random seed for the environment (reproducible results)

# Get the shapes of the state space (observation_space) and action space (action_space)
state_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.n

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# Initialize the replay memory
replay_mem = ReplayMemory(replay_memory_capacity)

# Initialize the policy network
policy_net = DQN(state_space_dim, action_space_dim)

# Initialize the target network with the same weights of the policy network
target_net = DQN(state_space_dim, action_space_dim)
target_net.load_state_dict(
    policy_net.state_dict())

# Initialize the optimizer
optimizer = torch.optim.SGD(policy_net.parameters(),
                            lr=lr)  # The optimizer will update ONLY the parameters of the policy network

# Initialize the loss function (Huber loss)
loss_fn = nn.SmoothL1Loss()


def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
    # Sample the data from the replay memory
    batch = replay_mem.sample(batch_size)
    batch_size = len(batch)

    # Create tensors for each element of the batch
    states = torch.tensor([s[0] for s in batch], dtype=torch.float32)
    actions = torch.tensor([s[1] for s in batch], dtype=torch.int64)
    rewards = torch.tensor([s[3] for s in batch], dtype=torch.float32)

    # Compute a mask of non-final states (all the elements where the next state is not None)
    non_final_next_states = torch.tensor([s[2] for s in batch if s[2] is not None],
                                         dtype=torch.float32)  # the next state can be None if the game has ended
    non_final_mask = torch.tensor([s[2] is not None for s in batch], dtype=torch.bool)

    # Compute all the Q values (forward pass)
    policy_net.train()
    q_values = policy_net(states)
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1))

    # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
        target_net.eval()
        q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = torch.zeros(batch_size)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0]

    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1)  # Set the required tensor shape

    # Compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping (clip all the gradients greater than 2 for training stability)
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()


'''
TRAINING PROCESS
'''
# Initialize the Gym environment
env = gym.make('CartPole-v1')
env.seed(0)

for episode_num, tau in enumerate(tqdm(exploration_profile)):

    # Reset the environment and get the initial state
    state = env.reset()
    # Reset the score. The final score will be the total amount of steps before the pole falls
    score = 0
    done = False

    # Go on until the pole falls off
    while not done:

        # Choose the action following the policy
        action, q_values = choose_action_softmax(policy_net, state, temperature=tau)

        # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
        next_state, reward, done, info = env.step(action)

        # We apply a (linear) penalty when the cart is far from center
        pos_weight = 1
        reward = reward - pos_weight * np.abs(state[0])

        # Update the final score (+1 for each step)
        score += 1

        # Apply penalty for bad state
        if done:  # if the pole has fallen down
            reward += bad_state_penalty
            next_state = None

        # Update the replay memory
        replay_mem.push(state, action, next_state, reward)

        # Update the network
        if len(
                replay_mem) > min_samples_for_training:
            update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size)

        # Visually render the environment (disable to speed up the training)
        env.render()

        # Set the current state for the next iteration
        state = next_state

    # Update the target network every target_net_update_steps episodes
    if episode_num % target_net_update_steps == 0:
        print('Updating target network...')
        target_net.load_state_dict(
            policy_net.state_dict())  # This will copy the weights of the policy network to the target network

    # Print the final score
    print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}")  # Print the final score

env.close()
