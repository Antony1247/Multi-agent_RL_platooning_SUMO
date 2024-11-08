import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PlatooningEnv import PlatooningParallelEnv
from agent import QNetwork
import matplotlib.pyplot as plt

def train_dqn(env, agents, episodes, start_epsilon=1.0, end_epsilon=0.01, decay_rate=0.995):
    optimizers = {agent: optim.Adam(agents[agent].parameters(), lr=0.01) for agent in env.agents}
    lr_scheduler = {agent: optim.lr_scheduler.StepLR(optimizers[agent], step_size=100, gamma=0.9) for agent in env.agents}
    epsilon = start_epsilon

    rewards = []
    epsilons = []
    episodes_list = []
    agent_rewards = {agent: [] for agent in env.agents}
    agent_headways = {agent: [] for agent in env.agents}  # Dictionary to store headways for each agent

    for episode in range(episodes):
        state = env.reset()
        done = {agent: False for agent in env.agents}
        episode_rewards = {agent: 0 for agent in env.agents}

        while not all(done.values()):
            actions = {}
            for agent in env.agents:
                if np.random.rand() < epsilon:
                    action = env.action_spaces[agent].sample()
                else:
                    state_tensor = torch.from_numpy(state[agent]).float()
                    with torch.no_grad():
                        action = torch.argmax(agents[agent](state_tensor)).item()
                actions[agent] = action

            next_state, reward, done, _ = env.step(actions)

            for agent in reward:
                episode_rewards[agent] += reward[agent]
                # Collect headway data for each agent
                agent_headways[agent].append(next_state[agent][1])  # Update the index if necessary

            # Update the optimizer
            for agent in env.agents:
                optimizer = optimizers[agent]
                optimizer.zero_grad()

                state_tensor = torch.from_numpy(state[agent]).float()
                action_values = agents[agent](state_tensor)
                selected_action_value = action_values.gather(0, torch.tensor([actions[agent]], dtype=torch.long))

                target = torch.tensor([reward[agent]], dtype=torch.float)
                loss = F.mse_loss(selected_action_value, target)
                loss.backward()
                optimizer.step()

            state = next_state

        for scheduler in lr_scheduler.values():
            scheduler.step()

        total_reward = sum(episode_rewards.values())
        rewards.append(total_reward)
        epsilons.append(epsilon)
        episodes_list.append(episode)
        for agent in env.agents:
            agent_rewards[agent].append(episode_rewards[agent])
        epsilon = max(end_epsilon, epsilon * decay_rate)

    plot_training_results(episodes_list, agent_rewards)
    plot_headway(agent_headways)

def plot_headway(agent_headways):
    for agent, headways in agent_headways.items():
        plt.figure(figsize=(8, 4))
        plt.plot(headways, label=f'Agent {agent} Headway')
        plt.title(f"Headway over Time Steps for Agent {agent}")
        plt.xlabel("Time Step")
        plt.ylabel("Headway")
        plt.legend()
        plt.show()


def plot_training_results(episodes, agent_rewards):
    for agent, rewards in agent_rewards.items():
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, rewards, label=f'Rewards for {agent}')
        plt.title(f'Training Rewards for {agent}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    env = PlatooningParallelEnv()
    agents = {agent: QNetwork(env.observation_spaces[agent].shape[0], env.action_spaces[agent].n) for agent in env.agents}
    train_dqn(env, agents, episodes=300)  # Increased the number of episodes for better training
