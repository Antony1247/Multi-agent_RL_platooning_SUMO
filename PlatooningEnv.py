from pettingzoo import ParallelEnv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
from sumolib import checkBinary
import matplotlib.pyplot as plt

TOTAL_STEPS = 3000

class PlatooningParallelEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], 'name': "platooning_v2"}

    def __init__(self):
        super().__init__()

        self.num_platoons = 2  # Number of platoons
        self.agents = [f"platoon_{i}_follower" for i in range(self.num_platoons)]
        self.possible_agents = self.agents.copy()

        self.action_spaces = {agent: spaces.Discrete(3) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Box(low=np.array([0, 0]), high=np.array([50, 200]), dtype=np.float32) for agent in self.agents
        }

        self.STEPS = 0
        self.sumo_binary = checkBinary('sumo-gui') # Adjust this path to your SUMO binary
        traci.start([self.sumo_binary, "-c", "./maps/singlelane/singlelane.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        for _ in range(20):
            traci.simulationStep()
            self.STEPS += 1

        self.vehicles = self.initialize_vehicles()

    def initialize_vehicles(self):

        vehicles = {}
        vehicles1 = traci.vehicle.getIDList()
        if not vehicles1:
            raise ValueError("No vehicles found in the simulation.")

        # Sort vehicles based on their lane position (x coordinate)
        vehicles_sorted_by_x = ['car3.0', 'car2.0', 'car1.0']
        if len(vehicles_sorted_by_x) < 2:
            raise ValueError("Not enough vehicles in the simulation to form a platoon.")

        for i in range(self.num_platoons):
            leader_index = i
            follower_index = i+1
            # follower2_index = 2
            vehicle0 = vehicles_sorted_by_x[leader_index]
            vehicle1  = vehicles_sorted_by_x[follower_index]
            # vehicle2  = vehicles_sorted_by_x[follower2_index]
            vehicles[f"platoon_{i}_follower"] = {"leader": vehicle0 , "follower": vehicle1}

        return vehicles
    
    def calculate_distance(self, leader_id, follower_id):
        leader_pos = traci.vehicle.getPosition(leader_id)
        follower_pos = traci.vehicle.getPosition(follower_id)

        distance = ((leader_pos[0] - follower_pos[0])**2 + (leader_pos[1] - follower_pos[1])**2)**0.5
        return distance

    def step(self, actions):
        self.STEPS += 1
        rewards = {}
        observations = {}
        dones = {}
        infos = {agent: {} for agent in self.agents}

        for agent, action in actions.items():
            self.adjust_leader_speed(agent)
            self.apply_action(agent, action)

        traci.simulationStep()

        for agent in self.agents:
            observations[agent] = self.observe(agent)
            rewards[agent] = self.calculate_reward(agent)
            dones[agent] = self.STEPS >= TOTAL_STEPS

        if self.STEPS >= TOTAL_STEPS:
            traci.close()

        return observations, rewards, dones, infos
    

    def adjust_leader_speed(self, agent):
        leader = self.vehicles[agent]["leader"]
        # Check if the leader is specifically "car3.0"
        if leader == "car3.0":
            desired_speed = 3  # Set desired speed to 5 for "car3.0"
            traci.vehicle.setSpeed(leader, desired_speed)
            

    def apply_action(self, agent, action):
        follower_id = self.vehicles[agent]["follower"]
        leader = self.vehicles[agent]["leader"]
        leader_lane_index = traci.vehicle.getLaneIndex(leader)
        traci.vehicle.changeLane(follower_id,leader_lane_index, 0)
        traci.vehicle.setLaneChangeMode(follower_id, 512)
        leader_speed = traci.vehicle.getSpeed(leader)
        current_speed = traci.vehicle.getSpeed(follower_id)
        if action == 0:  # Accelerate
            traci.vehicle.setSpeed(follower_id, leader_speed + 0.5)
        elif action == 1:  # Decelerate
            traci.vehicle.setSpeed(follower_id, leader_speed - 0.5)
        else:
            traci.vehicle.setSpeed(follower_id, leader_speed)


    def observe(self, agent):
        follower_id = self.vehicles[agent]["follower"]
        leader_id = self.vehicles[agent]["leader"]
        follower_speed = traci.vehicle.getSpeed(follower_id)
        current_headway = self.calculate_distance(leader_id , follower_id )
        return np.array([follower_speed, current_headway], dtype=np.float32)

    def calculate_reward(self, agent):
        reward = 0
        follower_id = self.vehicles[agent]["follower"]
        leader_id = self.vehicles[agent]["leader"]
        current_headway = self.calculate_distance(leader_id, follower_id)
        optimal_headway = 40  # Define the optimal headway distance in meters

        # Calculate the difference from the optimal headway
        distance_error = abs(current_headway - optimal_headway)

        # Implement a quadratic penalty for deviation from the optimal distance
        reward -= (distance_error ** 2) * 0.1  # Adjust the scaling factor to fine-tune sensitivity

        return reward

    def reset(self):
        traci.close()
        traci.start([self.sumo_binary, "-c", "./maps/singlelane/singlelane.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        for _ in range(20):
            traci.simulationStep()
            self.STEPS += 1
        self.vehicles = self.initialize_vehicles()
        self.STEPS = 0
        return {agent: self.observe(agent) for agent in self.agents}

    def render(self, mode='human'):
        pass

    def close(self):
        traci.close()

        # Example function to plot at the end
    def plot_metrics(metrics):
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['time_steps'], metrics['average_speed'], label='Average Speed')
        plt.xlabel('Time Step')
        plt.ylabel('Speed')
        plt.title('Average Speed Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
