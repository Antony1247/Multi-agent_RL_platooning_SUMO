import gymnasium as gym
from stable_baselines3 import PPO
import traci
from PlatooningEnv import PlatooningEnv  # Ensure this is correctly imported from its file or location

def run_model_in_sumo():
    # Load the trained model
    model = PPO.load("multi_agent")
    
    # Create an instance of your SUMO environment
    env = PlatooningEnv()
    
    obs = env.reset()
    done = False
    while not done:
        # Predict the action to take based on the current observation
        action, _states = model.predict(obs, deterministic=True)
        
        # Apply the action in the environment to get the new state, reward, etc.
        obs, reward, done, truncated, info = env.step(action)
        
        # Optionally, you can render the environment if it supports rendering
        env.render()

    # Close the environment and SUMO instance properly
    env.close()
    print("Simulation finished!")

# Run the simulation with the trained model
run_model_in_sumo()
