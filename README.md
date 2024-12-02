# Platooning Reinforcement Learning Environment

This repository contains the implementation of a reinforcement learning (RL) framework for simulating and training autonomous platooning systems. The project is designed for experimenting with RL algorithms in a custom simulation environment.

## Repository Structure

- **`agent.py`**  
  Contains the implementation of the reinforcement learning agent. This includes the agent's policy, training logic, and interaction with the environment.

- **`PlatooningEnv.py`**  
  Defines the custom platooning environment, which simulates the dynamics of vehicles in a platoon. This environment is compatible with OpenAI's Gym interface.

- **`run.py`**  
  Script to test the trained agent in the platooning environment. This file demonstrates how the agent performs in the simulation.

- **`train_agent.py`**  
  Script to train the RL agent using the platooning environment. It includes the setup for training loops, performance monitoring, and saving the trained model.

## Prerequisites

### Python Dependencies

- Python 3.8+
- Gym (for custom environment integration)
- NumPy
- TensorFlow or PyTorch (based on the implementation in `agent.py`)
- Matplotlib (optional, for visualizing results)

Install the dependencies using:

```bash
pip install -r requirements.txt
