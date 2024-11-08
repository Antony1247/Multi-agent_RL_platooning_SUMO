from PlatooningEnv import PlatooningParallelEnv

env = PlatooningParallelEnv()
observations = env.reset()

done = False
while not done:
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}  # Random actions
    observations, rewards, dones, infos = env.step(actions)
    done = all(dones.values())

env.close()
print("Test episode completed successfully.")
