import os
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

def my_custom_reward(env):
    rewards = {}
    for ts in env.ts_ids:
        wait_time = sum(env.traffic_signals[ts].get_waiting_time_per_lane())
        
        queue = sum(env.traffic_signals[ts].get_lanes_queue())
        
        rewards[ts] = -(wait_time * 0.5 + queue * 0.5)
        rewards[ts] += env.traffic_signals[ts].get_last_step_vehicles_passed() * 2
    
    return rewards

if __name__ == "__main__":
    env = SumoEnvironment(
    net_file="../sumo-config/2x2grid/2x2.net.xml",
    route_file="../sumo-config/2x2grid/2x2.rou.xml",
    out_csv_name="outputs/2x2/improved_multi_dqn",
    single_agent=False,
    use_gui=False,
    num_seconds=100000,
    min_green=5, 
    max_green=60,
    delta_time=5, 
    yellow_time=3,
    )
    
    agents = env.ts_ids
    
    models = {}
    for agent_id in agents:
        models[agent_id] = DQN(
            env=env,
            policy="MlpPolicy",
            learning_rate=0.001,
            learning_starts=0,
            train_freq=1,
            buffer_size=50000,
            batch_size=32,
            target_update_interval=500,
            exploration_initial_eps=0.05,
            exploration_final_eps=0.005,
            verbose=1,
        )
    
    episodes = 10
    for episode in range(episodes):
        obs = env.reset()
        done = {"__all__": False}
        
        while not done["__all__"]:
            actions = {}
            for agent_id in agents:
                action, _ = models[agent_id].predict(obs[agent_id], deterministic=False)
                actions[agent_id] = action
            
            next_obs, rewards, done, info = env.step(actions)
            
            for agent_id in agents:
                models[agent_id].replay_buffer.add(
                    np.array([obs[agent_id]]),
                    np.array([actions[agent_id]]),
                    np.array([rewards[agent_id]]),
                    np.array([next_obs[agent_id]]),
                    np.array([done[agent_id]])
                )
                
                models[agent_id].train(gradient_steps=1)
            
            obs = next_obs
            
        print(f"Episode {episode} completed")