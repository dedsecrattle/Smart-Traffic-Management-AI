import os
import sys
from datetime import datetime

import fire
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl import SumoEnvironment


class SingleAgentWrapper(gym.Env):
    """Wrapper to convert the multi-agent environment to single-agent for PPO."""
    
    def __init__(self, env, agent_id):
        super().__init__()
        self.env = env
        self.agent_id = agent_id
        self.observation_space = env.observation_spaces(agent_id)
        self.action_space = env.action_spaces(agent_id)
        
    def reset(self, **kwargs):
        obs = self.env.reset()
        return np.array(obs[self.agent_id]), {}
        
    def step(self, action):
        actions = {self.agent_id: action}
        next_obs, rewards, dones, infos = self.env.step(actions)
        
        # Check if we need to handle truncations for newer gym versions
        if isinstance(dones, dict) and "__all__" in dones:
            truncated = False
        else:
            # Assume no truncation if the env doesn't provide it
            truncated = False
        
        return (
            np.array(next_obs[self.agent_id]), 
            rewards[self.agent_id], 
            dones[self.agent_id], 
            truncated,
            infos.get(self.agent_id, {}) if isinstance(infos, dict) else {}
        )
        
    def render(self):
        # Implement a dummy render method if needed
        pass
        
    def close(self):
        self.env.close()


def make_env(net_file, route_file, agent_id, out_dir, use_gui=False):
    """Create a wrapped environment for a specific agent."""
    def _init():
        env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            single_agent=False,
            out_csv_name=os.path.join(out_dir, f"ppo_{agent_id}"),
            use_gui=use_gui,
            num_seconds=10000,
            yellow_time=3,
            min_green=5,
            max_green=60,
        )
        return SingleAgentWrapper(env, agent_id)
    return _init


def train_ppo_agent(agent_id, net_file, route_file, out_dir, log_dir, model_dir, total_timesteps=100000):
    """Train a PPO agent for a specific traffic signal."""
    # Create the vectorized environment
    env = DummyVecEnv([make_env(net_file, route_file, agent_id, out_dir)])
    env = VecMonitor(env)
    
    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.95,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        tensorboard_log=log_dir,
        device="cpu"
    )
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    model.learn(total_timesteps=50000, callback=eval_callback)
    
    # Save the final model
    final_model_path = os.path.join(model_dir, f"ppo_{agent_id}_final")
    model.save(final_model_path)
    
    # Close the environment
    env.close()
    
    return final_model_path


def run(use_gui=False, runs=1, total_timesteps=100000):
    """Run PPO training for all traffic signals in the network."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"outputs/double/ppo_{timestamp}"
    log_dir = f"logs/double/ppo_{timestamp}"
    model_dir = f"models/double/ppo_{timestamp}"
    
    # Create directories
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Network and route files
    net_file = "sumo-config/double/network.net.xml"
    route_file = "sumo-config/double/flow.rou.xml"
    
    # Create a temporary environment to get traffic signal IDs
    temp_env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        single_agent=False,
        use_gui=False,
        num_seconds=10,
    )
    ts_ids = temp_env.ts_ids
    temp_env.close()
    
    print(f"Training PPO agents for {len(ts_ids)} traffic signals")
    
    # Train PPO agent for each traffic signal
    for run in range(1, runs + 1):
        print(f"Run {run}/{runs}")
        
        trained_models = {}
        for ts_id in ts_ids:
            print(f"Training agent for traffic signal {ts_id}")
            model_path = train_ppo_agent(
                ts_id, net_file, route_file, out_dir, log_dir, model_dir, total_timesteps
            )
            trained_models[ts_id] = model_path
            
        print(f"Completed run {run}")
        print(f"Trained models saved at: {model_dir}")
    
    print("All training runs completed")


if __name__ == "__main__":
    fire.Fire(run)