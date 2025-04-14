import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

import sumo_rl


def make_env():
    """Create a SUMO environment wrapped for Stable Baselines3."""
    # Create a SUMO environment
    env = sumo_rl.grid4x4(use_gui=False, out_csv_name="outputs/grid4x4/ppo_test")
    
    # Store the original environment to access parameters later
    orig_env = env
    
    # Reset to get initial observations and agent list
    env.reset()
    agents = list(env.agents)
    
    # Get observation and action spaces from the first agent
    first_agent = agents[0]
    obs_space = env.observation_space(first_agent)
    act_space = env.action_space(first_agent)
    
    # Create a wrapper that converts multi-agent to single-agent for SB3
    class SB3Wrapper(gym.Env):
        def __init__(self):
            self.env = orig_env
            self.observation_space = obs_space
            self.action_space = act_space
            self.agents = agents
        
        def reset(self, **kwargs):
            obs = self.env.reset(**kwargs)
            # Return only the first agent's observation
            return np.array(list(obs.values())[0]), {}
        
        def step(self, action):
            # Apply the same action to all agents
            actions = {agent: action for agent in self.env.agents}
            obs, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Combine terminations and truncations for SB3 compatibility
            dones = {agent: terminations[agent] or truncations[agent] for agent in self.env.agents}
            
            # Use values from the first agent
            first_agent = list(self.env.agents)[0]
            
            return (
                np.array(obs[first_agent]), 
                rewards[first_agent], 
                dones[first_agent], 
                False,  # truncated
                infos[first_agent]
            )
        
        def close(self):
            self.env.close()
    
    return SB3Wrapper()


if __name__ == "__main__":
    # Make sure output directories exist
    os.makedirs("outputs/grid4x4", exist_ok=True)
    os.makedirs("logs/grid4x4/ppo_test", exist_ok=True)
    os.makedirs("models/grid4x4/ppo_test", exist_ok=True)
    
    # Create a vectorized environment
    vec_env = DummyVecEnv([make_env])
    
    print("Environment created")

    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
        tensorboard_log="./logs/grid4x4/ppo_test",
    )

    print("Starting training")
    # Set up evaluation callback to save best model
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path="./models/grid4x4/ppo_test/",
        log_path="./logs/grid4x4/ppo_test/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    model.learn(total_timesteps=50000, callback=eval_callback)
    
    # Save final model
    model.save("models/grid4x4/ppo_final")

    print("Training finished. Starting evaluation")
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=5)

    print(f"Mean reward: {mean_reward}")
    print(f"Std reward: {std_reward}")

    print("All done, cleaning up")
    vec_env.close()