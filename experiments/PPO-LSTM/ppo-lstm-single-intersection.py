import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy
from reward_logger import RewardLogger

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment


class LoggingEnvWrapper(gym.Wrapper):
    def __init__(self, env, logger, log_dir):
        super(LoggingEnvWrapper, self).__init__(env)
        self.logger = logger
        self.ts_id = env.ts_ids[0]
        self.last_action = None
        self.episode_count = 0
        self.log_dir = log_dir
        
    def step(self, action):
        
        self.last_action = action
        
        
        step_result = self.env.step(action)
        
        
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
            terminated = truncated = done
        
        
        self.logger.log_step(
            self.env,
            {self.ts_id: action},
            {self.ts_id: reward}
        )
        
        
        if done:
            self.episode_count += 1
            self.logger.save(suffix=f"_episode_{self.episode_count}")
            print(f"Episode {self.episode_count} completed. Logs saved.")
            
            
            self.logger = RewardLogger(
                [self.ts_id], 
                filename=os.path.join(self.log_dir, f"ppo_lstm_episode_{self.episode_count+1}.csv")
            )
        
        
        if len(step_result) == 5:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, done, info
    
    def reset(self, **kwargs):
        
        result = self.env.reset(**kwargs)
        return result


if __name__ == "__main__":
    net_file = "../sumo-config/single-intersection/single-intersection.net.xml"
    route_file = "../sumo-config/single-intersection/single-intersection.rou.xml"
    out_dir = f"outputs/single-intersection/ppo_lstm"
    os.makedirs(out_dir, exist_ok=True)

    
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        single_agent=True,
        out_csv_name=os.path.join(out_dir, f"ppo_lstm_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        use_gui=False,
        num_seconds=100000,
        min_green=10,
        max_green=50,
    )
    
    ts_id = env.ts_ids[0]
    logger = RewardLogger([ts_id], filename=os.path.join(out_dir, "ppo_rewards.csv"))
    env = LoggingEnvWrapper(env, logger, out_dir)

    model = RecurrentPPO(
        policy=RecurrentActorCriticPolicy,
        env=env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.95,
        n_steps=128,
        batch_size=64,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=100000)
    env.close()