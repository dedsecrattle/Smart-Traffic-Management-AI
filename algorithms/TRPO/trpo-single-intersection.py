import os
import sys
from sb3_contrib import TRPO
from sumo_rl import SumoEnvironment
from reward_logger import RewardLogger
import gymnasium as gym


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


out_dir = f"outputs/single-intersection/trpo"
os.makedirs(out_dir, exist_ok=True)
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
                filename=os.path.join(self.log_dir, f"ppo_episode_{self.episode_count+1}.csv")
            )
        
        
        if len(step_result) == 5:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, done, info
    
    def reset(self, **kwargs):
        
        result = self.env.reset(**kwargs)
        return result
    
env = SumoEnvironment(
    net_file="../sumo-config/single-intersection/single-intersection.net.xml",
    route_file="../sumo-config/single-intersection/single-intersection.rou.xml",
    out_csv_name="outputs/single-intersection/trpo",
    single_agent=True,
    use_gui=False,
    num_seconds=100000,
    min_green=10,
    max_green=50,
)

ts_id = env.ts_ids[0]
logger = RewardLogger([ts_id], filename=os.path.join(out_dir, "ppo_rewards.csv"))
env = LoggingEnvWrapper(env, logger, out_dir)

model = TRPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.001,
    n_steps=2048,
    batch_size=128,
    gamma=0.95,
    gae_lambda=0.95,
    cg_max_steps=15,
    cg_damping=0.1,
    line_search_shrinking_factor=0.8,
    line_search_max_iter=10,
    n_critic_updates=10,
    normalize_advantage=True,
    target_kl=0.01,
    device="cpu",
)


model.learn(total_timesteps=100000)


model.save("trpo_single_intersection")
