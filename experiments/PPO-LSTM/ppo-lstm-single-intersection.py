import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment


if __name__ == "__main__":
    net_file = "../sumo-config/single-intersection/single-intersection.net.xml"
    route_file = "../sumo-config/single-intersection/single-intersection.rou.xml"
    out_dir = "outputs/single-intersection/ppo_lstm"
    log_dir = "logs/single-intersection/ppo_lstm"
    model_dir = "models/single-intersection/ppo_lstm"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    temp_env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        single_agent=False,
        use_gui=False,
        num_seconds=1000,
        min_green=10,
        max_green=50,
    )
    
    available_ts = list(temp_env.traffic_signals.keys())
    if not available_ts:
        raise ValueError("No traffic signals found in the environment")
    
    agent_id = available_ts[0] 
    print(f"Using traffic signal ID: {agent_id}")
    
    temp_env.close()
    
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        single_agent=True,
        out_csv_name=os.path.join(out_dir, f"ppo_lstm_{agent_id}"),
        use_gui=False,
        num_seconds=100000,
        min_green=10,
        max_green=50,
    )
    

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
    model.save(os.path.join(model_dir, f"ppo_lstm_{agent_id}_final"))
    env.close()