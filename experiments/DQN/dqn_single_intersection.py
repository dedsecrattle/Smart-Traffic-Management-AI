import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="../sumo-config/single-intersection/single-intersection.net.xml",
        route_file="../sumo-config/single-intersection/single-intersection.rou.xml",
        out_csv_name="outputs/single-intersection/dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=100000,
        min_green=10,
        max_green=50
    )

    model = DQN(
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
    model.learn(total_timesteps=100000)
