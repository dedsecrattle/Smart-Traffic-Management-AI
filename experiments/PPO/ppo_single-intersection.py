import os
import sys
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

env = SumoEnvironment(
    net_file="../sumo-config/single-intersection/single-intersection.net.xml",
    route_file="../sumo-config/single-intersection/single-intersection.rou.xml",
    out_csv_name="outputs/single-intersection/ppo",
    single_agent=True,
    use_gui=False,
    num_seconds=100000,
    min_green=10,
    max_green=50,
)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.001,
    n_steps=2048,
    batch_size=128,
    gamma=0.95,
    gae_lambda=0.95,
)

model.learn(total_timesteps=100000)
model.save("ppo_single_intersection")
