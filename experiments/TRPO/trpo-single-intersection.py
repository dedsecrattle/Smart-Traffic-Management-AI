import os
import sys
from sb3_contrib import TRPO
from sumo_rl import SumoEnvironment


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


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
)


model.learn(total_timesteps=100000)


model.save("trpo_single_intersection")
