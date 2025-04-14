import os
import sys
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from datetime import datetime

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment


class RLlibSumoEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.env = SumoEnvironment(
            net_file=env_config["net_file"],
            route_file=env_config["route_file"],
            use_gui=env_config.get("use_gui", False),
            out_csv_name=env_config["out_csv_name"],
            single_agent=False,
            num_seconds=env_config.get("num_seconds", 100_000),
            min_green=env_config.get("min_green", 10),
            max_green=env_config.get("max_green", 50),
        )
        self.ts_ids = self.env.ts_ids

    def reset(self):
        obs = self.env.reset()
        return {ts: obs[ts] for ts in self.ts_ids}

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        dones["__all__"] = all(dones.values())
        return obs, rewards, dones, infos


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # Timestamped output
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = f"outputs/2x2grid/rllib_shared_policy_{timestamp}.csv"

    # PPO config with shared policy
    config = (
        PPOConfig()
        .environment(env=RLlibSumoEnv, env_config={
            "net_file": "../sumo-config/2x2grid/2x2.net.xml",
            "route_file": "../sumo-config/2x2grid/2x2.rou.xml",
            "out_csv_name": out_csv,
            "use_gui": False,
            "num_seconds": 100_000,
            "min_green": 10,
            "max_green": 50,
        })
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id: "shared_policy"
        )
        .training(
            gamma=0.95,
            lr=0.0003,
            lambda_=0.95,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
        )
    )

    tune.run(
        "PPO",
        name=f"ppo_shared_{timestamp}",
        stop={"timesteps_total": 100_000},
        config=config.to_dict(),
        local_dir="results/rllib",  # Where models/logs are saved
        checkpoint_at_end=True
    )
