import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from sumo_rl import SumoEnvironment

# Ensure SUMO is set
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


class SingleAgentWrapper(gym.Env):
    def __init__(self, base_env, agent_id):
        super().__init__()
        self.base_env = base_env
        self.agent_id = agent_id
        self.observation_space = base_env.observation_spaces(agent_id)
        self.action_space = base_env.action_spaces(agent_id)

    def reset(self, *, seed=None, options=None):
        obs = self.base_env.reset()
        return obs[self.agent_id], {}

    def step(self, action):
        action_dict = {ts: 0 for ts in self.base_env.ts_ids}
        action_dict[self.agent_id] = action

        result = self.base_env.step(action_dict)
        if len(result) == 5:
            next_obs, rewards, terminateds, truncateds, infos = result
        else:
            next_obs, rewards, dones, infos = result
            terminateds = dones
            truncateds = {ts: False for ts in self.base_env.ts_ids}

        return (
            next_obs[self.agent_id],
            rewards[self.agent_id],
            terminateds[self.agent_id],
            truncateds[self.agent_id],
            infos.get(self.agent_id, {})
        )

    def render(self):
        pass

    def close(self):
        self.base_env.close()


def train_dqn(agent_id, base_env, total_timesteps=100_000):
    wrapped_env = DummyVecEnv([lambda: SingleAgentWrapper(base_env, agent_id)])
    model = DQN(
        policy="MlpPolicy",
        env=wrapped_env,
        learning_rate=0.001,
        buffer_size=50000,
        learning_starts=1000,
        train_freq=1,
        batch_size=32,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.005,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(f"models/dqn_agent_{agent_id}")
    wrapped_env.close()


if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="../sumo-config/2x2grid/2x2.net.xml",
        route_file="../sumo-config/2x2grid/2x2.rou.xml",
        out_csv_name="outputs/2x2/dqn_multiagent",
        single_agent=False,
        use_gui=False,
        num_seconds=100000,
        min_green=5,
        max_green=60,
        delta_time=5,
        yellow_time=3,
    )

    for ts_id in env.ts_ids:
        print(f"Training DQN for agent {ts_id}")
        train_dqn(ts_id, env, total_timesteps=50000)

    print("All DQN agents trained!")
