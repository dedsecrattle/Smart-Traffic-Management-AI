import os
from sumo_rl import SumoEnvironment
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

def main():
    env = SumoEnvironment(
        net_file=os.path.join("config/3x3grid", "3x3Grid2lanes.net.xml"),
        route_file=os.path.join("config/3x3grid", "routes14000.rou.xml"),
        single_agent=True,
        out_csv_name="outputs/dqn_train",
        use_gui=False,
        num_seconds=300
    )
    
    check_env(env)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1,
        tensorboard_log="./tensorboard/dqn/"
    )

    model.learn(total_timesteps=100000)
    model.save("dqn_model.zip")
    print("DQN model saved to dqn_model.zip")

    env.close()

if __name__ == "__main__":
    main()
