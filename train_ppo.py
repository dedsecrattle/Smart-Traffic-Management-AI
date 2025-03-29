import os
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def main():
    env = SumoEnvironment(
        net_file=os.path.join("./config/3x3grid/3x3Grid2lanes.net.xml"),
        route_file=os.path.join("./config/3x3grid/routes14000.rou.xml"),
        single_agent=True,
        out_csv_name="outputs/ppo_train",
        use_gui=False,
        num_seconds=300
    )

    check_env(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        verbose=1,
        tensorboard_log="./tensorboard/ppo/"
    )
    model.learn(total_timesteps=20000)

    model.save("ppo_model.zip")
    print("PPO model saved to ppo_model.zip")

    env.close()

if __name__ == "__main__":
    main()
