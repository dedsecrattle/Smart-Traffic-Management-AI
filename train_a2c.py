import os
from sumo_rl import SumoEnvironment
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

def main():
    env = SumoEnvironment(
        net_file=os.path.join("config/3x3grid", "3x3Grid2lanes.net.xml"),
        route_file=os.path.join("config/3x3grid", "routes14000.rou.xml"),
        single_agent=True,
        out_csv_name="outputs/a2c_train",
        use_gui=False,
        num_seconds=2000
    )
    
    check_env(env)

    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        verbose=1,
        tensorboard_log="./tensorboard/a2c/"
    )
    model.learn(total_timesteps=20000)

    model.save("a2c_model.zip")
    print("A2C model saved to a2c_model.zip")

    env.close()

if __name__ == "__main__":
    main()
