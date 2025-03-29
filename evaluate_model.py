import os
import pandas as pd
import matplotlib.pyplot as plt

from sumo_rl import SumoEnvironment
from stable_baselines3 import DQN, PPO, A2C

def evaluate_and_plot(
    model_type: str,
    model_path: str,
    net_file: str,
    route_file: str,
    output_folder: str = "evaluation",
    num_seconds: int = 300,
    use_gui: bool = False
):
    """
    Loads a specified RL model ("DQN", "PPO", or "A2C") from model_path,
    runs an evaluation episode in SUMO-RL, and plots results from the CSV logs.
    """
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        single_agent=True,
        out_csv_name=os.path.join(output_folder, "eval_metrics"),
        use_gui=use_gui,
        num_seconds=num_seconds
    )

    model_type = model_type.upper()
    if model_type == "DQN":
        ModelClass = DQN
    elif model_type == "PPO":
        ModelClass = PPO
    elif model_type == "A2C":
        ModelClass = A2C
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    print(f"Loading {model_type} model from {model_path} ...")
    model = ModelClass.load(model_path, env=env)

    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result

    done = False
    total_reward = 0.0
    steps = 0
    info = {}
    
    while not done:
        action, _ = model.predict(obs)
        step_result = env.step(action)
        
        # Handle both old and new Gym API formats for step
        if len(step_result) == 4:  # Old API: obs, reward, done, info
            obs, reward, done, info = step_result
        elif len(step_result) == 5:  # New API: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        total_reward += reward
        steps += 1

    print(f"Evaluation ended. Steps: {steps}, Total reward: {total_reward}")
    env.close()

    csv_path = os.path.join(output_folder, "eval_metrics.csv")
    if not os.path.exists(csv_path):
        print("No CSV file found at:", csv_path)
        return

    df = pd.read_csv(csv_path)
    print("Loaded CSV with columns:", df.columns.tolist())

    os.makedirs(output_folder, exist_ok=True)

    if "queue_length" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["step"], df["queue_length"], label="Queue Length")
        plt.xlabel("Simulation Step")
        plt.ylabel("Queue Length (# vehicles)")
        plt.title("Queue Length Over Time")
        plt.legend()
        plt.savefig(os.path.join(output_folder, "queue_length_plot.png"))
        plt.close()

    if "mean_waiting_time" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["step"], df["mean_waiting_time"], label="Mean Waiting Time")
        plt.xlabel("Simulation Step")
        plt.ylabel("Waiting Time (s)")
        plt.title("Mean Waiting Time Over Time")
        plt.legend()
        plt.savefig(os.path.join(output_folder, "waiting_time_plot.png"))
        plt.close()

    if "reward" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["step"], df["reward"], label="Reward")
        plt.xlabel("Simulation Step")
        plt.ylabel("Reward")
        plt.title("Reward Over Time")
        plt.legend()
        plt.savefig(os.path.join(output_folder, "reward_plot.png"))
        plt.close()

    print(f"Plots saved in {output_folder}. \nFinal total reward: {total_reward}")

def main():
    evaluate_and_plot(
        model_type="DQN",
        model_path="dqn_model.zip",
        net_file="config/3x3grid/3x3Grid2lanes.net.xml",
        route_file="config/3x3grid/routes14000.rou.xml",
        output_folder="evaluation/dqn_eval",
        num_seconds=300,
        use_gui=False
    )

if __name__ == "__main__":
    main()