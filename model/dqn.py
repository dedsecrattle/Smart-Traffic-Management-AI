import os
import sys
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# Ensure SUMO environment is available
if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare the environment variable 'SUMO_HOME' (SUMO installation path)")
else:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
import traci

from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent on a 2-way single-intersection SUMO scenario")
    parser.add_argument("--use-gui", action="store_true", help="Whether to run SUMO with GUI (visualize simulation)")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total training timesteps for DQN agent")
    parser.add_argument("--output-dir", type=str, default="outputs/dqn", help="Directory to save the model and plots")
    parser.add_argument("--eval-episode-length", type=int, default=1000, help="Simulation time (seconds) for each evaluation episode")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    env = SumoEnvironment(
        net_file="sumo-config/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo-config/2way-single-intersection/single-intersection-gen.rou.xml",
        use_gui=args.use_gui,
        single_agent=True,
        num_seconds=5400,
        yellow_time=4,
        min_green=5,
        max_green=60,
    )
    env = Monitor(env)

    model = DQN(env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1)
    model.learn(total_timesteps=args.total_timesteps)
    model.save(os.path.join(args.output_dir, "dqn_model"))
    print("DQN model saved to:", os.path.join(args.output_dir, "dqn_model.zip"))

    episode_rewards = env.get_episode_rewards() 
    env.close()

    eval_env = SumoEnvironment(
        net_file="sumo-config/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo-config/2way-single-intersection/single-intersection-gen.rou.xml",
        use_gui=args.use_gui,
        single_agent=True,
        num_seconds=args.eval_episode_length
    )

    eval_rewards = [] 
    eval_waiting_times = [] 
    eval_queue_lengths = []

    for ep in range(args.eval_episodes):
        obs, info = eval_env.reset() 
        terminated, truncated = False, False
        episode_reward = 0.0
        total_wait_time = 0.0
        steps = 0
        total_stopped = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            if "system_total_stopped" in info:
                total_stopped += info["system_total_stopped"]
            for veh_id in eval_env.sumo.simulation.getArrivedIDList():
                total_wait_time += eval_env.sumo.vehicle.getAccumulatedWaitingTime(veh_id)
            steps += 1

        avg_wait = (total_wait_time / eval_env.num_arrived_vehicles) if eval_env.num_arrived_vehicles > 0 else 0.0
        avg_queue = total_stopped / steps if steps > 0 else 0.0
        eval_rewards.append(episode_reward)
        eval_waiting_times.append(avg_wait)
        eval_queue_lengths.append(avg_queue)
        print(f"Evaluation Episode {ep+1}: Reward = {episode_reward:.2f}, Avg Waiting Time = {avg_wait:.2f}s, Avg Queue Length = {avg_queue:.2f} vehicles")

    eval_env.close()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o')
    plt.title('DQN Training – Cumulative Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "dqn_training_reward.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(eval_waiting_times) + 1), eval_waiting_times, marker='o', color='orange')
    plt.title('DQN Evaluation – Average Waiting Time per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Avg Waiting Time per Vehicle (s)')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "dqn_eval_waiting_time.png"))
    plt.close()
