import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sumo_rl import SumoEnvironment
from stable_baselines3 import DQN

def evaluate_model(model_path, num_episodes=5, max_steps=300):
    """
    Evaluate a trained DQN model on the SUMO environment
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary containing evaluation metrics
    """

    def my_reward_fn(traffic_signal):
        return traffic_signal.get_average_speed()
    
    env = SumoEnvironment(
        net_file=os.path.join("config/3x3grid", "3x3Grid2lanes.net.xml"),
        route_file=os.path.join("config/3x3grid", "routes14000.rou.xml"),
        single_agent=True,
        out_csv_name="outputs/dqn_evaluation",
        use_gui=False,
        num_seconds=max_steps,
        reward_fn=my_reward_fn
    )

    
    
    model = DQN.load(model_path)
    print(f"Loaded model from {model_path}")
    
    episode_rewards = []
    episode_waiting_times = []
    episode_queue_lengths = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_waiting_times = []
        step_queue_lengths = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            episode_reward += reward
            
            if 'waiting_time' in info:
                step_waiting_times.append(info['waiting_time'])
            if 'queue_length' in info:
                step_queue_lengths.append(info['queue_length'])
        
        episode_rewards.append(episode_reward)
        if step_waiting_times:
            episode_waiting_times.append(np.mean(step_waiting_times))
        if step_queue_lengths:
            episode_queue_lengths.append(np.mean(step_queue_lengths))
        
        print(f"Episode {episode+1} reward: {episode_reward}")
    
    env.close()
    
    metrics = {
        'avg_reward': np.mean(episode_rewards),
        'avg_waiting_time': np.mean(episode_waiting_times) if episode_waiting_times else None,
        'avg_queue_length': np.mean(episode_queue_lengths) if episode_queue_lengths else None,
        'episode_rewards': episode_rewards,
        'episode_waiting_times': episode_waiting_times,
        'episode_queue_lengths': episode_queue_lengths
    }
    
    return metrics

def load_and_process_csv(csv_path):
    """
    Load and process the CSV output from SUMO-RL
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Processed DataFrame
    """
    df = pd.read_csv(csv_path)
    return df

def plot_metrics(eval_metrics, output_dir="plots"):
    """
    Plot key metrics from training and evaluation
    
    Args:
        train_csv: Path to training CSV output
        eval_metrics: Dictionary of evaluation metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(eval_metrics['episode_rewards'])+1), eval_metrics['episode_rewards'])
    plt.axhline(y=eval_metrics['avg_reward'], color='r', linestyle='-', label=f'Average: {eval_metrics["avg_reward"]:.2f}')
    plt.title('Evaluation Rewards by Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'eval_rewards.png'))
    plt.close()
    
    if eval_metrics['avg_waiting_time'] is not None:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(eval_metrics['episode_waiting_times'])+1), eval_metrics['episode_waiting_times'])
        plt.axhline(y=eval_metrics['avg_waiting_time'], color='r', linestyle='-', 
                   label=f'Average: {eval_metrics["avg_waiting_time"]:.2f}')
        plt.title('Average Waiting Times by Episode')
        plt.xlabel('Episode')
        plt.ylabel('Average Waiting Time (s)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'waiting_times.png'))
        plt.close()
    
    if eval_metrics['avg_queue_length'] is not None:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(eval_metrics['episode_queue_lengths'])+1), eval_metrics['episode_queue_lengths'])
        plt.axhline(y=eval_metrics['avg_queue_length'], color='r', linestyle='-', 
                   label=f'Average: {eval_metrics["avg_queue_length"]:.2f}')
        plt.title('Average Queue Lengths by Episode')
        plt.xlabel('Episode')
        plt.ylabel('Average Queue Length')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'queue_lengths.png'))
        plt.close()
    
    print(f"Plots saved to {output_dir}")

def main():
    model_path = "dqn_model.zip"
    
    print("Evaluating model...")
    eval_metrics = evaluate_model(model_path, num_episodes=100)
    
    print("\nEvaluation Summary:")
    print(f"Average Reward: {eval_metrics['avg_reward']:.2f}")
    if eval_metrics['avg_waiting_time'] is not None:
        print(f"Average Waiting Time: {eval_metrics['avg_waiting_time']:.2f} seconds")
    if eval_metrics['avg_queue_length'] is not None:
        print(f"Average Queue Length: {eval_metrics['avg_queue_length']:.2f} vehicles")
    
    print("\nGenerating plots...")
    plot_metrics(eval_metrics)

if __name__ == "__main__":
    main()