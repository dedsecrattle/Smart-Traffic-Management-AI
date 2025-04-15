import os
import sys
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random

from sumo_rl import SumoEnvironment
from reward_logger import RewardLogger

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)

class PCQLAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.95, tau=0.005, alpha=0.1, conservative_weight=1.0):
        self.q_net = QNetwork(obs_dim, act_dim)
        self.target_q_net = QNetwork(obs_dim, act_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.conservative_weight = conservative_weight
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.act_dim = act_dim

    def select_action(self, state, epsilon=0.05):
        if random.random() < epsilon:
            return random.randint(0, self.act_dim - 1)
        with torch.no_grad():
            q_vals = self.q_net(torch.FloatTensor(state))
            return q_vals.argmax().item()

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_vals = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_vals = self.target_q_net(next_states).max(1, keepdim=True)[0]
            targets = rewards + self.gamma * (1 - dones) * next_q_vals

        bellman_loss = nn.functional.mse_loss(q_vals, targets)
        all_q = self.q_net(states)
        logsumexp = torch.logsumexp(all_q, dim=1, keepdim=True)
        conservative_loss = (logsumexp - q_vals).mean()
        total_loss = bellman_loss + self.conservative_weight * conservative_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"outputs/single-intersection/pcql_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    num_seconds = 100000
    env = SumoEnvironment(
        net_file="../sumo-config/single-intersection/single-intersection.net.xml",
        route_file="../sumo-config/single-intersection/single-intersection.rou.xml",
        out_csv_name=os.path.join(out_dir, "pcql_summary"),
        single_agent=True,
        use_gui=False,
        num_seconds=num_seconds,
        min_green=10,
        max_green=50,
    )

    ts_id = env.ts_ids[0]
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = PCQLAgent(obs_dim, act_dim)

    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        logger = RewardLogger([ts_id], filename=os.path.join(out_dir, f"pcql_rewards_episode_{ep+1}.csv"))
        
        total_reward = 0
        step_count = 0
        done = False
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated or (step_count * env.delta_time >= num_seconds)
            
            logger.log_step(env, {ts_id: action}, {ts_id: reward})
            agent.store_transition((obs, action, reward, next_obs, done))
            agent.update()
            obs = next_obs
            total_reward += reward
            step_count += 1
            
            if done:
                break

        logger.save()
        print(f"Episode {ep+1}, Total reward: {total_reward}, Steps: {step_count}")
    
    env.close()