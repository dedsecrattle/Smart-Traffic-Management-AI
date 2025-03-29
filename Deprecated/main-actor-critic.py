import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time
import argparse
import traci
import sumolib
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# --------------------------------------------------------------------------------
# CONFIGURATION DICTIONARY
# --------------------------------------------------------------------------------
config = {
    'gui': False,
    'max_steps': 3600,
    'num_episodes': 100,
    'yellow_time': 3,
    'min_green_time': 10,
    'max_green_time': 60,
    'reward_weight_waiting': -0.5,
    'reward_weight_emergency': 5.0,
    'reward_weight_switch': -1.0,
    'actor_lr': 0.0005,
    'critic_lr': 0.001,
    'discount_factor': 0.95,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'batch_size': 64,
    'model_path': 'models_actor_critic/',
    'results_path': 'results_actor_critic/',
    'state_size': 20,
    'action_size': 4,
    'hidden_size': 128,
    'load_model': False,
    'save_model': True
}

# --------------------------------------------------------------------------------
# ACTOR–CRITIC NETWORK DEFINITIONS
# --------------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_pi(x)
        return F.softmax(logits, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

# --------------------------------------------------------------------------------
# TRAFFIC SIGNAL AGENT (ACTOR–CRITIC)
# --------------------------------------------------------------------------------

class TrafficSignalAgentAC:
    def __init__(
        self, 
        tls_id, 
        incoming_lanes, 
        outgoing_lanes, 
        state_size, 
        action_size, 
        hidden_size,
        phase_count,
        actor_lr, 
        critic_lr, 
        discount_factor
    ):
        self.tls_id = tls_id
        self.incoming_lanes = incoming_lanes
        self.outgoing_lanes = outgoing_lanes
        self.state_size = state_size
        self.action_size = action_size
        self.phase_count = phase_count
        self.discount_factor = discount_factor

        self.actor = ActorNetwork(state_size, action_size, hidden_size)
        self.critic = CriticNetwork(state_size, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.current_phase = 0
        self.time_since_last_change = 0
        self.current_phase_duration = 0
        self.episode_rewards = []
        self.total_waiting_time = 0
        self.total_emergency_vehicles_handled = 0
        self.total_switches = 0

    def get_state(self):
        state = np.zeros(self.state_size)
        for i, lane in enumerate(self.incoming_lanes):
            if i < len(state) // 4:
                state[i] = len(traci.lane.getLastStepVehicleIDs(lane)) / 20.0
        offset = len(state) // 4
        for i, lane in enumerate(self.incoming_lanes):
            if i + offset < len(state) // 2:
                emergency_count = sum(
                    1 for v in traci.lane.getLastStepVehicleIDs(lane)
                    if traci.vehicle.getVehicleClass(v) == "emergency"
                )
                state[i + offset] = emergency_count / 5.0
        phase_index = self.current_phase + (len(state) // 2)
        if phase_index < len(state):
            state[phase_index] = 1.0
        state[-1] = min(self.time_since_last_change / 60.0, 1.0)
        return state

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state_tensor).squeeze(0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_value(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return self.critic(state_tensor).squeeze(0)

    def update(self, log_prob, value, reward, next_value, done):
        target = reward + (1 - done) * self.discount_factor * next_value.detach()
        advantage = target - value
        critic_loss = advantage.pow(2).mean()
        actor_loss = -(log_prob * advantage.detach())
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def get_reward(self):
        total_waiting_time = 0
        for lane in self.incoming_lanes:
            for vehicle in traci.lane.getLastStepVehicleIDs(lane):
                total_waiting_time += traci.vehicle.getWaitingTime(vehicle)
        reward = config['reward_weight_waiting'] * total_waiting_time
        self.total_waiting_time += total_waiting_time

        emergency_vehicles_handled = 0
        for lane in self.incoming_lanes:
            if self._is_lane_with_green_light(lane):
                emergency_count = sum(
                    1 for v in traci.lane.getLastStepVehicleIDs(lane)
                    if traci.vehicle.getVehicleClass(v) == "emergency"
                )
                emergency_vehicles_handled += emergency_count
        reward += config['reward_weight_emergency'] * emergency_vehicles_handled
        self.total_emergency_vehicles_handled += emergency_vehicles_handled

        if self.time_since_last_change < config['min_green_time']:
            reward += config['reward_weight_switch']
            self.total_switches += 1
        return reward

    def _is_lane_with_green_light(self, lane):
        for link in traci.lane.getLinks(lane):
            tls_link_index = link[3]
            if tls_link_index >= 0:
                tls_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
                return tls_state[tls_link_index] == 'G'
        return False

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, os.path.join(path, f'agent_{self.tls_id}.pth'))

    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, f'agent_{self.tls_id}.pth'))
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

# --------------------------------------------------------------------------------
# TRAFFIC MANAGEMENT SYSTEM
# --------------------------------------------------------------------------------

class TrafficManagementSystemAC:
    def __init__(self, sumo_config, gui=False):
        self.sumo_config = sumo_config
        self.gui = gui
        self.agents = {}
        self.episode = 0
        self.step = 0

        self.total_rewards = []
        self.total_waiting_times = []
        self.total_emergency_vehicles_handled = []
        self.total_switches = []

        self._start_sumo()
        self._initialize_agents()

    def _start_sumo(self):
        sumo_binary = sumolib.checkBinary('sumo-gui' if self.gui else 'sumo')
        sumo_cmd = [sumo_binary, "-c", self.sumo_config, "--no-warnings", "--waiting-time-memory", "1000"]
        traci.start(sumo_cmd)

    def _initialize_agents(self):
        traffic_lights = traci.trafficlight.getIDList()
        for tls_id in traffic_lights:
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            incoming_lanes = set()
            outgoing_lanes = set()
            for link_tuple in controlled_links:
                for link in link_tuple:
                    incoming_lanes.add(link[0])
                    outgoing_lanes.add(link[1])
            programs = traci.trafficlight.getAllProgramLogics(tls_id)
            phase_count = len(programs[0].phases) if programs else 0
            agent = TrafficSignalAgentAC(
                tls_id=tls_id,
                incoming_lanes=list(incoming_lanes),
                outgoing_lanes=list(outgoing_lanes),
                state_size=config['state_size'],
                action_size=config['action_size'],
                hidden_size=config['hidden_size'],
                phase_count=phase_count,
                actor_lr=config['actor_lr'],
                critic_lr=config['critic_lr'],
                discount_factor=config['discount_factor']
            )
            self.agents[tls_id] = agent

    def run_episode(self, evaluate=False):
        traci.close()
        self._start_sumo()
        for agent in self.agents.values():
            agent.time_since_last_change = 0
            agent.current_phase_duration = 0
            agent.episode_rewards = []
            agent.total_waiting_time = 0
            agent.total_emergency_vehicles_handled = 0
            agent.total_switches = 0

        states = {tls_id: agent.get_state() for tls_id, agent in self.agents.items()}
        self.step = 0
        done = False

        while self.step < config['max_steps'] and not done:
            actions_info = {}
            for tls_id, agent in self.agents.items():
                action, log_prob = agent.select_action(states[tls_id])
                actions_info[tls_id] = (action, log_prob)
            for tls_id, (action, _) in actions_info.items():
                self._apply_action(tls_id, action)
            traci.simulation.step()
            self.step += 1
            for agent in self.agents.values():
                agent.time_since_last_change += 1
                agent.current_phase_duration += 1
            next_states = {}
            rewards = {}
            for tls_id, agent in self.agents.items():
                next_states[tls_id] = agent.get_state()
                rewards[tls_id] = agent.get_reward()
                agent.episode_rewards.append(rewards[tls_id])
            if traci.simulation.getMinExpectedNumber() <= 0:
                done = True
            for tls_id, agent in self.agents.items():
                action, log_prob = actions_info[tls_id]
                reward = rewards[tls_id]
                value_s = agent.get_value(states[tls_id])
                value_s_next = agent.get_value(next_states[tls_id])
                agent.update(log_prob, value_s, reward, value_s_next, float(done))
            states = next_states

        episode_reward = sum(sum(agent.episode_rewards) for agent in self.agents.values())
        self.total_rewards.append(episode_reward)
        self.total_waiting_times.append(sum(agent.total_waiting_time for agent in self.agents.values()))
        self.total_emergency_vehicles_handled.append(sum(agent.total_emergency_vehicles_handled for agent in self.agents.values()))
        self.total_switches.append(sum(agent.total_switches for agent in self.agents.values()))
        self.episode += 1

    def _apply_action(self, tls_id, action):
        agent = self.agents[tls_id]
        if agent.current_phase_duration >= config['max_green_time']:
            new_phase = (agent.current_phase + 1) % agent.phase_count
            self._set_phase(tls_id, new_phase)
            agent.time_since_last_change = 0
            agent.current_phase_duration = 0
            return
        if action == 0:
            return
        if agent.time_since_last_change < config['min_green_time']:
            return
        if 0 < action <= 3:
            new_phase = action - 1
            if new_phase < agent.phase_count:
                self._set_phase(tls_id, new_phase)
                agent.time_since_last_change = 0
                agent.current_phase_duration = 0

    def _set_phase(self, tls_id, phase):
        agent = self.agents[tls_id]
        current_phase = traci.trafficlight.getPhase(tls_id)
        if current_phase == phase:
            return
        yellow_state = self._get_yellow_state(tls_id)
        if yellow_state:
            traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
            for _ in range(config['yellow_time']):
                traci.simulation.step()
                self.step += 1
        traci.trafficlight.setPhaseIndex(tls_id, phase)
        agent.current_phase = phase

    def _get_yellow_state(self, tls_id):
        current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
        return ''.join('y' if c == 'G' else c for c in current_state)

    def plot_results(self, path):
        os.makedirs(path, exist_ok=True)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.total_rewards)
        plt.title('Total Rewards')
        plt.subplot(2, 2, 2)
        plt.plot(self.total_waiting_times)
        plt.title('Total Waiting Times')
        plt.subplot(2, 2, 3)
        plt.plot(self.total_emergency_vehicles_handled)
        plt.title('Emergency Vehicles Handled')
        plt.subplot(2, 2, 4)
        plt.plot(self.total_switches)
        plt.title('Traffic Light Switches')
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'results.png'))
        plt.close()
        pd.DataFrame({
            'rewards': self.total_rewards,
            'waiting_times': self.total_waiting_times,
            'emergency_vehicles': self.total_emergency_vehicles_handled,
            'switches': self.total_switches
        }).to_csv(os.path.join(path, 'results.csv'), index=False)

    def save_models(self, path):
        for agent in self.agents.values():
            agent.save_model(path)

    def load_models(self, path):
        for agent in self.agents.values():
            agent.load_model(path)

    def run_training(self):
        for _ in range(config['num_episodes']):
            self.run_episode(evaluate=False)
        if config['save_model']:
            self.save_models(config['model_path'])
        self.plot_results(config['results_path'])
        traci.close()

    def run_evaluation(self):
        self.load_models(config['model_path'])
        self.run_episode(evaluate=True)
        self.plot_results(os.path.join(config['results_path'], 'evaluation'))
        traci.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Actor–Critic Multi-Agent RL for Traffic Signal Control')
    parser.add_argument('--sumo-config', required=True, help='Path to SUMO configuration file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=3600, help='Max steps per episode')
    parser.add_argument('--load-model', action='store_true', help='Load pre-trained models')
    parser.add_argument('--no-save', action='store_false', dest='save_model', help='Do not save models')
    parser.add_argument('--model-path', default='models_actor_critic/', help='Path to save/load models')
    parser.add_argument('--results-path', default='results_actor_critic/', help='Path to save results')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation instead of training')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    config.update(vars(args))
    tms = TrafficManagementSystemAC(args.sumo_config, args.gui)
    if config['load_model']:
        tms.load_models(config['model_path'])
    if args.evaluate:
        tms.run_evaluation()
    else:
        tms.run_training()