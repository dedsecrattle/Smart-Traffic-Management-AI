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
from torch.distributions import Categorical

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

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
    'learning_rate': 0.001,   
    'discount_factor': 0.95,   
    'epsilon_start': 1.0,      
    'epsilon_end': 0.01,       
    'epsilon_decay': 0.995,   
    'batch_size': 64,         
    'memory_size': 10000,     
    'target_update': 10,       
    'state_size': 20,         
    'action_size': 4,          
    'hidden_size': 128,       
    'load_model': False,       
    'save_model': True,        
    'model_path': 'models/',   
    'results_path': 'results/'
}

class ReplayBuffer:
    """Experience replay buffer to store and sample experiences."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q-Network for traffic signal control."""
    
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TrafficSignalAgent:
    """
    Agent class for a single traffic signal.
    Each traffic light is modeled as an intelligent agent.
    """
    
    def __init__(self, tls_id, incoming_lanes, outgoing_lanes, state_size, action_size, hidden_size,
                 learning_rate, discount_factor, epsilon_start, epsilon_end, epsilon_decay):
        self.tls_id = tls_id
        self.incoming_lanes = incoming_lanes
        self.outgoing_lanes = outgoing_lanes
        self.state_size = state_size
        self.action_size = action_size
        
        self.current_phase = 0
        self.time_since_last_change = 0
        self.current_phase_duration = 0
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        
        self.policy_net = DQN(state_size, action_size, hidden_size)
        self.target_net = DQN(state_size, action_size, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.memory = ReplayBuffer(config['memory_size'])
        
        self.episode_rewards = []
        self.total_waiting_time = 0
        self.total_emergency_vehicles_handled = 0
        self.total_switches = 0
        
    def get_state(self):
        """
        Get the current state representation for the traffic light.
        State includes:
        - Queue length for each incoming lane
        - Number of emergency vehicles for each incoming lane
        - Current phase (one-hot encoding)
        - Time since last phase change
        """
        state = np.zeros(self.state_size)
        
        for i, lane in enumerate(self.incoming_lanes):
            if i < len(state) // 4:
                queue_length = self._get_queue_length(lane)
                state[i] = queue_length / 20.0  # Normalize
                
        for i, lane in enumerate(self.incoming_lanes):
            if i < len(state) // 4:
                emergency_count = self._get_emergency_vehicles(lane)
                state[i + len(state) // 4] = emergency_count / 5.0  # Normalize
                
        phase_index = self.current_phase + len(state) // 2
        if phase_index < len(state):
            state[phase_index] = 1.0
            
        state[-1] = min(self.time_since_last_change / 60.0, 1.0)  # Normalize
        
        return state
    
    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update_model(self, batch_size):
        """Update the model using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        
        expected_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        """Update the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_reward(self):
        """
        Calculate the reward for the current state.
        Reward includes:
        - Penalty for waiting time
        - Bonus for handling emergency vehicles
        - Penalty for switching too frequently
        """
        reward = 0
        
        total_waiting_time = 0
        for lane in self.incoming_lanes:
            waiting_time = self._get_waiting_time(lane)
            total_waiting_time += waiting_time
        reward += config['reward_weight_waiting'] * total_waiting_time
        self.total_waiting_time += total_waiting_time
        
        emergency_vehicles_handled = 0
        for lane in self.incoming_lanes:
            if self._is_lane_with_green_light(lane):
                emergency_count = self._get_emergency_vehicles(lane)
                emergency_vehicles_handled += emergency_count
        reward += config['reward_weight_emergency'] * emergency_vehicles_handled
        self.total_emergency_vehicles_handled += emergency_vehicles_handled
        
        if self.time_since_last_change < config['min_green_time']:
            reward += config['reward_weight_switch']
            self.total_switches += 1
            
        return reward
    
    def _get_queue_length(self, lane):
        """Get the number of vehicles in the queue for a lane."""
        return len(traci.lane.getLastStepVehicleIDs(lane))
    
    def _get_waiting_time(self, lane):
        """Get the total waiting time for vehicles in a lane."""
        vehicles = traci.lane.getLastStepVehicleIDs(lane)
        waiting_time = 0
        for vehicle in vehicles:
            waiting_time += traci.vehicle.getWaitingTime(vehicle)
        return waiting_time
    
    def _get_emergency_vehicles(self, lane):
        """Get the number of emergency vehicles in a lane."""
        vehicles = traci.lane.getLastStepVehicleIDs(lane)
        emergency_count = 0
        for vehicle in vehicles:
            if traci.vehicle.getVehicleClass(vehicle) == "emergency":
                emergency_count += 1
        return emergency_count
    
    def _is_lane_with_green_light(self, lane):
        """Check if a lane has a green light."""
        lane_links = traci.lane.getLinks(lane)
        for link in lane_links:
            tls_link_index = link[3]
            if tls_link_index >= 0:
                tls_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
                if tls_state[tls_link_index] == 'G':
                    return True
        return False
    
    def save_model(self, path):
        """Save the model to a file."""
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, os.path.join(path, f'agent_{self.tls_id}.pth'))
    
    def load_model(self, path):
        """Load the model from a file."""
        checkpoint = torch.load(os.path.join(path, f'agent_{self.tls_id}.pth'))
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

class TrafficManagementSystem:
    """
    Main class for the traffic management system.
    Coordinates multiple traffic signal agents.
    """
    
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
        """Start the SUMO simulation."""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_config, "--no-warnings", "--waiting-time-memory", "1000"]
        traci.start(sumo_cmd)
        
    def _initialize_agents(self):
        """Initialize traffic signal agents for all traffic lights in the network."""
        traffic_lights = traci.trafficlight.getIDList()
        
        for tls_id in traffic_lights:
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            incoming_lanes = set()
            outgoing_lanes = set()
            
            for link_tuple in controlled_links:
                for link in link_tuple:
                    if link:
                        incoming_lanes.add(link[0])
                        outgoing_lanes.add(link[1])
            
            agent = TrafficSignalAgent(
                tls_id=tls_id,
                incoming_lanes=list(incoming_lanes),
                outgoing_lanes=list(outgoing_lanes),
                state_size=config['state_size'],
                action_size=config['action_size'],
                hidden_size=config['hidden_size'],
                learning_rate=config['learning_rate'],
                discount_factor=config['discount_factor'],
                epsilon_start=config['epsilon_start'],
                epsilon_end=config['epsilon_end'],
                epsilon_decay=config['epsilon_decay']
            )
            
            self.agents[tls_id] = agent
    
    def run_episode(self):
        """Run a single episode of the simulation."""
        print(f"Starting episode {self.episode + 1}/{config['num_episodes']}")
        
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
            actions = {}
            for tls_id, agent in self.agents.items():
                actions[tls_id] = agent.select_action(states[tls_id])
            
            for tls_id, action in actions.items():
                self._apply_action(tls_id, action)

            traci.simulationStep()
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
                agent.memory.push(
                    states[tls_id],
                    actions[tls_id],
                    rewards[tls_id],
                    next_states[tls_id],
                    done
                )
            
            states = next_states
            
            if self.step % 10 == 0:
                for agent in self.agents.values():
                    agent.update_model(config['batch_size'])
        
        print(f"Episode {self.episode + 1} finished after {self.step} steps")
        
        episode_reward = sum(sum(agent.episode_rewards) for agent in self.agents.values())
        episode_waiting_time = sum(agent.total_waiting_time for agent in self.agents.values())
        episode_emergency_vehicles = sum(agent.total_emergency_vehicles_handled for agent in self.agents.values())
        episode_switches = sum(agent.total_switches for agent in self.agents.values())
        
        self.total_rewards.append(episode_reward)
        self.total_waiting_times.append(episode_waiting_time)
        self.total_emergency_vehicles_handled.append(episode_emergency_vehicles)
        self.total_switches.append(episode_switches)
        
        print(f"Episode reward: {episode_reward}")
        print(f"Episode waiting time: {episode_waiting_time}")
        print(f"Episode emergency vehicles handled: {episode_emergency_vehicles}")
        print(f"Episode switches: {episode_switches}")
        
        if self.episode % config['target_update'] == 0:
            for agent in self.agents.values():
                agent.update_target_network()
        
        for agent in self.agents.values():
            agent.decay_epsilon()
        
        self.episode += 1
    
    def _apply_action(self, tls_id, action):
        """Apply an action to a traffic light."""
        agent = self.agents[tls_id]
        
        if agent.current_phase_duration >= config['max_green_time']:
            new_phase = (agent.current_phase + 1) % traci.trafficlight.getPhaseCount(tls_id)
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
            if new_phase < traci.trafficlight.getPhaseCount(tls_id):
                self._set_phase(tls_id, new_phase)
                agent.time_since_last_change = 0
                agent.current_phase_duration = 0
    
    def _set_phase(self, tls_id, phase):
        """Set the phase of a traffic light with yellow transition."""
        current_phase = traci.trafficlight.getPhase(tls_id)
        
        if current_phase == phase:
            return
        
        yellow_state = self._get_yellow_state(tls_id)
        if yellow_state:
            traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
            for _ in range(config['yellow_time']):
                traci.simulationStep()
                self.step += 1
        
        traci.trafficlight.setPhase(tls_id, phase)
        self.agents[tls_id].current_phase = phase
    
    def _get_yellow_state(self, tls_id):
        """Get the yellow state for a traffic light."""
        current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
        yellow_state = ""
        
        for c in current_state:
            if c == 'G':
                yellow_state += 'y'
            else:
                yellow_state += c
                
        return yellow_state
    
    def plot_results(self, path):
        """Plot the results of the training."""
        if not os.path.exists(path):
            os.makedirs(path)
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.total_rewards)
        plt.title('Total Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(2, 2, 2)
        plt.plot(self.total_waiting_times)
        plt.title('Total Waiting Times')
        plt.xlabel('Episode')
        plt.ylabel('Waiting Time (s)')
        
        plt.subplot(2, 2, 3)
        plt.plot(self.total_emergency_vehicles_handled)
        plt.title('Emergency Vehicles Handled')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        
        plt.subplot(2, 2, 4)
        plt.plot(self.total_switches)
        plt.title('Traffic Light Switches')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'results.png'))
        plt.close()
        
        results = {
            'rewards': self.total_rewards,
            'waiting_times': self.total_waiting_times,
            'emergency_vehicles': self.total_emergency_vehicles_handled,
            'switches': self.total_switches
        }
        pd.DataFrame(results).to_csv(os.path.join(path, 'results.csv'), index=False)
    
    def save_models(self, path):
        """Save all agent models."""
        for tls_id, agent in self.agents.items():
            agent.save_model(path)
    
    def load_models(self, path):
        """Load all agent models."""
        for tls_id, agent in self.agents.items():
            agent.load_model(path)
    
    def run_training(self):
        """Run the full training process."""
        for _ in range(config['num_episodes']):
            self.run_episode()
        
        if config['save_model']:
            self.save_models(config['model_path'])
        
        self.plot_results(config['results_path'])
        traci.close()
        
    def run_evaluation(self):
        """Run a single evaluation episode."""
        self.load_models(config['model_path'])
        
        for agent in self.agents.values():
            agent.epsilon = 0
        
        self.run_episode()
        
        self.plot_results(os.path.join(config['results_path'], 'evaluation'))
        traci.close()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-Agent Reinforcement Learning for Traffic Signal Control')
    
    parser.add_argument('--sumo-config', required=True, help='Path to the SUMO configuration file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=3600, help='Maximum simulation steps per episode')
    parser.add_argument('--load', action='store_true', help='Load pre-trained models')
    parser.add_argument('--no-save', action='store_false', dest='save', help='Do not save models')
    parser.add_argument('--model-path', default='models/', help='Path to save/load models')
    parser.add_argument('--results-path', default='results/', help='Path to save results')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation instead of training')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    config['gui'] = args.gui
    config['num_episodes'] = args.episodes
    config['max_steps'] = args.steps
    config['load_model'] = args.load
    config['save_model'] = args.save
    config['model_path'] = args.model_path
    config['results_path'] = args.results_path
    
    tms = TrafficManagementSystem(args.sumo_config, args.gui)
    
    if config['load_model']:
        tms.load_models(config['model_path'])
    
    if args.evaluate:
        tms.run_evaluation()
    else:
        tms.run_training()
