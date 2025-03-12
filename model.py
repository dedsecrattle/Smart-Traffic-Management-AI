import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

class Vehicle:
    def __init__(self, vehicle_id, arrival_time, direction, is_priority=False):
        """
        Initialize a vehicle in the simulation.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            arrival_time: Time step when vehicle arrives at the intersection
            direction: 0 for North-South, 1 for East-West
            is_priority: Boolean indicating if this is a priority vehicle (e.g., ambulance)
        """
        self.id = vehicle_id
        self.arrival_time = arrival_time
        self.direction = direction  # 0: North-South, 1: East-West
        self.is_priority = is_priority
        self.waiting_time = 0
        self.passed = False
    
    def update_waiting(self):
        """Increase waiting time by 1 time step if vehicle hasn't passed yet"""
        if not self.passed:
            self.waiting_time += 1
    
    def __repr__(self):
        direction_str = "North-South" if self.direction == 0 else "East-West"
        priority_str = " (Priority)" if self.is_priority else ""
        status = "Passed" if self.passed else f"Waiting for {self.waiting_time} steps"
        return f"Vehicle {self.id}{priority_str}: {direction_str}, {status}"

class TrafficSimulation:
    def __init__(
        self, 
        max_vehicles=100, 
        arrival_probability=[0.3, 0.3], 
        priority_probability=0.05,
        min_green_time=3
    ):
        """
        Initialize the traffic simulation environment.
        
        Args:
            max_vehicles: Maximum number of vehicles in the simulation
            arrival_probability: Probability of a new vehicle arriving in each direction per time step
            priority_probability: Probability that a new vehicle is a priority vehicle
            min_green_time: Minimum number of time steps a light must remain green before changing
        """
        self.max_vehicles = max_vehicles
        self.arrival_probability = arrival_probability
        self.priority_probability = priority_probability
        self.min_green_time = min_green_time
        

        self.current_light = 0
        
        self.vehicles = []
        self.vehicle_count = 0
        
        self.time_since_last_change = 0
        
        self.total_waiting_time = 0
        self.passed_vehicles = 0
        self.light_changes = 0
        

        self.waiting_time_history = []
        self.queue_length_history = []
        self.light_state_history = []
    
    def get_state(self):
        """
        Get the current state representation for the RL agent.
        
        Returns:
            A tuple containing:
            - Queue lengths in each direction
            - Number of priority vehicles in each direction
            - Time since last traffic light change
            - Current light state
        """
        queue_lengths = [0, 0]
        priority_counts = [0, 0]
        
        for vehicle in self.vehicles:
            if not vehicle.passed:
                queue_lengths[vehicle.direction] += 1
                if vehicle.is_priority:
                    priority_counts[vehicle.direction] += 1
        
        return (
            queue_lengths[0],  # Queue length North-South
            queue_lengths[1],  # Queue length East-West
            priority_counts[0],  # Priority vehicles North-South
            priority_counts[1],  # Priority vehicles East-West
            self.time_since_last_change,  # Time since last light change
            self.current_light  # Current light state
        )
    
    def generate_vehicles(self, time_step):
        """Generate new vehicles based on arrival probabilities"""
        for direction in [0, 1]:  # 0: North-South, 1: East-West
            if random.random() < self.arrival_probability[direction]:
                is_priority = random.random() < self.priority_probability
                self.vehicles.append(
                    Vehicle(
                        self.vehicle_count, 
                        time_step, 
                        direction, 
                        is_priority
                    )
                )
                self.vehicle_count += 1
    
    def update_waiting_times(self):
        """Update waiting times for all vehicles"""
        for vehicle in self.vehicles:
            vehicle.update_waiting()
    
    def process_vehicles(self):
        """Process vehicles that can pass through the intersection"""
        # Sort vehicles by priority and arrival time
        vehicles_to_process = [v for v in self.vehicles if not v.passed]
        vehicles_to_process.sort(
            key=lambda v: (v.direction != self.current_light, not v.is_priority, v.arrival_time)
        )
        
        # Process vehicles in the current green light direction
        vehicles_passed = 0
        max_vehicles_per_step = 3  # Number of vehicles that can pass in one time step
        
        for vehicle in vehicles_to_process:
            if vehicle.direction == self.current_light and vehicles_passed < max_vehicles_per_step:
                vehicle.passed = True
                self.total_waiting_time += vehicle.waiting_time
                self.passed_vehicles += 1
                vehicles_passed += 1
    
    def step(self, action):
        """
        Take a step in the simulation based on the action.
        
        Args:
            action: 0 to keep current light, 1 to change light
        
        Returns:
            next_state: The new state after taking the action
            reward: The reward for taking the action
            done: Whether the episode is done
            info: Additional information
        """
        can_change = self.time_since_last_change >= self.min_green_time
        
        changed_light = False
        if action == 1 and can_change:
            self.current_light = 1 - self.current_light  # Toggle light
            self.time_since_last_change = 0
            self.light_changes += 1
            changed_light = True
        else:
            self.time_since_last_change += 1
        
        self.process_vehicles()
        self.update_waiting_times()
        
        ns_queue = sum(1 for v in self.vehicles if not v.passed and v.direction == 0)
        ew_queue = sum(1 for v in self.vehicles if not v.passed and v.direction == 1)
        

        ns_priority = sum(1 for v in self.vehicles if not v.passed and v.direction == 0 and v.is_priority)
        ew_priority = sum(1 for v in self.vehicles if not v.passed and v.direction == 1 and v.is_priority)
        
        waiting_penalty = -(ns_queue + ew_queue) * 0.1
        priority_penalty = -(ns_priority + ew_priority) * 1.0
        

        change_penalty = -2.0 if changed_light else 0
        
        reward = waiting_penalty + priority_penalty + change_penalty
        

        done = self.passed_vehicles >= self.max_vehicles
        
        self.waiting_time_history.append(sum(v.waiting_time for v in self.vehicles if not v.passed))
        self.queue_length_history.append((ns_queue, ew_queue))
        self.light_state_history.append(self.current_light)
        
        next_state = self.get_state()
        
        self.generate_vehicles(len(self.waiting_time_history))
        
        info = {
            "ns_queue": ns_queue,
            "ew_queue": ew_queue,
            "ns_priority": ns_priority,
            "ew_priority": ew_priority,
            "changed_light": changed_light,
            "vehicles_passed": self.passed_vehicles,
        }
        
        return next_state, reward, done, info
    
    def reset(self):
        """Reset the simulation to initial state"""
        self.vehicles = []
        self.vehicle_count = 0
        self.current_light = 0
        self.time_since_last_change = 0
        self.total_waiting_time = 0
        self.passed_vehicles = 0
        self.light_changes = 0
        self.waiting_time_history = []
        self.queue_length_history = []
        self.light_state_history = []
        

        self.generate_vehicles(0)
        
        return self.get_state()
    
    def render(self):
        """Render the current state of the traffic simulation"""
        print(f"\nTime step: {len(self.waiting_time_history)}")
        print(f"Traffic light: {'North-South Green' if self.current_light == 0 else 'East-West Green'}")
        print(f"Time since last change: {self.time_since_last_change}")
        
        ns_waiting = [v for v in self.vehicles if not v.passed and v.direction == 0]
        ew_waiting = [v for v in self.vehicles if not v.passed and v.direction == 1]
        
        print(f"Vehicles waiting North-South: {len(ns_waiting)} (Priority: {sum(v.is_priority for v in ns_waiting)})")
        print(f"Vehicles waiting East-West: {len(ew_waiting)} (Priority: {sum(v.is_priority for v in ew_waiting)})")
        print(f"Vehicles passed: {self.passed_vehicles}/{self.max_vehicles}")
        print(f"Total waiting time: {self.total_waiting_time}")
        print(f"Traffic light changes: {self.light_changes}")
    
    def visualize_results(self):
        """Visualize simulation results"""
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 1, 1)
        ns_queue = [q[0] for q in self.queue_length_history]
        ew_queue = [q[1] for q in self.queue_length_history]
        plt.plot(ns_queue, label='North-South Queue')
        plt.plot(ew_queue, label='East-West Queue')
        plt.xlabel('Time Step')
        plt.ylabel('Queue Length')
        plt.title('Queue Lengths Over Time')
        plt.legend()
        plt.grid(True)
        

        plt.subplot(3, 1, 2)
        plt.plot(self.waiting_time_history)
        plt.xlabel('Time Step')
        plt.ylabel('Total Waiting Time')
        plt.title('Total Waiting Time Over Time')
        plt.grid(True)
        

        plt.subplot(3, 1, 3)
        plt.plot(self.light_state_history)
        plt.xlabel('Time Step')
        plt.ylabel('Light State (0: NS, 1: EW)')
        plt.title('Traffic Light State Over Time')
        plt.yticks([0, 1], ['North-South', 'East-West'])
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        """
        Initialize a Q-learning agent.
        
        Args:
            state_size: Size of the state space (for creating state encoding)
            action_size: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Rate at which exploration rate decays
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.q_table = {}
        
    def encode_state(self, state):
        """
        Encode the state to use as a key in the Q-table.
        
        Args:
            state: The state tuple (ns_queue, ew_queue, ns_priority, ew_priority, time_since_change, current_light)
        
        Returns:
            A string representation of the state suitable for the Q-table
        """
        ns_queue, ew_queue, ns_priority, ew_priority, time_since_change, current_light = state
        
        ns_queue_bucket = min(3, ns_queue // 3)
        ew_queue_bucket = min(3, ew_queue // 3)
        

        ns_priority_bucket = min(2, ns_priority)
        ew_priority_bucket = min(2, ew_priority)
        
        time_bucket = 1 if time_since_change >= 3 else 0
        
        return f"{ns_queue_bucket}_{ew_queue_bucket}_{ns_priority_bucket}_{ew_priority_bucket}_{time_bucket}_{current_light}"
    
    def get_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: The current state
        
        Returns:
            The chosen action
        """
        encoded_state = self.encode_state(state)
        
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_size)
        
        if encoded_state not in self.q_table:
            self.q_table[encoded_state] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[encoded_state])
    
    def train(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-learning algorithm.
        
        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The next state
            done: Whether the episode is done
        """
        encoded_state = self.encode_state(state)
        encoded_next_state = self.encode_state(next_state)
        
        if encoded_state not in self.q_table:
            self.q_table[encoded_state] = np.zeros(self.action_size)
        
        if encoded_next_state not in self.q_table:
            self.q_table[encoded_next_state] = np.zeros(self.action_size)
        
        q_current = self.q_table[encoded_state][action]
        q_next_max = np.max(self.q_table[encoded_next_state]) if not done else 0
        
        self.q_table[encoded_state][action] = q_current + self.learning_rate * (
            reward + self.discount_factor * q_next_max - q_current
        )
        
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


def train_agent(episodes=100, max_steps=200, render_freq=20):
    """
    Train a Q-learning agent on the traffic simulation.
    
    Args:
        episodes: Number of episodes to train
        max_steps: Maximum number of steps per episode
        render_freq: Frequency of episodes to render
    
    Returns:
        The trained agent and the environment
    """
    env = TrafficSimulation(max_vehicles=50)
    agent = QLearningAgent(state_size=6, action_size=2)
    
    rewards_history = []
    waiting_time_history = []
    light_changes_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        rewards_history.append(total_reward)
        waiting_time_history.append(env.total_waiting_time / max(1, env.passed_vehicles))
        light_changes_history.append(env.light_changes)
        
        if (episode + 1) % render_freq == 0:
            print(f"Episode: {episode+1}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Average Waiting Time: {waiting_time_history[-1]:.2f}")
            print(f"Traffic Light Changes: {light_changes_history[-1]}")
            print(f"Exploration Rate: {agent.exploration_rate:.2f}")
            print("-----------------------------")
            
            if episode == episodes - 1:
                env.visualize_results()
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(waiting_time_history)
    plt.xlabel('Episode')
    plt.ylabel('Average Waiting Time')
    plt.title('Average Waiting Time per Episode')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(light_changes_history)
    plt.xlabel('Episode')
    plt.ylabel('Traffic Light Changes')
    plt.title('Traffic Light Changes per Episode')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return agent, env


def run_baseline_policy(max_steps=200):
    """
    Run a baseline policy for comparison that changes the light based on simple rules.
    
    Args:
        max_steps: Maximum number of steps to run
    
    Returns:
        The environment after running the baseline policy
    """
    env = TrafficSimulation(max_vehicles=50)
    state = env.reset()
    
    for step in range(max_steps):
        ns_queue, ew_queue, ns_priority, ew_priority, time_since_change, current_light = state
        
        if time_since_change < env.min_green_time:
            action = 0  # Keep current light
        else:
            current_dir_weight = (ns_queue + 3 * ns_priority) if current_light == 0 else (ew_queue + 3 * ew_priority)
            other_dir_weight = (ew_queue + 3 * ew_priority) if current_light == 0 else (ns_queue + 3 * ns_priority)
            
            action = 1 if other_dir_weight > current_dir_weight else 0
        
        state, reward, done, info = env.step(action)
        
        if done:
            break
    
    print("\nBaseline Policy Results:")
    print(f"Total Waiting Time: {env.total_waiting_time}")
    print(f"Average Waiting Time: {env.total_waiting_time / max(1, env.passed_vehicles):.2f}")
    print(f"Traffic Light Changes: {env.light_changes}")
    
    env.visualize_results()
    return env



if __name__ == "__main__":

    print("Training RL agent...")
    agent, env = train_agent(episodes=100, max_steps=200, render_freq=10)
    
    print("\nRunning baseline policy for comparison...")
    baseline_env = run_baseline_policy(max_steps=200)
    
    rl_avg_waiting = env.total_waiting_time / max(1, env.passed_vehicles)
    baseline_avg_waiting = baseline_env.total_waiting_time / max(1, baseline_env.passed_vehicles)
    
    print("\nComparison:")
    print(f"RL Agent - Average Waiting Time: {rl_avg_waiting:.2f}, Light Changes: {env.light_changes}")
    print(f"Baseline - Average Waiting Time: {baseline_avg_waiting:.2f}, Light Changes: {baseline_env.light_changes}")