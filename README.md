# Smart Traffic Management System

This project implements a reinforcement learning approach to traffic light control for optimizing traffic flow at intersections. It's the first iteration of a multi-agent coordination system for smart traffic management.

## Project Overview

Traffic congestion in urban areas causes delays, pollution, and economic losses. Traditional traffic light control systems use fixed schedules or simple adaptive heuristics that fail to respond effectively to real-time traffic conditions. This project aims to incorporate reinforcement learning to optimize traffic signals dynamically, improving traffic flow and reducing congestion.

## Features

- **Traffic Simulation Environment**: Simulates vehicles arriving at and passing through an intersection
- **Vehicle Prioritization**: Supports priority vehicles (e.g., ambulances) that receive preferential treatment
- **Q-Learning Agent**: Learns optimal traffic light control policies based on traffic conditions
- **Performance Visualization**: Tools to visualize queue lengths, waiting times, and traffic light patterns
- **Baseline Comparison**: Includes a rule-based baseline strategy for performance comparison

## Getting Started

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/smart-traffic-management.git
cd smart-traffic-management
pip install -r requirements.txt
```

### Usage

To run the simulation with the Q-learning agent:

```python
from traffic_management import train_agent

# Train the RL agent
agent, env = train_agent(episodes=100, max_steps=200, render_freq=10)
```

To run with the baseline policy for comparison:

```python
from traffic_management import run_baseline_policy

# Run baseline policy
baseline_env = run_baseline_policy(max_steps=200)
```

## Simulation Details

### Environment Parameters

- `max_vehicles`: Maximum number of vehicles in the simulation
- `arrival_probability`: Probability of a new vehicle arriving in each direction per time step
- `priority_probability`: Probability that a new vehicle is a priority vehicle
- `min_green_time`: Minimum number of time steps a light must remain green before changing

### Vehicle Properties

- Each vehicle has a direction (North-South or East-West)
- Vehicles can be marked as priority vehicles
- Waiting time is tracked for each vehicle

### Reward Function

The reward function considers:

- Penalties for vehicles waiting at the intersection
- Higher penalties for priority vehicles waiting
- Penalties for changing the traffic light too frequently

### State Representation

The state space includes:

- Queue lengths in each direction
- Number of priority vehicles in each direction
- Time since last traffic light change
- Current light state

## Project Structure

```
smart-traffic-management/
├── traffic_management.py    # Main implementation file
├── README.md                # Project documentation
└── requirements.txt         # Project library requirements
```

## Results

The Q-learning agent learns to:

- Prioritize directions with longer queues
- Give preference to queues containing priority vehicles
- Balance the need to change lights with maintaining consistent flow
- Respect minimum green time constraints

## Next Model plannings

next iteration will include:

- Multi-intersection scenarios with coordinating agents
- More sophisticated MARL algorithms (QMIX, MADDPG)
- Enhanced traffic simulation with realistic vehicle dynamics
- Additional metrics for performance evaluation
- Integration with traffic simulation frameworks like SUMO

## Acknowledgments

- CS4246: AI Planning and Decision Making course
- National University of Singapore, School of Computing
