# SMART Signal Control with Reinforcement Learning

This project implements and evaluates various reinforcement learning algorithms for traffic signal control using the SUMO traffic simulator and SUMO-RL framework. The repository contains multiple implementations, including Q-Learning, SARSA, DQN, PPO, TRPO, and PCQL (Penalized Conservative Q-Learning).

## Project Structure

The project is organized as follows:

```
CS4246/
├── algorithms/
│   ├── DQN/
│   ├── PCQL/
│   ├── PPO/
│   ├── PPO-LSTM/
│   ├── Q-Learning/
│   ├── SARSA/
│   ├── TRPO/
│   └── sumo-config/ (contains SUMO configuration files)
├── utils/
│   ├── plot.py (for plotting results)
│   ├── reward_functions.py (custom reward functions)
│   └── reward_logger.py (base reward logging functionality)
└── Deprecated/ (old implementations)
```

Each algorithm folder contains:
- Implementation of the specific RL algorithm
- A reward_logger.py file for tracking metrics
- Scripts for running simulations with different network configurations

## Requirements

### Prerequisites

1. [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/) - version 1.8.0 or higher
2. Python 3.8 or higher

### Python Dependencies

```
torch
numpy
pandas
matplotlib
gymnasium
sumo-rl
stable-baselines3
sb3-contrib (for TRPO)
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd CS4246
   ```

2. Install SUMO following the [official instructions](https://sumo.dlr.de/docs/Installing/index.html)

3. Set the SUMO_HOME environment variable:
   ```bash
   # For Windows
   set SUMO_HOME=C:\path\to\sumo

   # For Linux/Mac
   export SUMO_HOME=/path/to/sumo
   ```

4. Install Python dependencies:
   ```bash
   pip install torch numpy pandas matplotlib gymnasium sumo-rl stable-baselines3 sb3-contrib
   ```

## Running Simulations

### PCQL Algorithm Example

To run the PCQL (Penalized Conservative Q-Learning) algorithm on a single intersection scenario:

```bash
cd algorithms/PCQL
python pcql_single_intersection.py
```

This will:
1. Create an output directory with timestamped results
2. Run the SUMO simulation for the specified number of episodes
3. Log rewards and other metrics for each episode

### Other Algorithms

Similar commands can be used for other algorithms:

```bash
cd algorithms/DQN
python dqn_single_intersection.py

cd algorithms/PPO
python ppo_4x4grid.py
```

## Generating Plots

The project includes a versatile plotting utility to visualize experiment results:

```bash
python utils/plot.py -f <path_to_csv_files> -xaxis step -yaxis reward -ma 100
```

Parameters:
- `-f`: Path pattern to CSV files containing results
- `-xaxis`: Column to use for the x-axis (e.g., 'step')
- `-yaxis`: Column to use for the y-axis (e.g., 'reward', 'waiting_time', 'queue_length')
- `-ma`: Moving average window size to smooth the plots
- `-t`: Plot title
- `-l`: Labels for the plotted lines
- `-output`: Output filename for saving the plot (PDF format)

Example:
```bash
python utils/plot.py -f outputs/single-intersection/pcql_2023-04-15_14-30-00/pcql_rewards_episode_1.csv -xaxis step -yaxis reward -ma 100 -t "PCQL Performance" -output pcql_results.pdf
```

## Logging and Metrics

The `RewardLogger` class (in each algorithm's folder) tracks various metrics during simulation:
- Rewards
- Queue lengths
- Waiting times
- Vehicle throughput
- Fuel consumption and emissions

Example output CSV structure:
```
step,action,reward,queue_length,waiting_time,vehicles_passed,fuel_consumption,co2_emission
```

## Customizing Simulations

To modify simulation parameters, edit the respective algorithm script:
- Change network and route files in the `SumoEnvironment` constructor
- Adjust simulation duration with `num_seconds`
- Modify hyperparameters like learning rate, discount factor, etc.
- Toggle GUI with `use_gui=True/False`

## Additional Information

For more details on:
- SUMO: [SUMO Documentation](https://sumo.dlr.de/docs/)
- SUMO-RL: [SUMO-RL GitHub](https://github.com/LucasAlegre/sumo-rl)
- RL Algorithms: See individual implementations in the algorithms directory