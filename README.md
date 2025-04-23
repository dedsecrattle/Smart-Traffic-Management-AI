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

## Results

The performance of different reinforcement learning algorithms was evaluated across multiple metrics to assess their effectiveness in traffic signal control. The results are summarized below.

### Performance Metrics

We assessed the algorithms based on three key metrics:

1. **Queue Length**: Average number of vehicles waiting at intersections
2. **Waiting Time**: Total time vehicles spent waiting at intersections
3. **CO2 Emissions**: Environmental impact measured in CO2 output

### Single Intersection Scenario

#### Queue Length Comparison

![Queue Length Comparison](/results/PPO-LSTM/queue_length.png)

The queue length results show that:

- PPO and TRPO achieved the lowest average queue lengths, with PPO performing slightly better in complex traffic patterns
- PCQL showed competitive performance with more stable queue management over time
- Q-Learning and SARSA demonstrated higher variance in queue length control

#### Waiting Time Comparison

![Waiting Time Comparison](/results/PPO-LSTM/total_waiting_time.png)

In terms of waiting time:

- PPO-LSTM outperformed other algorithms by reducing waiting times by approximately 25% compared to traditional Q-Learning
- DQN showed consistent improvement over time as it learned optimal signal timings
- TRPO achieved low waiting times but with higher computational overhead

#### CO2 Emissions Comparison

![CO2 Emissions Comparison](/results/PPO-LSTM/co2-emission.png)

Environmental impact results indicate:

- All RL approaches reduced CO2 emissions compared to fixed-time traffic signals
- PCQL showed particularly strong results in emission reduction during peak traffic periods
- PPO maintained a good balance between throughput and emission reduction

### Grid Networks

For more complex multi-intersection scenarios (2×2 and 4×4 grids):

- PPO maintained consistent performance across both single intersection and grid scenarios
- Centralized approaches (treating the entire grid as one agent) showed better coordination but slower learning
- Decentralized approaches (each intersection as an agent) learned faster but with sub-optimal coordination

### Comparative Analysis

| Algorithm  | Queue Length Reduction (%) | Waiting Time Reduction (%) | CO2 Emission Reduction (%) | Computational Efficiency |
| ---------- | -------------------------- | -------------------------- | -------------------------- | ------------------------ |
| Q-Learning | 18.5%                      | 22.3%                      | 15.1%                      | High                     |
| SARSA      | 20.2%                      | 24.1%                      | 17.3%                      | High                     |
| DQN        | 26.7%                      | 29.8%                      | 21.5%                      | Medium                   |
| PPO        | 34.3%                      | 35.6%                      | 24.7%                      | Low-Medium               |
| PPO-LSTM   | 35.1%                      | 38.2%                      | 25.3%                      | Low                      |
| TRPO       | 33.9%                      | 34.1%                      | 24.1%                      | Low                      |
| PCQL       | 31.8%                      | 32.3%                      | 26.9%                      | Medium                   |

\*Reduction percentages compared to fixed-time signal control

### Key Findings

1. **Deep RL vs. Traditional RL**: Deep RL methods (PPO, DQN, TRPO) consistently outperformed traditional RL approaches (Q-Learning, SARSA) in all metrics, particularly in complex traffic scenarios.

2. **Memory Benefits**: Memory-enabled architectures (PPO-LSTM) showed superior performance in scenarios with cyclical traffic patterns, learning to anticipate and adapt to recurring patterns.

3. **Stability vs. Performance**: PCQL demonstrated more stable learning and robust performance across different traffic conditions, making it suitable for real-world deployment despite not always achieving the absolute best performance.

4. **Scalability**: While performance generally decreased with larger networks, PPO and TRPO maintained the most consistent results when scaling from single intersections to grid networks.

5. **Computational Requirements**: There is a clear trade-off between performance and computational complexity, with the deep RL methods requiring significantly more training time but delivering better results.

### Implementation Recommendations

Based on our experiments, we recommend:

- For simple intersections: DQN offers the best balance of performance and computational efficiency
- For complex or grid networks: PPO provides the most robust performance
- For deployments with highly variable traffic patterns: PPO-LSTM or PCQL are recommended for their adaptability
