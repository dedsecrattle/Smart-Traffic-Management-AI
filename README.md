# Multi-Agent Reinforcement Learning for Traffic Signal Control

## Overview
This project implements a multi-agent reinforcement learning approach for optimizing traffic signal control using SUMO (Simulation of Urban MObility) and PyTorch. Each traffic signal acts as an independent agent, learning to optimize traffic flow through Deep Q-Networks (DQN).

## Features
- **Multi-Agent System**: Each traffic signal is controlled by an independent reinforcement learning agent.
- **Deep Q-Network (DQN)**: Uses deep reinforcement learning to optimize traffic light timings.
- **Experience Replay**: Implements a replay buffer to store and sample past experiences.
- **Emergency Vehicle Handling**: Gives priority to emergency vehicles.
- **Evaluation Mode**: Allows evaluation of trained models in a simulation environment.
- **Visualization**: Generates plots for rewards, waiting times, emergency vehicle handling, and traffic light switches.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- SUMO (Simulation of Urban MObility)
- PyTorch

### Install Dependencies
Run the following command to install required dependencies:
```sh
pip install -r requirements.txt
```

## Usage
### Running Training
To train the traffic signal control system, run:
```sh
python main.py --sumo-config <path_to_sumo_config>
```

Optional arguments:
- `--gui`: Enables SUMO GUI visualization.
- `--episodes`: Number of training episodes (default: 100).
- `--steps`: Maximum steps per episode (default: 3600).
- `--load`: Load pre-trained models.
- `--no-save`: Disable model saving.
- `--model-path`: Path to save/load models.
- `--results-path`: Path to save results.

### Running Evaluation
To evaluate a trained model:
```sh
python main.py --sumo-config <path_to_sumo_config> --evaluate --load
```

## Project Structure
```
├── main.py                # Main entry point
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── models/                # Saved models
├── results/               # Training results
```

## Results and Visualization
After training, results are saved in `results/` and can be visualized using the generated plots.

## License
This project is licensed under the MIT License.