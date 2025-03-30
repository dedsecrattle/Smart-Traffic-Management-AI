# SUMO-RL Multi-Algorithm Project

This project uses [SUMO-RL](https://github.com/LucasAlegre/sumo-rl) environment with multiple algorithms from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). 

## Files
- `train_dqn.py`, `train_a2c.py`, `train_ppo.py`: Each script trains a different RL algorithm.
- `evaluate.py`: Loads a saved model, runs a SUMO simulation for evaluation, and plots results.
- `sumo-config/`: Contains different Configuration of Network

## Steps
1. **Install** Python packages:
   ```bash
   pip install -r requirements.txt
2. **Run Training**:
    ```bash
    python train_dqn.py
    ```
3. **Evaluate Model**
    ```bash
    python evaluate.py
    ```
