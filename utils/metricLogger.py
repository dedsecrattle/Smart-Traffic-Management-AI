import csv
import os
from collections import defaultdict

class SumoMetricLogger:
    def __init__(self, env, log_path="custom_metrics_log.csv", log_interval=1):
        self.env = env
        self.log_path = log_path
        self.log_interval = log_interval
        self.step_counter = 0

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "step", "reward", "total_queue", "total_wait_time",
                "avg_wait_time_per_lane", "max_queue_on_lane", "num_active_vehicles"
            ])

    def reset(self):
        self.step_counter = 0
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if self.step_counter % self.log_interval == 0:
            total_queue = sum(ts.get_total_queued() for ts in self.env.traffic_signals.values())
            wait_times = [sum(ts.get_waiting_time_per_lane()) for ts in self.env.traffic_signals.values()]
            all_lane_waits = [t for ts in self.env.traffic_signals.values() for t in ts.get_waiting_time_per_lane()]
            all_lane_queues = [ts.get_lane_queue_lengths() for ts in self.env.traffic_signals.values()]
            flat_queues = [length for d in all_lane_queues for length in d.values()]

            avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
            max_queue = max(flat_queues) if flat_queues else 0

            num_active_vehicles = sum(len(ts.get_controlled_lanes()) for ts in self.env.traffic_signals.values())

            with open(self.log_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    self.step_counter, reward, total_queue, sum(wait_times),
                    avg_wait_time, max_queue, num_active_vehicles
                ])

        self.step_counter += 1
        return obs, reward, done, truncated, info

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def __getattr__(self, name):
        return getattr(self.env, name)
