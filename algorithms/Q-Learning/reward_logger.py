import pandas as pd
import traci

class RewardLogger:
    def __init__(self, ts_ids, filename="log.csv"):
        self.ts_ids = ts_ids
        self.filename = filename
        self.logs = []

    def log_step(self, env, actions, rewards):
        for ts_id in self.ts_ids:
            ts = env.traffic_signals[ts_id]

            queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in ts.lanes)
            wait = sum(traci.lane.getWaitingTime(l) for l in ts.lanes)
            fuel = co2 = 0.0

            for lane in ts.lanes:
                for veh in traci.lane.getLastStepVehicleIDs(lane):
                    try:
                        fuel += traci.vehicle.getFuelConsumption(veh)
                        co2 += traci.vehicle.getCO2Emission(veh)
                    except traci.TraCIException:
                        continue

            # Use queue as inverse proxy for throughput
            vehicles_passed = max(0, 10 - queue)

            self.logs.append({
                "step": env.sim_step,
                "action": actions.get(ts_id),
                "reward": rewards.get(ts_id),
                "queue_length": queue,
                "waiting_time": wait,
                "vehicles_passed": vehicles_passed,
                "fuel_consumption": fuel,
                "co2_emission": co2,
            })

    def save(self, suffix=""):
        output_path = self.filename.replace(".csv", f"{suffix}.csv") if suffix else self.filename
        df = pd.DataFrame(self.logs)
        df.to_csv(output_path, index=False)
        print(f"[RewardLogger] Log saved to {output_path}")
