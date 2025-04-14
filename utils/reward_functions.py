import traci
from collections import defaultdict

DEFAULT_WEIGHTS = {
    'vehicles_passed': 2.0,
    'queue': -0.05,
    'waiting_time': -0.01,
    'fuel': -0.1,
    'co2': -0.05,
    'fair_penalty': -10,
    'fair_threshold': 10  
}

def get_lane_vehicles_emissions(lane_ids):
    fuel = co2 = 0.0
    for lane in lane_ids:
        for veh_id in traci.lane.getLastStepVehicleIDs(lane):
            try:
                fuel += traci.vehicle.getFuelConsumption(veh_id)
                co2 += traci.vehicle.getCO2Emission(veh_id)
            except traci.TraCIException:
                continue
    return fuel, co2

def get_queue_and_wait(lane_ids):
    queue = wait = 0.0
    for lane in lane_ids:
        try:
            queue += traci.lane.getLastStepHaltingNumber(lane)
            wait += traci.lane.getWaitingTime(lane)
        except traci.TraCIException:
            continue
    return queue, wait

def reward_combined(env, weights=None, return_stats=False):
    weights = weights or DEFAULT_WEIGHTS
    rewards = {}
    stats = defaultdict(dict)

    for ts_id in env.ts_ids:
        ts = env.traffic_signals[ts_id]
        lanes = ts.lanes

        passed = ts.get_last_step_vehicles_passed()
        queue, wait = get_queue_and_wait(lanes)
        fuel, co2 = get_lane_vehicles_emissions(lanes)

        reward = (
            weights['vehicles_passed'] * passed +
            weights['queue'] * queue +
            weights['waiting_time'] * wait +
            weights['fuel'] * fuel +
            weights['co2'] * co2
        )

        
        if passed == 0 and queue > weights['fair_threshold']:
            reward += weights['fair_penalty']

        rewards[ts_id] = reward

        
        if return_stats:
            stats[ts_id] = {
                'reward': reward,
                'passed': passed,
                'queue': queue,
                'wait': wait,
                'fuel': fuel,
                'co2': co2
            }

    return (rewards, stats) if return_stats else rewards

def reward_emissions_only(env):
    rewards = {}
    for ts_id in env.ts_ids:
        ts = env.traffic_signals[ts_id]
        fuel, co2 = get_lane_vehicles_emissions(ts.lanes)
        rewards[ts_id] = -0.1 * fuel - 0.05 * co2
    return rewards

def reward_throughput_only(env):
    return {ts_id: env.traffic_signals[ts_id].get_last_step_vehicles_passed()
            for ts_id in env.ts_ids}

def reward_no_op(env):
    return {ts_id: 0.0 for ts_id in env.ts_ids}
