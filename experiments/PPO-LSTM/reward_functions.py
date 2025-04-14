import traci

# ========================
# Configurable Weights
# ========================
DEFAULT_WEIGHTS = {
    'vehicles_passed': 2.0,
    'queue': -0.05,
    'waiting_time': -0.01,
    'fuel': -0.1,
    'co2': -0.05,
    'fair_penalty': -10,
    'fair_threshold': 10
}

# ========================
# Utility Functions
# ========================

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

# ========================
# Per-Agent Reward Functions (used by sumo-rl)
# ========================

def reward_combined(ts, weights=None):
    """
    Reward based on throughput, emissions, queue length, and waiting time.
    """
    weights = weights or DEFAULT_WEIGHTS

    queue, wait = get_queue_and_wait(ts.lanes)
    fuel, co2 = get_lane_vehicles_emissions(ts.lanes)

    # Approximate "vehicles passed" as 10 - queue length
    passed = max(0, 10 - queue)

    reward = (
        weights['vehicles_passed'] * passed +
        weights['queue'] * queue +
        weights['waiting_time'] * wait +
        weights['fuel'] * fuel +
        weights['co2'] * co2
    )

    if passed == 0 and queue > weights['fair_threshold']:
        reward += weights['fair_penalty']

    return reward

def reward_emissions_only(ts):
    fuel, co2 = get_lane_vehicles_emissions(ts.lanes)
    return -0.1 * fuel - 0.05 * co2

def reward_throughput_only(ts):
    queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in ts.lanes)
    return max(0, 10 - queue)

def reward_no_op(ts):
    return 0.0

# ========================
# Optional Multi-agent Wrappers
# ========================

def multiagent_wrapper(per_agent_fn):
    """
    Converts per-agent reward into multi-agent dict.
    """
    def wrapped(env):
        return {
            ts_id: per_agent_fn(env.traffic_signals[ts_id])
            for ts_id in env.ts_ids
        }
    return wrapped
