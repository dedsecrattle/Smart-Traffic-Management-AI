import argparse
import os
import sys
from datetime import datetime

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
from reward_functions import reward_combined
from reward_logger import RewardLogger


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Q-Learning Single-Intersection with Reward Logging"""
    )
    prs.add_argument("-route", dest="route", type=str, default="../sumo-config/single-intersection/single-intersection.rou.xml")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1)
    prs.add_argument("-g", dest="gamma", type=float, default=0.95)
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05)
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005)
    prs.add_argument("-d", dest="decay", type=float, default=0.995)
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10)
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50)
    prs.add_argument("-gui", action="store_true", default=False)
    prs.add_argument("-fixed", action="store_true", default=False)
    prs.add_argument("-ns", dest="ns", type=int, default=42)
    prs.add_argument("-we", dest="we", type=int, default=42)
    prs.add_argument("-s", dest="seconds", type=int, default=100000)
    prs.add_argument("-v", action="store_true", default=False)
    prs.add_argument("-runs", dest="runs", type=int, default=4)
    args = prs.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_csv = f"outputs/single-intersection/q-learning_{timestamp}"

    env = SumoEnvironment(
        net_file="../sumo-config/single-intersection/single-intersection.net.xml",
        route_file=args.route,
        use_gui=args.gui,
        out_csv_name=out_csv,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        reward_fn=reward_combined
    )

    for run in range(1, args.runs + 1):
        initial_states = env.reset()

        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=args.alpha,
                gamma=args.gamma,
                exploration_strategy=EpsilonGreedy(
                    initial_epsilon=args.epsilon,
                    min_epsilon=args.min_epsilon,
                    decay=args.decay
                ),
            )
            for ts in env.ts_ids
        }

        logger = RewardLogger(env.ts_ids, filename=f"{out_csv}_run{run}.csv")  # ✅ Create logger
        done = {"__all__": False}

        if args.fixed:
            while not done["__all__"]:
                _, _, done, _ = env.step({})
        else:
            obs = initial_states
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
                next_obs, rewards, done, _ = env.step(action=actions)

                for agent_id in ql_agents.keys():
                    ql_agents[agent_id].learn(
                        next_state=env.encode(next_obs[agent_id], agent_id),
                        reward=rewards[agent_id]
                    )

                logger.log_step(env, actions, rewards)
                obs = next_obs

        env.save_csv(out_csv, run)
        logger.save()
        env.close()
