import argparse
import os
import sys
from datetime import datetime

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Multi-Agent SARSA(λ) for SUMO"""
    )
    prs.add_argument("-route", dest="route", type=str,
                     default="../sumo-config/2x2grid/2x2.rou.xml")
    prs.add_argument("-a", dest="alpha", type=float, default=0.001)
    prs.add_argument("-g", dest="gamma", type=float, default=0.95)
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05)
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10)
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50)
    prs.add_argument("-gui", action="store_true", default=False)
    prs.add_argument("-fixed", action="store_true", default=False)
    prs.add_argument("-s", dest="seconds", type=int, default=100000)
    prs.add_argument("-runs", dest="runs", type=int, default=4)
    args = prs.parse_args()

    out_csv = "outputs/2x2/multi_sarsa_lambda"

    env = SumoEnvironment(
        net_file="../sumo-config/2x2grid/2x2.net.xml",
        route_file=args.route,
        out_csv_name=out_csv,
        single_agent=False,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
    )

    for run in range(1, args.runs + 1):
        obs = env.reset()

        sarsa_agents = {
            ts: TrueOnlineSarsaLambda(
                env.observation_spaces(ts),
                env.action_spaces(ts),
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon=args.epsilon,
                fourier_order=7,
                lamb=0.95,
            )
            for ts in env.ts_ids
        }

        done = {"__all__": False}

        if args.fixed:
            while not done["__all__"]:
                _, _, done, _ = env.step({})
        else:
            while not done["__all__"]:
                actions = {ts: sarsa_agents[ts].act(obs[ts]) for ts in sarsa_agents}
                next_obs, rewards, done, infos = env.step(actions)

                for ts in sarsa_agents:
                    sarsa_agents[ts].learn(
                        state=obs[ts],
                        action=actions[ts],
                        reward=rewards[ts],
                        next_state=next_obs[ts],
                        done=done.get(ts, False)
                    )

                obs = next_obs

        env.save_csv(out_csv, run)
        env.close()
