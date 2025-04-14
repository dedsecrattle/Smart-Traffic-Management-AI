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
from sumo_rl import SumoEnvironment
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from reward_logger import RewardLogger

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""SarsaLambda Single-Intersection with Reward Logging"""
    )
    prs.add_argument("-route", dest="route", type=str, default="../sumo-config/single-intersection/single-intersection.rou.xml")
    prs.add_argument("-a", dest="alpha", type=float, default=0.0001)
    prs.add_argument("-g", dest="gamma", type=float, default=0.95)
    prs.add_argument("-e", dest="epsilon", type=float, default=0.01)
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5)
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50)
    prs.add_argument("-gui", action="store_true", default=False)
    prs.add_argument("-fixed", action="store_true", default=False)
    prs.add_argument("-s", dest="seconds", type=int, default=100000)
    prs.add_argument("-runs", dest="runs", type=int, default=4)
    args = prs.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_csv = f"outputs/single-intersection/sarsa_lambda_{timestamp}"

    env = SumoEnvironment(
        net_file="../sumo-config/single-intersection/single-intersection.net.xml",
        single_agent=True,
        route_file=args.route,
        out_csv_name=out_csv,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
    )

    for run in range(1, args.runs + 1):
        obs, info = env.reset()

        agent = TrueOnlineSarsaLambda(
            env.observation_space,
            env.action_space,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            fourier_order=7,
            lamb=0.95,
        )

        logger = RewardLogger(env.ts_ids, filename=f"{out_csv}_run{run}.csv")  # ✅ RewardLogger logs one agent

        terminated, truncated = False, False
        if args.fixed:
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step({})
        else:
            while not (terminated or truncated):
                action = agent.act(obs)
                next_obs, r, terminated, truncated, info = env.step(action=action)

                agent.learn(state=obs, action=action, reward=r, next_state=next_obs, done=terminated)
                logger.log_step(env, {env.ts_ids[0] :action}, {env.ts_ids[0]: r})

                obs = next_obs

        env.save_csv(out_csv, run)
        logger.save()
        env.close()
