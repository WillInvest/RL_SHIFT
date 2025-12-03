#!/usr/bin/env python3
import os
import glob
import time
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from make_shift_env import make_env


def _latest_zip(path_like: str) -> str | None:
    """
    If path_like is a dir, return latest *.zip inside.
    If path_like is a file, return it (if exists).
    Else return None.
    """
    p = Path(path_like)
    if p.is_dir():
        zips = sorted(p.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
        return str(zips[0]) if zips else None
    if p.is_file():
        return str(p)
    return None


def _default_path(algo: str) -> str | None:
    """
    Choose a sensible default path per algo.
    PPO: prefer latest in ckpt_ppo/, else models/ppo_shift.zip
    DQN: prefer latest in ckpt_dqn/, else models/dqn_shift.zip
    """
    algo = algo.upper()
    if algo == "PPO":
        return (_latest_zip("ckpt_ppo")
                or _latest_zip("models/ppo_shift.zip"))
    if algo == "DQN":
        return (_latest_zip("ckpt_dqn")
                or _latest_zip("models/dqn_shift.zip"))
    return None


def eval_once(algo: str, path: str | None, seed: int = 777, max_steps: int | None = None):
    algo = algo.upper()
    # Resolve path
    if path:
        resolved = _latest_zip(path)
        if resolved is None:
            raise FileNotFoundError(f"No model found at '{path}'")
        model_path = resolved
    else:
        model_path = _default_path(algo)
        if model_path is None:
            raise FileNotFoundError(
                f"No default model found for {algo}. "
                f"Pass --path or save a model in ckpt_{algo.lower()}/ or models/"
            )

    print(f"[EVAL] algo={algo}  model={model_path}")

    # Create env
    env = make_env(seed=seed)

    # Load model
    if algo == "DQN":
        model = DQN.load(model_path, env=env)  # env passed to bind normalize spaces etc.
    elif algo == "PPO":
        model = PPO.load(model_path, env=env)
    else:
        env.close()
        raise ValueError("algo must be one of: DQN, PPO")

    # Roll one episode
    try:
        obs, info = env.reset()
        total_r, done = 0.0, False
        steps = 0
        print(f"[EVAL {algo}] starting episode…")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total_r += float(r)
            done = bool(term or trunc)
            steps += 1

            # optional cap
            if max_steps is not None and steps >= max_steps:
                done = True

            # light logging every 10 steps
            if getattr(env.unwrapped, "_steps", 0) % 10 == 0:
                mid, spread, net_sh = env.unwrapped._obs_np()
                realPL = info.get("realized_pl", np.nan)
                print(
                    f"[STEP {env.unwrapped._steps:03d}] "
                    f"mid={mid:.4f} spr={spread:.4f} net={net_sh} "
                    f"a={int(action)} r={float(r):.4f} realPL={realPL:.2f}"
                )
            time.sleep(0.01)

        realPL = info.get("realized_pl", np.nan)
        print(f"[EVAL {algo}] total_reward={total_r:.4f} realizedPL={realPL:.2f}")
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["DQN", "PPO"], required=True,
                    help="Which algorithm’s model to evaluate.")
    ap.add_argument("--path", type=str, default=None,
                    help="Path to a .zip model OR a directory of checkpoints. "
                         "If omitted, picks latest from ckpt_{algo}/ or models/..")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Optional cap on steps during evaluation.")
    args = ap.parse_args()

    eval_once(args.algo, args.path, seed=args.seed, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
