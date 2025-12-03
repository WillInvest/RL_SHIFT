# make_shift_env.py
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from obs_wrapper import ObsTransform
from rl_trader import ShiftEnv   # your env

def make_env(seed: int | None = None):
    # base trading env
    env = ShiftEnv()

    # remember horizon before wrapping
    max_steps = env.max_steps

    # fixed episode length for SB3
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    # transform observations: [mid, spread, net_shares] -> clipped [ret_cents, spread_cents, net_lots]
    env = ObsTransform(env)

    # Monitor adds episode stats so PPO can log ep_rew_mean, ep_len_mean, etc.
    env = Monitor(env)

    # optional seeding
    if seed is not None:
        env.reset(seed=seed)

    return env
