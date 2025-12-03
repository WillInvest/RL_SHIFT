# obs_wrapper.py
import numpy as np
import gymnasium as gym

class ObsTransform(gym.ObservationWrapper):
    """
    Map [mid, spread, net_shares] -> [ret_cents, spread_cents, net_lots]
    with clipping to small ranges so SB3 learns faster.
    """
    def __init__(self, env, clip_ret=5.0, clip_spread=10.0, clip_lots=5.0):
        super().__init__(env)
        self._prev_mid = None
        low  = np.array([-clip_ret, 0.0, -clip_lots], dtype=np.float32)
        high = np.array([ clip_ret, clip_spread,  clip_lots], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        mid, spread, net_sh = obs
        ret_cents = 0.0 if self._prev_mid is None else (mid - self._prev_mid) * 100.0
        spread_c  = spread * 100.0
        net_lots  = float(net_sh) / 100.0
        x = np.array([ret_cents, spread_c, net_lots], dtype=np.float32)
        x = np.clip(x, self.observation_space.low, self.observation_space.high)
        if mid > 0.0:
            self._prev_mid = mid
        return x

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_mid = None
        return self.observation(obs), info
