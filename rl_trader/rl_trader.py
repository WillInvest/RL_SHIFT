#!/usr/bin/env python3
# pip install gymnasium numpy
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import shift

# -----------------------------------
USER   = "test001"
CFG    = "initiator.cfg"
PASS   = "password"
SYMBOL = "CS1"

MAX_STEPS      = 250
STEP_SLEEP     = 0.25 # inc
LOTS           = 1           # 1 lot = 100 shares
INV_PENALTY    = 0.0001      # reward penalty * |net_shares|
SPREAD_MAX     = 0.03        # only trade if spread <= 5¢
INV_CAP_LOTS   = 2           # cap |net| <= 2 lots (±200 sh)
COOLDOWN_TRADE = 1.0         # extra pause after trades
WAIT_FIRST_Q   = 10.0        # seconds to wait for first quote
# -----------------------------------

# Actions:
# 0 = HOLD
# 1 = MARKET BUY
# 2 = MARKET SELL
# 3 = LIMIT BUY  at best bid
# 4 = LIMIT SELL at best ask
# 5 = CANCEL ALL PENDING ORDERS
A_HOLD, A_MB, A_MS, A_LB, A_LS, A_CANCEL_ALL = range(6)





def _wait_market_advance(t, max_wait_s=300, poll=1.0):
    """Wait until get_last_trade_time() advances, up to max_wait_s."""
    t0 = t.get_last_trade_time()
    start = time.time()
    while time.time() - start < max_wait_s:
        time.sleep(poll)
        if t.get_last_trade_time() != t0:
            return True
    return False


def _wait_first_quote(t, symbol, timeout=WAIT_FIRST_Q):
    end = time.time() + timeout
    while time.time() < end:
        bp = t.get_best_price(symbol)
        b, a = bp.get_bid_price(), bp.get_ask_price()
        if b > 0 and a > 0 and a >= b:
            return True
        time.sleep(0.2)
    return False


def _obs_tuple(t, symbol):
    """Return (mid, spread, net_shares). mid/spread=0 if quote invalid."""
    bp = t.get_best_price(symbol)
    b, a = bp.get_bid_price(), bp.get_ask_price()
    valid = (b > 0 and a > 0 and a >= b)
    spread = (a - b) if valid else 0.0
    mid    = (a + b) / 2.0 if valid else 0.0
    net_sh = t.get_portfolio_item(symbol).get_shares()
    return mid, spread, net_sh


def _flatten(t, symbol):
    """Cancel resting orders and close any position at market."""
    t.cancel_all_pending_orders()
    it = t.get_portfolio_item(symbol)
    long_lots  = it.get_long_shares()  // 100
    short_lots = it.get_short_shares() // 100
    if long_lots > 0:
        t.submit_order(shift.Order(shift.Order.Type.MARKET_SELL, symbol, int(long_lots)))
    if short_lots > 0:
        t.submit_order(shift.Order(shift.Order.Type.MARKET_BUY,  symbol, int(short_lots)))
    time.sleep(1.0)


class ShiftEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around SHIFT.
    Observation: np.array([mid, spread, net_shares], dtype=float32)
    Action space: Discrete(6) =
        {HOLD,
         MARKET_BUY, MARKET_SELL,
         LIMIT_BUY@bid, LIMIT_SELL@ask,
         CANCEL_ALL_PENDING_ORDERS}
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 user=USER, cfg=CFG, password=PASS, symbol=SYMBOL,
                 max_steps=MAX_STEPS):
        super().__init__()
        self.user = user
        self.cfg = cfg
        self.password = password
        self.symbol = symbol
        self.max_steps = max_steps

        # Gym spaces
        self.action_space = spaces.Discrete(6)
        # Conservative bounds; net_shares could vary with sim, treat wide
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1e6], dtype=np.float32),
            high=np.array([1e6, 1e3,  1e6], dtype=np.float32),
            dtype=np.float32
        )

        self.t = shift.Trader(self.user)
        self._prev_total = None
        self._steps = 0
        self._np_random = None  # set in reset(seed)

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._np_random, _ = gym.utils.seeding.np_random(seed)

        self._connect_if_needed()

        # Instead of raising, patiently wait for replay to move
        if not _wait_market_advance(self.t, max_wait_s=300, poll=1.0):
            print("[ENV] market paused; waiting longer until it resumes…")
            # keep waiting in chunks; avoids crashing SB3
            while not _wait_market_advance(self.t, max_wait_s=300, poll=1.0):
                pass

        if not _wait_first_quote(self.t, self.symbol, timeout=WAIT_FIRST_Q):
            # same idea: keep waiting until we see a valid quote
            print("[ENV] no quote yet; waiting…")
            while not _wait_first_quote(self.t, self.symbol, timeout=WAIT_FIRST_Q):
                pass

        _flatten(self.t, self.symbol)
        self._prev_total = None
        self._steps = 0
        obs = self._obs_np()
        return obs, {}



    def _connect_if_needed(self):
        """Idempotent connect + subscribe."""
        if not self.t.is_connected():
            self.t.connect(self.cfg, self.password)
            print("[ENV] connected:", self.t.is_connected())
            # small settle time after connect
            time.sleep(0.2)
        # subscribe may be idempotent; if not, catch/ignore dup
        try:
            self.t.sub_order_book(self.symbol)
        except Exception:
            pass
        time.sleep(0.3)  # give stream a moment

    def step(self, action):
        # Current obs (mainly to check guardrails)
        mid, spread, net_sh = _obs_tuple(self.t, self.symbol)

        # ---- guardrails ----
        # If spread is invalid or too wide, force HOLD for all trading actions,
        # but still allow explicit CANCEL_ALL.
        if spread == 0.0 or spread > SPREAD_MAX:
            if action != A_CANCEL_ALL:
                action = A_HOLD
        else:
            #If the agent is already at or past the inventory limit, 
            # block any action that would increase risk in that direction.
            # Inventory caps for any action that increases absolute position
            if net_sh >= 100 * INV_CAP_LOTS and action in (A_MB, A_LB):
                action = A_HOLD
            elif net_sh <= -100 * INV_CAP_LOTS and action in (A_MS, A_LS):
                action = A_HOLD

        # ---- execute action ----
        if action == A_MB:
            # Market buy LOTS
            self.t.submit_order(
                shift.Order(shift.Order.Type.MARKET_BUY, self.symbol, LOTS)
            )
            time.sleep(COOLDOWN_TRADE)

        elif action == A_MS:
            # Market sell LOTS
            self.t.submit_order(
                shift.Order(shift.Order.Type.MARKET_SELL, self.symbol, LOTS)
            )
            time.sleep(COOLDOWN_TRADE)

        elif action == A_LB:
            # Limit buy at current best bid
            bp = self.t.get_best_price(self.symbol)
            bid = bp.get_bid_price()
            if bid > 0:
                self.t.submit_order(
                    shift.Order(shift.Order.Type.LIMIT_BUY, self.symbol, LOTS, bid)
                )
                time.sleep(COOLDOWN_TRADE)

        elif action == A_LS:
            # Limit sell at current best ask
            bp = self.t.get_best_price(self.symbol)
            ask = bp.get_ask_price()
            if ask > 0:
                self.t.submit_order(
                    shift.Order(shift.Order.Type.LIMIT_SELL, self.symbol, LOTS, ask)
                )
                time.sleep(COOLDOWN_TRADE)

        elif action == A_CANCEL_ALL:
            # Explicit cancel-all action
            self.t.cancel_all_pending_orders()
            time.sleep(COOLDOWN_TRADE)

        # allow sim to advance
        time.sleep(STEP_SLEEP)

        # compute reward: Δ(realized+unrealized) - λ|inventory|
        real_all = self.t.get_portfolio_summary().get_total_realized_pl()
        unreal   = self.t.get_unrealized_pl(self.symbol)
        total    = real_all + unreal
        inv_abs  = abs(self.t.get_portfolio_item(self.symbol).get_shares())

        if self._prev_total is None:
            reward = 0.0
        else:
            reward = float((total - self._prev_total) - INV_PENALTY * inv_abs)
        self._prev_total = total

        obs = self._obs_np()

        self._steps += 1
        terminated = (self._steps >= self.max_steps)
        truncated = False
        info = {
            "action": int(action),
            "realized_pl": real_all,
            "unrealized_pl": unreal
        }

        if terminated or truncated:
            _flatten(self.t, self.symbol)

        return obs, reward, terminated, truncated, info

    def render(self):
        return  # no-op (optional to implement)

    def close(self):
        try:
            _flatten(self.t, self.symbol)
        finally:
            if self.t.is_connected():
                self.t.disconnect()
            print("[ENV] disconnected.")

    # ---------- helpers ----------
    def _obs_np(self):
        mid, spread, net_sh = _obs_tuple(self.t, self.symbol)
        return np.array([mid, spread, net_sh], dtype=np.float32)


# ---------- quick demo ----------
def run_demo_episode():
    env = ShiftEnv()
    try:
        obs, info = env.reset()
        total_r = 0.0
        print("[DEMO] starting episode…")

        prev_mid = None  # track last valid mid

        done = False
        while not done:
            mid, spread, net_sh = obs

            # --- tiny momentum policy with guardrails ---
            # only act when we have a prior mid and decent spread
            if prev_mid is not None and spread > 0.0 and spread <= 0.05 and abs(mid - prev_mid) >= 0.01:
                if mid > prev_mid:
                    action = A_MB
                elif mid < prev_mid:
                    action = A_MS
                else:
                    action = A_HOLD
            else:
                # bias to HOLD when signal is weak or spread is wide
                # (we only sample among HOLD, MARKET_BUY, MARKET_SELL here)
                action = random.choices(
                    [A_HOLD, A_MB, A_MS],
                    weights=[0.7, 0.15, 0.15]
                )[0]

            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated

            # update prev_mid only when quote looks valid
            if mid > 0.0 and spread > 0.0:
                prev_mid = mid

            if env._steps % 10 == 0:
                mid, spread, net_sh = obs
                print(f"[STEP {env._steps:03d}] mid={mid:.4f} spread={spread:.4f} "
                      f"net={net_sh} realPL={info['realized_pl']:.2f} a={info['action']} r={r:.4f}")

        print(f"[DEMO] done. total_reward={total_r:.4f}  realizedPL={info['realized_pl']:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    run_demo_episode()
