# tb_heartbeat.py
from stable_baselines3.common.callbacks import BaseCallback

class TBHeartbeat(BaseCallback):
    """
    Logs a few simple scalars regularly so TensorBoard shows data
    even before learning_starts is reached.
    """
    def __init__(self, every_n_steps=50, verbose=0):
        super().__init__(verbose)
        self.every_n_steps = every_n_steps

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every_n_steps == 0:
            # num_timesteps is SB3's global counter
            self.logger.record("debug/num_timesteps", float(self.num_timesteps))
            # You can add more, e.g. inventory if env exposes it:
            try:
                net_sh = float(self.training_env.get_attr("t")[0].get_portfolio_item(
                    self.training_env.get_attr("symbol")[0]).get_shares())
                self.logger.record("debug/net_shares", net_sh)
            except Exception:
                pass
        return True
