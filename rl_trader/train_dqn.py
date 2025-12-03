# train_dqn.py
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from make_shift_env import make_env
from tb_heartbeat import TBHeartbeat

def main():
    seed = 42
    env = make_env(seed=seed)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=20_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        verbose=1,
        tensorboard_log="./tb_dqn/",
        seed=seed,
    )

    ckpt = CheckpointCallback(save_freq=10_000, save_path="./ckpt_dqn/", name_prefix="dqn_shift")
    hb   = TBHeartbeat(every_n_steps=50)
    model.learn(total_timesteps=200_00, callback=[ckpt, hb])

    os.makedirs("models", exist_ok=True)
    model.save("models/dqn_shift.zip")
    env.close()

if __name__ == "__main__":
    main()

