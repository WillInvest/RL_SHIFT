# train_ppo.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from make_shift_env import make_env

def main():
    seed = 123
    # PPO prefers a VecEnv, even if single
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.0,
        vf_coef=0.5,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./tb_ppo/",
        seed=seed,
    )

    ckpt = CheckpointCallback(save_freq=10_000, save_path="./ckpt_ppo/", name_prefix="ppo_shift")
    model.learn(total_timesteps=200_00, callback=ckpt)  # 20k to start; scale up later

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_shift.zip")
    env.close()

if __name__ == "__main__":
    main()
