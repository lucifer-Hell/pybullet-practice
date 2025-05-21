import os
import re
from gymnasium import Env, spaces
import numpy as np
import mujoco
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
import torch.nn as nn


class SpiderEnv(Env):
    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path("./default_world.xml")
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 5
        self.training = True

        self.seq_count = 20
        self.action_timer = 0
        self.current_action = np.zeros(4, dtype=np.float32)  # Holds action between updates


        # Action: 4 joints
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: [x, y, z] + [vx, vy, vz] + [wx, wy, wz] + [4 actuator values]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        obs = self._get_obs()
        self.current_action = np.zeros(4, dtype=np.float32)
        self.action_timer = 0
        return obs, {}  # required by Gymnasium

    def _get_obs(self):
        pos = self.data.sensordata[0:3]   # x, y, z
        vel = self.data.sensordata[3:6]   # vx, vy, vz
        gyro = self.data.sensordata[6:9]  # wx, wy, wz
        controls = self.data.ctrl[:]      # actuator inputs
        return np.concatenate([pos, vel, gyro, controls])

    # def step(self, action):
    #     if self.action_timer == 0:
    #         self.current_action = action.copy()

    #     self.data.ctrl[:] =  self.current_action
    #     self.action_timer = (self.action_timer + 1) % self.seq_count
    #     x_pos_before=self.data.sensordata[0]
    #     y_pos_before=self.data.sensordata[1]
    #     z_pos_before=self.data.sensordata[2]

    #     for _ in range(self.frame_skip):
    #         mujoco.mj_step(self.model, self.data)
    #     x_pos_after=self.data.sensordata[0]
    #     y_pos_after=self.data.sensordata[1]
    #     z_pos_after=self.data.sensordata[2]

    #     # Extract values
    #     x_delta= x_pos_after-x_pos_before
    #     y_delta= y_pos_after-y_pos_before
    #     z_delta= z_pos_after-z_pos_before

    #     gyro = self.data.sensordata[6:9]

    #     forward_reward= x_delta / (self.frame_skip*0.01)
    #     control_penalty = -0.01 * np.sum(np.square(action))
    #     gyro_penalty = -0.05 * np.sum(np.square(gyro))

    #     z_abs = self.data.sensordata[2]  # current Z
    #     alive_bonus = 1.0 if z_abs > 0.2 else -1.0
    #     jump_penalty = -2.0 * max(0.0, z_abs - 0.6)


    #     reward = (
    #         forward_reward+
    #         alive_bonus +
    #         control_penalty +
    #         gyro_penalty +
    #         jump_penalty
    #     )

    #     terminated = z_abs < 0.2  # Fell down
    #     truncated = False         # No time limit set

    #     obs = self._get_obs()
    #     return obs, reward, terminated, truncated, {}  # Gymnasium 5-tuple

    def step(self, action):
        if self.action_timer == 0:
            self.current_action = action.copy()

        self.data.ctrl[:] = self.current_action
        self.action_timer = (self.action_timer + 1) % self.seq_count

        x_pos_before = self.data.sensordata[0]
        z_pos_before = self.data.sensordata[2]

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        x_pos_after = self.data.sensordata[0]
        z_pos_after = self.data.sensordata[2]
        x_delta = int(np.floor(x_pos_after)) - int(np.floor(x_pos_before))

        # Raw reward components
        forward_reward = x_delta / (self.frame_skip * 0.01) if x_delta>0 else -0.5
        control_penalty = -0.01 * np.sum(np.square(action))
        # gyro = self.data.sensordata[6:9]
        # gyro_penalty = -0.05 * np.sum(np.square(gyro))
        z_abs = z_pos_after
        alive_bonus = 0.0 if z_abs > 0.2 else -1.0
        jump_penalty = -1.0 * max(0.0, z_abs - 0.6)

        # Final reward
        reward = (
            forward_reward +
            alive_bonus +
            control_penalty +
            jump_penalty
        )

        terminated = z_abs < 0.2
        truncated = False
        obs = self._get_obs()

        return obs, reward, terminated, truncated, {}



def make_env():
    def _init():
        env = SpiderEnv()
        return env
    return _init



def get_latest_checkpoint(checkpoint_dir, name_prefix):
    max_step = -1
    latest_path = None
    pattern = re.compile(rf"{re.escape(name_prefix)}_(\d+)_steps\.zip")

    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                latest_path = os.path.join(checkpoint_dir, fname)

    return latest_path


# === PPO Training ===
if __name__ == "__main__":
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.utils import get_schedule_fn

    num_envs = 1  # Or 8 or 16 depending on your CPU

    env = SubprocVecEnv([make_env() for _ in range(num_envs)])


    # env = SpiderEnv()

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=50_00_000,
    #     save_path='./checkpoints/',
    #     name_prefix='ppo_spider'
    # )

    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     device="cpu",
    #     learning_rate=0.0003,
    #     n_steps=2048,
    #     batch_size=256,
    #     ent_coef=0.01,
    #     verbose=1,
    #     policy_kwargs=dict(
    #         activation_fn=nn.ReLU,
    #         net_arch=dict(pi=[256, 256], vf=[256, 256]),
    #         log_std_init=-1.0,
    #         ortho_init=True
    #     )
    # )

    checkpoint_dir = "./checkpoints/"
    checkpoint_prefix = "ppo_spider_latest"

    
    checkpoint_callback = CheckpointCallback(
    save_freq=10_00_000,
    save_path=checkpoint_dir,
    name_prefix=checkpoint_prefix,
    save_replay_buffer=False,
    save_vecnormalize=False
    )

    latest_ckpt = get_latest_checkpoint(checkpoint_dir, checkpoint_prefix)

    if latest_ckpt:
        print(f"[✓] Loading checkpoint: {latest_ckpt}")
        model = PPO.load(latest_ckpt, env=env, device="cpu")
    else:
        print("[+] No checkpoint found, starting from scratch...")
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=256,
            ent_coef=0.01,
            verbose=1,
            policy_kwargs=dict(
                activation_fn=nn.ReLU,
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                log_std_init=-1.0,
                ortho_init=True
            )
        )

    # model.learn(total_timesteps=100_000_000, callback=checkpoint_callback)
    model.learn(total_timesteps=1000_00_000, callback=checkpoint_callback)

    model.save("ppo_spider_walk")
