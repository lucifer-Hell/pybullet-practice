from gymnasium import Env, spaces
import numpy as np
import mujoco
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
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
        self.current_action = np.zeros(4, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_action = np.zeros(4, dtype=np.float32)
        self.action_timer = 0
        return self._get_obs(), {}

    def _get_obs(self):
        pos = self.data.sensordata[0:3]
        vel = self.data.sensordata[3:6]
        gyro = self.data.sensordata[6:9]
        controls = self.data.ctrl[:]
        return np.concatenate([pos, vel, gyro, controls])

    def step(self, action):
        if self.action_timer == 0:
            self.current_action = action.copy()

        self.data.ctrl[:] = self.current_action
        self.action_timer = (self.action_timer + 1) % self.seq_count

        x_before = self.data.sensordata[0]
        y_before = self.data.sensordata[1]
        z_before = self.data.sensordata[2]

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        x_after = self.data.sensordata[0]
        y_after = self.data.sensordata[1]
        z_after = self.data.sensordata[2]

        x_delta = x_after - x_before
        y_delta = y_after - y_before
        z_abs = z_after
        gyro = self.data.sensordata[6:9]

        path_reward = np.clip(x_delta * 50, -5, 5)
        y_drift_penalty = -1.0 * abs(y_delta)
        control_penalty = -0.01 * np.sum(np.square(action))
        gyro_penalty = -0.05 * np.sum(np.square(gyro))
        alive_bonus = 1.0 if z_abs > 0.2 else -10.0
        jump_penalty = -2.0 * max(0.0, z_abs - 0.6)

        reward = path_reward + alive_bonus + control_penalty + gyro_penalty + jump_penalty + y_drift_penalty

        terminated = z_abs < 0.2
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}


def make_env():
    def _init():
        return SpiderEnv()
    return _init


if __name__ == "__main__":
    num_envs=6
    env = DummyVecEnv([make_env() for _ in range(num_envs)])

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path='./checkpoints/',
        name_prefix='td3_spider',
    )

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=256,
        train_freq=(1, "step"),
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=nn.ReLU
        ),
        verbose=1
    )

    model.learn(total_timesteps=10_00_000, callback=checkpoint_callback)
    model.save("td3_spider_walk")