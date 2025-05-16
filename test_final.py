import gym
from gym import spaces
import numpy as np
import mujoco
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn
import torch.nn as nn

class SpiderEnv(gym.Env):
    def __init__(self):
        super(SpiderEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path("./default_world.xml")
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 5

        # Action: 4 joint positions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: [x, y, z] + [vx, vy, vz] + [4 actuator values] = 10D
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        pos = self.data.sensordata[0:3]     # x, y, z from <framepos>
        vel = self.data.sensordata[3:6]     # vx, vy, vz from <velocimeter>
        controls = self.data.ctrl[:]        # 4 actuator values
        obs = np.concatenate([pos, vel, controls])
        return obs

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        z_pos = self.data.sensordata[2]
        vx = self.data.sensordata[3]
        vy = self.data.sensordata[4]

        forward_reward = 5.0 * vx if vx > 0 else 2.0 * vx
        penalty_sideways = -0.1 * abs(vy)
        death_penalty = 0.0 if z_pos > 0.2 else -5.0
        control_penalty = -0.001 * np.sum(np.square(action))
        velocity_bonus = 5.0 if vx > 1.0 else 0.0


        reward = forward_reward + control_penalty + penalty_sideways + death_penalty + velocity_bonus
        done = z_pos < 0.2

        return self._get_obs(), reward, done, {}

# === Training ===
from stable_baselines3 import PPO

env = SpiderEnv()

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path='./checkpoints/',
    name_prefix='ppo_spider'
)

model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=get_schedule_fn(0.0001),
    ent_coef=0.0002,
    verbose=1,
    policy_kwargs=dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        log_std_init=0,
        ortho_init=False
    )
)

model.learn(total_timesteps=10_00_000, callback=checkpoint_callback)
model.save("ppo_spider_walk")
