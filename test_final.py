import gym
from gym import spaces
import numpy as np
import mujoco
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
import torch.nn as nn

class SpiderEnv(gym.Env):
    def __init__(self):
        super(SpiderEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path("./default_world.xml")
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 5
        self.training = True  # enable logging while training

        # Action space: 4 joint positions (scaled between -1 and 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: [x, y, z] + [vx, vy, vz] + [4 actuator values] = 10D
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        pos = self.data.sensordata[0:3]     # x, y, z (from framepos)
        vel = self.data.sensordata[3:6]     # vx, vy, vz (from velocimeter)
        controls = self.data.ctrl[:]        # actuator values
        obs = np.concatenate([pos, vel, controls])
        return obs

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Read relevant values
        z_pos = self.data.sensordata[2]
        vx = self.data.sensordata[3]

        # Reward components
        forward_reward = vx                           # move forward
        alive_bonus = 1.0 if z_pos > 0.2 else -10.0    # stay upright
        control_penalty = -0.01 * np.sum(np.square(action))  # avoid excessive motor use

        reward = forward_reward + alive_bonus + control_penalty
        done = z_pos < 0.2

        # # Optional: print reward & velocity for debugging
        # if self.training:
        #     print(f"vx: {vx:.3f}, reward: {reward:.2f}, z: {z_pos:.2f}")

        return self._get_obs(), reward, done, {}

# === PPO Training ===
env = SpiderEnv()

# Save checkpoints during training
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path='./checkpoints/',
    name_prefix='ppo_spider'
)

# Define PPO model with optimized policy
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=256,
    ent_coef=0.0,
    verbose=1,
    policy_kwargs=dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-1.0,
        ortho_init=True
    )
)

# Train the model
model.learn(total_timesteps=10_00_000, callback=checkpoint_callback)
model.save("ppo_spider_walk")
