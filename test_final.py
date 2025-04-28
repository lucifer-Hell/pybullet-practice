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
        self.frame_skip=5
        
        # Action: 4 joint positions, each between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Observation: position (x, y, z) + actuator targets maybe
        self.observation_space =spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.prev_x = None
        self.prev_y= None


    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.prev_x = self.data.sensordata[0]  # x position
        self.prev_y = self.data.sensordata[1]  # x position
        return self._get_obs()

    def _get_obs(self):
        pos = self.data.sensordata[:3]
        controls = self.data.ctrl[:]    # (4 control values)
        obs = np.concatenate([pos, controls])
        return obs

    # def step(self, action):
    #     # Send action to robot
    #     self.data.ctrl[:] = action

    #     # Step the simulation forward
    #     mujoco.mj_step(self.model, self.data)

    #     # Compute reward: how much forward we moved
    #     x_pos = self.data.sensordata[0]
    #     y_pos= self.data.sensordata[1]
    #     z_pos= self.data.sensordata[2]
    #     # reward = x_pos - self.prev_x
    #     forward_reward = x_pos - self.prev_x
    #     alive_bonus = 0.5 if z_pos > 0.5 else -1.0  # Encourage staying above certain height
    #     control_penalty = -0.001 * np.sum(np.square(action))  # Don't use huge actuator commands
    #     reward = forward_reward + alive_bonus + control_penalty

    #     self.prev_x = x_pos

    #     # Done condition: you can customize
    #     done = False
    #     if self.data.sensordata[2] < 0.2:  # if z position falls too low
    #         done = True

    #     return self._get_obs(), reward, done, {}

    # def step(self, action):
    #     # Send action to robot
    #     self.data.ctrl[:] = action

    #     # Step the simulation forward
    #     mujoco.mj_step(self.model, self.data)

    #     # Current position
    #     x_pos = self.data.sensordata[0]
    #     y_pos = self.data.sensordata[1]
    #     z_pos = self.data.sensordata[2]

    #     # Approximate velocity in X direction
    #     vel_x = (x_pos - self.prev_x) / self.model.opt.timestep  # timestep=0.01 from your XML

    #     # Reward terms
    #     forward_reward = vel_x                      # reward based on speed
    #     alive_bonus = 0.5 if z_pos > 0.4 else -1.0   # positive reward if upright
    #     control_penalty = -0.001 * np.sum(np.square(action))  # small penalty for big actuator commands

    #     # Total reward
    #     reward = 10*forward_reward + alive_bonus + control_penalty

    #     # Update previous x for next velocity calculation
    #     self.prev_x = x_pos

    #     # Done condition (robot has fallen down)
    #     done = False
    #     if z_pos < 0.2:  # Fell on the ground
    #         done = True

    #     return self._get_obs(), reward, done, {}


    #     def render(self, mode='human'):
    #         pass  # Later if you want viewer

    #     def close(self):
    #         pass


    def step(self, action):
        self.data.ctrl[:] = action  # Direct raw action to actuator
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        x_pos = self.data.sensordata[0]
        y_pos = self.data.sensordata[1]  # y position
        z_pos = self.data.sensordata[2]

        # Compute how much we moved
        delta_x = (x_pos - self.prev_x)
        delta_y= (y_pos - self.prev_y)

        dt = self.model.opt.timestep * self.frame_skip

        self.prev_x = x_pos
        self.prev_y = y_pos

        # big_action_reward = 10 * np.sum(np.clip(np.abs(action), 0.2, 1.0))
        vel =  delta_x/dt
        forward_reward = 10.0 * vel if vel > 0 else 5 *vel 
        penalty_sideways = -5.0 * np.abs(delta_y)

        death_penalty = 0.0 if z_pos > 0.2 else -5.0
        control_penalty = -0.0001 * np.sum(np.square(action))
        # slow_penalty = -8 if vel_x < 2 else 0
        slow_penalty = 0
        reward = forward_reward + death_penalty  + slow_penalty + control_penalty + penalty_sideways
        done = z_pos < 0.2

        return self._get_obs(), reward, done, {}

from stable_baselines3 import PPO

env = SpiderEnv()

# Save a checkpoint every 100000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # Save every 100,000 environment steps
    save_path='./checkpoints/',  # Folder where models are saved
    name_prefix='ppo_spider'  # Naming prefix
)

model = PPO("MlpPolicy", 
            env, 
            # learning_rate=2.0633e-05,
            learning_rate=get_schedule_fn(0.0001),
            ent_coef=0.002,
            verbose=1,
            policy_kwargs=dict(
                activation_fn=nn.ReLU,
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                log_std_init=0,
                ortho_init=False
             ))
model.learn(total_timesteps=10_00_000)

model.save("ppo_spider_walk")
