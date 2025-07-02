import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from pathlib import Path
import mujoco

class SpiderAntEnv(MujocoEnv):
    def __init__(self, xml_file="./default_world.xml", frame_skip=5,render_mode="human"):
        xml_path = str(Path(__file__).parent / xml_file)

        # Load model temporarily to extract dimensions
        model = mujoco.MjModel.from_xml_path(xml_path)
        obs_dim = model.nq - 1 + model.nv  # exclude x position
        act_dim = model.nu

        # Define observation space (action_space will be set later)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Call MujocoEnv constructor with required args
        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode
        )

        # Now define action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float64
        )

    def step(self, action):
        xpos_before = self.data.qpos[0].copy()
        self.do_simulation(action, self.frame_skip)
        xpos_after = self.data.qpos[0].copy()

        reward = xpos_after - xpos_before  # forward reward
        observation = self._get_obs()
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],  # skip x
            self.data.qvel.flat,
        ]).astype(np.float64)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
