from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from spider_ant_env import SpiderAntEnv
# Directly use the env class (not through string id)
env = make_vec_env(lambda: SpiderAntEnv(), n_envs=1)

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_spider_tensorboard/")

model.learn(total_timesteps=500_000,progress_bar=True)

model.save("sac_spider_model")
