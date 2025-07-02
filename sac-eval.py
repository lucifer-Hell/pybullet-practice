import gymnasium as gym
from spider_ant_env import SpiderAntEnv
from stable_baselines3 import SAC

env = SpiderAntEnv()
model = SAC.load("sac_spider_model")

obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
