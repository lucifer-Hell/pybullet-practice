import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Choose your env (can also try "Walker2d-v4" or custom)
env = gym.make('Ant-v5')  # Or Walker2d-v4

# Wrap it to work with SB3
env = DummyVecEnv([lambda: env])

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_ant_tensorboard/")

# Train the model (adjust timesteps as needed)
model.learn(total_timesteps=1_000_000)

# Save model
model.save("ppo_ant_walk")
