import gymnasium as gym
env = gym.make('Ant-v5', render_mode="human")  # Or Walker2d-v4
obs = env.reset()
for _ in range(10_000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()