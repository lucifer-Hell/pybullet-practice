import gymnasium as gym
from stable_baselines3 import PPO
import time


# Load the trained model
model = PPO.load("ppo_ant_walk")

# Create the environment (with rendering)
env = gym.make("Ant-v5", render_mode="human",camera_id=0)  # or any other Mujoco env
obs, _ = env.reset()
viewer = env.unwrapped.mujoco_renderer.viewer
# Set camera to follow the torso
viewer.cam.trackbodyid = 1
viewer.cam.distance = 5  
print('test ',env )
# # Access the underlying MujocoEnv to get the viewer
# mujoco_env = env.unwrapped

# # Now control the camera
# mujoco_env.viewer.cam.trackbodyid = 1  # Follow the robot
# mujoco_env.viewer.cam.distance = 5     # Optional: zoom out

while True:
    # Predict the next action using the trained model
    action, _states = model.predict(obs, deterministic=True)
    
    # Take the action
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.1)  # Pause for 0.1 seconds after each step

    
    # If done, reset the environment
    if terminated or truncated:
        print('terminated or truncated')
        obs = env.reset()
        # # Now control the camera
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5        
