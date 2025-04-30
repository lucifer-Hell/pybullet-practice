import time
import mujoco
import mujoco.viewer
import mediapy as media
import numpy as np
from stable_baselines3 import PPO

model = mujoco.MjModel.from_xml_path("./default_world.xml")  # or "robot.xml"

data = mujoco.MjData(model)


# Load your trained PPO model
ppo_model = PPO.load("ppo_spider_walk")   # <<<<<< Load your trained PPO model here


# Launch viewer
viewer = mujoco.viewer.launch_passive(model,data)

# How many actuators you have?
n_actuators = model.nu  # nu = number of actuators

# Number of sensors
n_sensors = model.nsensor
print(f"Number of sensors: {n_sensors}")

# Size of sensor data
print(f"Size of sensordata array: {data.sensordata.shape}")

frame_skip=5

print(f"Number of actuators: {n_actuators}")  # Should print 4 in your case
pick_first=False
count = 2
# Simulation loop
while viewer.is_running():
    # Prepare current observation
    # Prepare current observation
    pos = data.sensordata[:3]       # (x, y, z)
    ctrl = data.ctrl[:]             # (4 control values)
    # Concatenate position + control
    obs = np.concatenate([pos, ctrl])

    obs = np.array(obs, dtype=np.float32)
    
    # Add batch dimension (PPO expects [batch_size, obs_dim])
    obs = obs.reshape(1, -1)

    # Get action from trained PPO model
    action, _ = ppo_model.predict(obs)
    # if count<=0:
    #     # Apply action
    data.ctrl[:] = action.flatten()
        # count=2
    # count=count-1
    # Step simulation
    for _ in range (frame_skip):
        mujoco.mj_step(model,data)
    viewer.sync()
    # # Optional: print actuator control values
    print(f"Controls: {data.ctrl}")

    # Read sensor values
    spider_position = data.sensordata[:3]  # Since framepos has 3 values: x, y, z position
    print(f"Spider position: {spider_position}")

    time.sleep(0.01)

viewer.close()
