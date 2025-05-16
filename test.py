import time
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

# Load MuJoCo model and data
model = mujoco.MjModel.from_xml_path("./default_world.xml")
data = mujoco.MjData(model)

# Load trained PPO model
ppo_model = PPO.load("ppo_spider_walk")

# Launch passive viewer
viewer = mujoco.viewer.launch_passive(model, data)

# Debug info
print(f"Number of sensors: {model.nsensor}")
print(f"Size of sensordata array: {data.sensordata.shape}")
print(f"Number of actuators: {model.nu}")

frame_skip = 5

# Simulation loop
while viewer.is_running():
    # Observation: [x, y, z] + [vx, vy, vz] + [ctrl0, ctrl1, ctrl2, ctrl3] → 10D
    pos = data.sensordata[0:3]        # x, y, z from <framepos>
    vel = data.sensordata[3:6]        # vx, vy, vz from <velocimeter>
    ctrl = data.ctrl[:]               # current actuator control

    obs = np.concatenate([pos, vel, ctrl]).astype(np.float32).reshape(1, -1)

    # Predict action using PPO model
    action, _ = ppo_model.predict(obs)
    data.ctrl[:] = action.flatten()

    # Step simulation
    for _ in range(frame_skip):
        mujoco.mj_step(model, data)

    # Sync viewer and debug print
    viewer.sync()
    print(f"Position: {pos}, Velocity: {vel}, Controls: {ctrl}")

    time.sleep(0.01)

viewer.close()
