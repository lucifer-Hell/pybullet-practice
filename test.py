import time
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

# Load MuJoCo model and data
model = mujoco.MjModel.from_xml_path("./default_world.xml")
data = mujoco.MjData(model)
# ppo_path='./checkpoints/ppo_spider_28800000_steps'
ppo_path='ppo_spider_walk'
# Load trained PPO model
ppo_model = PPO.load(ppo_path)

# Launch passive viewer
viewer = mujoco.viewer.launch_passive(model, data)

# Debug info
print(f"Number of sensors: {model.nsensor}")
print(f"Size of sensordata array: {data.sensordata.shape}")
print(f"Number of actuators: {model.nu}")

frame_skip = 5
action_timer=0
seq_count=20
# Simulation loop
while viewer.is_running():
    # === Collect Observation ===
    pos = data.sensordata[0:3]        # x, y, z from <framepos>
    vel = data.sensordata[3:6]        # vx, vy, vz from <velocimeter>
    gyro = data.sensordata[6:9]       # wx, wy, wz from <angularvelocity>
    ctrl = data.ctrl[:]               # current actuator control

    # Combine to match training format (13D input)
    obs = np.concatenate([pos, vel, gyro, ctrl]).astype(np.float32).reshape(1, -1)

    # Predict action using PPO model
    if action_timer==0:
        action, _ = ppo_model.predict(obs)

    data.ctrl[:] = action.flatten()

    # Step simulation
    for _ in range(frame_skip):
        mujoco.mj_step(model, data)

    # Sync viewer and print for debugging
    viewer.sync()
    print(f"Position: {pos}, Velocity: {vel}, Gyro: {gyro}, Controls: {ctrl}")
    action_timer = (action_timer +1)%seq_count
    time.sleep(0.01)

viewer.close()
