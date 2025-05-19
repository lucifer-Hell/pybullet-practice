import time
import mujoco
import numpy as np
from stable_baselines3 import TD3

# Load MuJoCo model and data
model = mujoco.MjModel.from_xml_path("./default_world.xml")
data = mujoco.MjData(model)

# Load trained TD3 model
td3_model = TD3.load("td3_spider_walk")

# Launch viewer
viewer = mujoco.viewer.launch_passive(model, data)

frame_skip = 5
action_timer = 0
seq_count = 20

while viewer.is_running():
    pos = data.sensordata[0:3]
    vel = data.sensordata[3:6]
    gyro = data.sensordata[6:9]
    ctrl = data.ctrl[:]

    obs = np.concatenate([pos, vel, gyro, ctrl]).astype(np.float32).reshape(1, -1)

    if action_timer == 0:
        action, _ = td3_model.predict(obs)

    data.ctrl[:] = action.flatten()

    for _ in range(frame_skip):
        mujoco.mj_step(model, data)

    viewer.sync()
    print(f"Position: {pos}, Velocity: {vel}, Gyro: {gyro}, Controls: {ctrl}")
    action_timer = (action_timer + 1) % seq_count
    time.sleep(0.01)

viewer.close()
