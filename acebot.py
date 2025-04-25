# import mujoco
# import numpy as np
# import time
# from pathlib import Path

# model = mujoco.MjModel.from_xml_path("./acebott_spider.xml")
# data = mujoco.MjData(model)

# # Run for 5 seconds of simulation
# duration = 1005.0
# fps = int(1 / model.opt.timestep)
# steps = int(duration * fps)

# print("Running headless simulation...")

# for step in range(steps):
#     t = step * model.opt.timestep
#     for i in range(8):
#         data.ctrl[i] = 0.5 * np.sin(t * 2 + i)
#     mujoco.mj_step(model, data)

# print("Simulation completed.")


import mujoco
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("./acebott_spider.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# Step and render one frame
for i in range(100):
    for j in range(8):
        data.ctrl[j] = 0.5 * np.sin(i * 0.05 + j)
    mujoco.mj_step(model, data)

# Render and show the final image
renderer.update_scene(data)
img = renderer.render()
plt.imshow(img)
plt.axis('off')
plt.show()
