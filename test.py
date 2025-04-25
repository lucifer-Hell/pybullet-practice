import mujoco
import mujoco.viewer
model = mujoco.MjModel.from_xml_path("./spider2.xml")  # or "robot.xml"

data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
