import mujoco_py
import time
import numpy as np

# Load the model
model = mujoco_py.load_model_from_path("acebott_spider.xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Define a simple walking cycle for the spider
# We'll control the leg motors to create a simple gait pattern
motor_ids = [
    sim.model.actuator_name2id("front_left_hip_joint"),
    sim.model.actuator_name2id("front_left_knee_joint"),
    sim.model.actuator_name2id("front_right_hip_joint"),
    sim.model.actuator_name2id("front_right_knee_joint"),
    sim.model.actuator_name2id("back_left_hip_joint"),
    sim.model.actuator_name2id("back_left_knee_joint"),
    sim.model.actuator_name2id("back_right_hip_joint"),
    sim.model.actuator_name2id("back_right_knee_joint")
]

# Function to control motor movements in a simple cycle
def set_motor_control(phase):
    # Set motor control for each joint based on the phase of the gait
    # Phase represents the "time" in the gait cycle (from 0 to 1)
    # We simulate a simple walking cycle where each leg moves in a coordinated manner
    cycle_positions = {
        0: [1, 0, -1, 0, 1, 0, -1, 0],  # initial position (legs 1, 2, 3 lifted)
        0.5: [-1, 0, 1, 0, -1, 0, 1, 0],  # half phase (legs 4, 5, 6 lifted)
    }

    motor_values = cycle_positions.get(phase, [0] * 8)
    sim.data.ctrl[:] = motor_values

# Simulate the walking movement
for _ in range(1000):  # Simulate for 1000 timesteps
    phase = (time.time() % 2) / 2  # Simple alternating phase for walking
    set_motor_control(phase)  # Apply motor control to simulate walking gait

    sim.step()  # Step the simulation forward
    viewer.render()  # Render the simulation
    
    time.sleep(0.01)  # Control the speed of simulation
