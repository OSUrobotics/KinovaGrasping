from mujoco_py import MjViewer, load_model_from_path, MjSim
# import gym

model = load_model_from_path("j2s7s300_end_effector.xml")
robot = MjSim(model)
viewer = MjViewer(robot)
while True:
	viewer.render()