import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
