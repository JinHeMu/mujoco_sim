"""
UR5e + Robotiq 2F-85 仿真控制示例
================================================
功能：
  1) 加载合并好的 scene.xml
  2) 在 viewer 中运行仿真
  3) 演示一段"抬起 -> 闭合夹爪 -> 放下 -> 张开"的简单动作
  4) 演示从 3 个相机采集图像并保存成 png

运行：
    python demo_control.py             # 可视化 + 动作循环
    python demo_control.py --capture   # 只采一帧三视角图像
"""

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


SCENE_PATH = str(Path(__file__).parent / "scene.xml")


# ---------- 工具：按名字查索引 ----------
def actuator_id(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def body_id(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


# ---------- 控制封装 ----------
class RobotController:
    """把 UR5e 6 关节 + 夹爪 1 控制量整理成更方便的接口。"""

    ARM_ACTUATORS = [
        "shoulder_pan", "shoulder_lift", "elbow",
        "wrist_1", "wrist_2", "wrist_3",
    ]
    GRIPPER_ACTUATOR = "fingers_actuator"

    HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.arm_ids = [actuator_id(model, n) for n in self.ARM_ACTUATORS]
        self.gripper_id = actuator_id(model, self.GRIPPER_ACTUATOR)

    # ----- 写控制指令 -----
    def set_arm(self, q: np.ndarray):
        """设置 6 个关节目标角（弧度）"""
        assert len(q) == 6
        for i, aid in enumerate(self.arm_ids):
            self.data.ctrl[aid] = q[i]

    def set_gripper(self, value: float):
        """
        设置夹爪开合：
          0   -> 完全张开
          255 -> 完全闭合
        """
        self.data.ctrl[self.gripper_id] = float(np.clip(value, 0, 255))

    # ----- 读状态 -----
    def get_arm_qpos(self) -> np.ndarray:
        """读当前 6 关节角度"""
        # UR5e 的 6 个关节在 qpos 的最前 6 位（因为 base 没有 freejoint）
        # 更稳妥的做法：通过 joint name 查 qposadr
        q = np.zeros(6)
        names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        for i, n in enumerate(names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            q[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
        return q


# ---------- 相机采集 ----------
def capture_cameras(model, data, camera_names, width=640, height=480):
    """用 mujoco.Renderer 依次渲染若干相机，返回 {name: np.uint8 array}"""
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = {}
    for name in camera_names:
        renderer.update_scene(data, camera=name)
        frames[name] = renderer.render().copy()
    renderer.close()
    return frames


def save_frames(frames: dict, out_dir: Path):
    """把采集到的帧保存为 png。用 imageio，没装就退回 PIL。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        for name, img in frames.items():
            imageio.imwrite(out_dir / f"{name}.png", img)
    except ImportError:
        from PIL import Image
        for name, img in frames.items():
            Image.fromarray(img).save(out_dir / f"{name}.png")
    print(f"[✓] 已保存 {len(frames)} 张图片到 {out_dir}")


# ---------- 主程序 ----------
def run_viewer():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    # 加载 "home" keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    ctl = RobotController(model, data)

    print("=== UR5e + Robotiq 2F-85 仿真 ===")
    print("关节 actuator:", ctl.ARM_ACTUATORS)
    print("夹爪 actuator:", ctl.GRIPPER_ACTUATOR)
    print("鼠标左键旋转视角，右键平移，滚轮缩放")
    print("按 ESC 退出\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 初始控制量
        ctl.set_arm(ctl.HOME)
        ctl.set_gripper(0)  # 张开

        start_time = time.time()
        phase = 0
        phase_end = 2.0  # 每个阶段 2 秒

        while viewer.is_running():
            t = data.time

            # ---------- 简单的状态机 ----------
            if t > phase_end:
                phase = (phase + 1) % 4
                phase_end = t + 2.0

            if phase == 0:
                # 张开夹爪、回 home
                ctl.set_arm(ctl.HOME)
                ctl.set_gripper(0)
            elif phase == 1:
                # 肘部下压一点，模拟伸向物体
                ctl.set_arm(ctl.HOME + np.array([0, 0.3, -0.5, 0.2, 0, 0]))
                ctl.set_gripper(0)
            elif phase == 2:
                # 闭合夹爪
                ctl.set_arm(ctl.HOME + np.array([0, 0.3, -0.5, 0.2, 0, 0]))
                ctl.set_gripper(200)
            elif phase == 3:
                # 抬起
                ctl.set_arm(ctl.HOME + np.array([0, 0.0, -0.3, 0.0, 0, 0]))
                ctl.set_gripper(200)

            mujoco.mj_step(model, data)

            # 实时同步：让仿真时间逼近真实时间
            time_until_next_step = model.opt.timestep - (time.time() - start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            start_time = time.time()

            viewer.sync()


def run_capture():
    """仅采集一次三视角图像"""
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    # 先让物理稳定几步
    for _ in range(200):
        mujoco.mj_step(model, data)

    frames = capture_cameras(
        model, data,
        camera_names=["scene_cam", "top_cam", "wrist_cam"],
        width=640, height=480,
    )
    save_frames(frames, Path(__file__).parent / "captures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true",
                        help="仅采集三视角图像到 captures/")
    args = parser.parse_args()

    if args.capture:
        run_capture()
    else:
        run_viewer()

