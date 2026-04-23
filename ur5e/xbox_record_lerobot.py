"""
UR5e + Robotiq 2F-85 手柄遥操作 + LeRobot 数据采集脚本
========================================================

在原 xbox_control.py 基础上，增加 LeRobot 数据集录制：
按 START 开始一个 episode，再按 START 结束并保存；X 用来丢弃失败的尝试；Y 用来重置方块。

按键映射（Xbox 手柄）
-----------------------------------
  左摇杆 X/Y    -> 末端 XY 平移（世界系）
  右摇杆 Y      -> 末端 Z 平移
  右摇杆 X      -> 绕世界 Z 旋转 yaw
  LB / RB       -> 绕 X 轴 pitch
  A             -> 夹爪打开 (ctrl=0)
  B             -> 夹爪闭合 (ctrl=255)
  Y             -> 随机化红/绿/蓝三个方块位置（开始新任务前按）
  X             -> 丢弃当前正在录制的 episode（失败重来）
  START         -> 开始 / 停止录制 episode
  BACK          -> 退出程序（自动 flush 数据集元数据）

依赖
-----
  pip install mujoco pygame numpy opencv-python
  pip install lerobot
  sudo apt install ffmpeg

用法
-----
  python xbox_record_lerobot.py \
      --repo-id local/ur5e_pick_place \
      --root ./data/ur5e_pick_place \
      --task "pick and place the cubes onto the matching-color zones" \
      --scene scene.xml

录制结束后，可以直接用 LeRobot 训练 ACT：
  python -m lerobot.scripts.train \
      --dataset.repo_id=local/ur5e_pick_place \
      --dataset.root=./data/ur5e_pick_place \
      --policy.type=act \
      --output_dir=outputs/train/ur5e_act


lerobot-train \
  --dataset.repo_id=local/ur5e_pick_place \
  --dataset.root=./data/ur5e_pick_place \
  --policy.type=act \
  --output_dir=outputs/train/ur5e_act \
  --job_name=ur5e_act \
  --policy.device=cuda \
  --batch_size=8 \
  --policy.push_to_hub=false
"""

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import pygame


# =========================================================
# 仿真参数
# =========================================================
integration_dt: float = 1.0
damping: float = 1e-4
gravity_compensation: bool = True
dt: float = 0.002
max_angvel: float = 3.14

# =========================================================
# 手柄 / 遥操作参数
# =========================================================
LINEAR_SPEED = 0.2
ANGULAR_SPEED = 1.0
DEAD_ZONE = 0.12

WS_X = (-0.8, 0.8)
WS_Y = (-0.8, 0.8)
WS_Z = (0.05, 1.2)

# 手柄轴 / 按钮编号（Xbox 常见映射，若不同请自行修改）
AXIS_LX, AXIS_LY = 0, 1
AXIS_RX, AXIS_RY = 3, 4
AXIS_LT, AXIS_RT = 2, 5

BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_Y = 0, 1, 2, 3
BUTTON_LB, BUTTON_RB = 4, 5
BUTTON_BACK, BUTTON_START = 6, 7

# =========================================================
# 录制参数
# =========================================================
RECORD_FPS = 30
IMG_H, IMG_W = 240, 320
CAMERA_NAMES = ["scene_cam", "top_cam", "wrist_cam"]

STATE_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3", "gripper",
]
ACTION_NAMES = list(STATE_NAMES)  # 关节空间 action 与 state 维度一致


# =========================================================
# 工具函数（与原脚本一致）
# =========================================================
def apply_deadzone(v: float, dz: float = DEAD_ZONE) -> float:
    if abs(v) < dz:
        return 0.0
    return (v - np.sign(v) * dz) / (1.0 - dz)


def yaw_to_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def set_free_body_pose(model, data, body_name, pos, quat):
    body_id = model.body(body_name).id
    jnt_adr = model.body_jntadr[body_id]
    qpos_adr = model.jnt_qposadr[jnt_adr]
    qvel_adr = model.jnt_dofadr[jnt_adr]
    data.qpos[qpos_adr:qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3:qpos_adr + 7] = quat
    data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def sample_point_in_zone(center_xy, radius, cube_half, yaw_margin=0.0):
    effective_r = max(0.0, radius - cube_half - yaw_margin)
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    r = effective_r * np.sqrt(np.random.uniform(0.0, 1.0))
    x = center_xy[0] + r * np.cos(theta)
    y = center_xy[1] + r * np.sin(theta)
    return np.array([x, y], dtype=np.float64)


def randomize_rgb_cubes_in_fixed_zones(model, data):
    table_top_z = 0.42
    cube_half = 0.02
    z = table_top_z + cube_half + 0.002

    zones = {
        "cube_red":   np.array([0.4, 0.0], dtype=np.float64),
        "cube_green": np.array([0.4, 0.0], dtype=np.float64),
        "cube_blue":  np.array([0.4, 0.0], dtype=np.float64),
    }
    zone_radius = 0.2

    for body_name, center_xy in zones.items():
        xy = sample_point_in_zone(center_xy, zone_radius, cube_half)
        yaw = np.random.uniform(-np.pi, np.pi)
        quat = yaw_to_quat(yaw)
        pos = np.array([xy[0], xy[1], z], dtype=np.float64)
        set_free_body_pose(model, data, body_name, pos, quat)
        print(f"  [{body_name}] x={xy[0]:.3f}, y={xy[1]:.3f}, yaw={yaw:.3f}")


def quat_mul_axis_angle(q, axis, angle):
    half = angle * 0.5
    dq = np.array([np.cos(half),
                   axis[0] * np.sin(half),
                   axis[1] * np.sin(half),
                   axis[2] * np.sin(half)], dtype=np.float64)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, dq, q)
    out /= np.linalg.norm(out) + 1e-12
    return out


def safe_get_axis(js, idx):
    return js.get_axis(idx) if idx < js.get_numaxes() else 0.0


def safe_get_button(js, idx):
    return js.get_button(idx) if idx < js.get_numbuttons() else 0


# =========================================================
# 渲染 & 显示
# =========================================================
def render_three_cameras_rgb(renderer, data, camera_names):
    """渲染三路相机，返回 RGB uint8 列表 [(H,W,3), ...]。"""
    imgs = []
    for cam_name in camera_names:
        renderer.update_scene(data, camera=cam_name)
        rgb = renderer.render().copy()  # copy 防止底层缓冲复用
        imgs.append(rgb)
    return imgs


def stack_canvas_bgr(images_rgb):
    """把 3 路 RGB 图像拼成 2x2 BGR canvas 用于 cv2 显示。"""
    top_row = np.hstack([images_rgb[0], images_rgb[1]])
    blank = np.zeros_like(images_rgb[2])
    bottom_row = np.hstack([images_rgb[2], blank])
    canvas = np.vstack([top_row, bottom_row])
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def draw_overlay(canvas_bgr, recording, episode_idx, frame_count,
                 total_episodes, task_name):
    """在画布顶部画录制状态条。"""
    h, w = canvas_bgr.shape[:2]
    cv2.rectangle(canvas_bgr, (0, 0), (w, 30), (0, 0, 0), -1)

    if recording:
        cv2.circle(canvas_bgr, (15, 15), 8, (0, 0, 255), -1)
        status = f"REC  ep={episode_idx}  frames={frame_count}"
        color = (0, 0, 255)
    else:
        status = f"IDLE  total_ep={total_episodes}  (press START to record)"
        color = (220, 220, 220)
    cv2.putText(canvas_bgr, status, (35, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 底栏画任务名
    cv2.rectangle(canvas_bgr, (0, h - 22), (w, h), (0, 0, 0), -1)
    task_short = (task_name[:80] + "...") if len(task_name) > 83 else task_name
    cv2.putText(canvas_bgr, f"task: {task_short}", (10, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 180), 1, cv2.LINE_AA)
    return canvas_bgr


# =========================================================
# LeRobot 数据集构建
# =========================================================
def build_features(use_videos: bool = True):
    img_dtype = "video" if use_videos else "image"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
    }
    for cam in CAMERA_NAMES:
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _import_lerobot_dataset():
    """兼容新旧两种 import 路径:
       - 新版 (lerobot >= 0.3): lerobot.datasets.lerobot_dataset
       - 老版:                   lerobot.common.datasets.lerobot_dataset
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset


def load_or_create_dataset(repo_id, root, fps, use_videos: bool = True):
    """新建或增量续录 LeRobotDataset。"""
    LeRobotDataset = _import_lerobot_dataset()

    root = Path(root)
    info_path = root / "meta" / "info.json"

    if info_path.exists():
        print(f"[Dataset] 检测到已有数据集, 进入追加模式: {root}")
        dataset = LeRobotDataset(repo_id, root=root)
    else:
        mode = "video (mp4)" if use_videos else "image (png)"
        print(f"[Dataset] 创建新数据集 [{mode}]: {root}")
        features = build_features(use_videos=use_videos)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            features=features,
            robot_type="ur5e_2f85",
            use_videos=use_videos,
            # 关键: 用独立子进程编码视频, 避免和 MuJoCo OpenGL renderer
            # 争抢资源导致 SIGSEGV
            image_writer_processes=1,
            image_writer_threads=2,
        )
    return dataset


# =========================================================
# 主循环
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="local/ur5e_pick_place")
    parser.add_argument("--root", type=str, default="./data/ur5e_pick_place")
    parser.add_argument(
        "--task", type=str,
        default="pick and place the cubes onto the matching-color zones",
    )
    parser.add_argument("--scene", type=str, default="scene.xml")
    parser.add_argument(
        "--image-mode", action="store_true",
        help="用 image (png) 而不是 video (mp4) 存储相机数据. "
             "如果 video 模式经常 SIGSEGV, 切到这个选项. 磁盘占用大约多 30 倍.",
    )
    args = parser.parse_args()

    assert mujoco.__version__ >= "3.1.0", "请升级到 mujoco >= 3.1.0"

    # -------------------- 手柄 --------------------
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("未检测到手柄!请插入手柄后重试。")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"[Gamepad] 已连接: {js.get_name()}")
    print(f"[Gamepad] 轴数={js.get_numaxes()}, 按钮数={js.get_numbuttons()}")

    # -------------------- 数据集 --------------------
    dataset = load_or_create_dataset(
        args.repo_id, args.root, RECORD_FPS,
        use_videos=not args.image_mode,
    )

    # -------------------- MuJoCo 模型 --------------------
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    renderer = mujoco.Renderer(model, width=IMG_W, height=IMG_H)
    for name in CAMERA_NAMES:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id == -1:
            raise ValueError(f"未找到相机: {name}")
        print(f"[Camera] {name} id={cam_id}")

    cv2.namedWindow("MuJoCo Multi-Camera", cv2.WINDOW_NORMAL)

    site_id = model.site("pinch").id
    mocap_id = model.body("target").mocapid[0]
    key_id = model.key("home").id

    # 重力补偿（只对机械臂 link）
    if gravity_compensation:
        for name in ["shoulder_link", "upper_arm_link", "forearm_link",
                     "wrist_1_link", "wrist_2_link", "wrist_3_link"]:
            model.body_gravcomp[model.body(name).id] = 1.0

    # 关节 / 驱动器索引
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    dof_ids = np.array([model.joint(n).id for n in joint_names])
    qpos_ids = model.jnt_qposadr[dof_ids]

    actuator_names = ["shoulder_pan", "shoulder_lift", "elbow",
                      "wrist_1", "wrist_2", "wrist_3"]
    actuator_ids = np.array([model.actuator(n).id for n in actuator_names])
    gripper_act_id = model.actuator("fingers_actuator").id
    gripper_ctrl = 0.0  # 0=open, 255=close

    # 夹爪真实开合度（用 left_driver_joint 的 qpos 做归一化）
    driver_joint_id = model.joint("left_driver_joint").id
    driver_qpos_adr = model.jnt_qposadr[driver_joint_id]
    DRIVER_RANGE_MAX = 0.8  # XML 里 driver class 的 range="0 0.8"

    # Diff-IK 预分配
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    err = np.zeros(6)
    err_pos = err[:3]
    err_ori = err[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    err_quat = np.zeros(4)

    # -------------------- 录制状态 (必须在 try 外初始化, finally 要用) --------------------
    recording = False
    frame_count = 0
    episode_idx = dataset.num_episodes  # 下一个 episode 的编号

    try:
        # -------------------- 启动 viewer --------------------
        with mujoco.viewer.launch_passive(
                model=model, data=data,
                show_left_ui=False, show_right_ui=False) as viewer:

            mujoco.mj_resetDataKeyframe(model, data, key_id)
            mujoco.mj_forward(model, data)

            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            data.mocap_pos[mocap_id] = data.site(site_id).xpos.copy()
            mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(site_id).xmat)

            # 录制节拍
            record_period = 1.0 / RECORD_FPS
            last_render_time = 0.0
            last_loop_time = time.time()
            MAX_SUBSTEPS = 50

            print("\n" + "=" * 64)
            print("按键说明")
            print("  START -> 开始 / 停止 录制当前 episode")
            print("  BACK  -> 退出程序（自动保存元数据）")
            print("  Y     -> 随机化方块位置（新任务前按）")
            print("  X     -> 丢弃当前录制（失败重来）")
            print("  A/B   -> 夹爪 开/合")
            print("  LB/RB -> 末端 pitch")
            print(f"已存在 episode 数: {episode_idx}")
            print("=" * 64 + "\n")

            running = True
            while viewer.is_running() and running:
                now = time.time()
                frame_dt = now - last_loop_time
                last_loop_time = now

                # ================= 1. 处理手柄按键事件 =================
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    elif event.type == pygame.JOYBUTTONDOWN:
                        # BACK -> 退出
                        if event.button == BUTTON_BACK:
                            running = False

                        # START -> 开始 / 停止录制
                        elif event.button == BUTTON_START:
                            if not recording:
                                recording = True
                                frame_count = 0
                                print(f"\n[REC] >>> 开始录制 episode {episode_idx}")
                            else:
                                recording = False
                                if frame_count > 0:
                                    try:
                                        dataset.save_episode()
                                        print(f"[REC] <<< 保存 episode {episode_idx}, "
                                              f"共 {frame_count} 帧")
                                        episode_idx = dataset.num_episodes
                                    except Exception as e:
                                        print(f"[REC] !! save_episode 失败: {e}")
                                        _safe_clear(dataset)
                                else:
                                    print("[REC] 没有帧, 跳过保存")
                                    _safe_clear(dataset)
                                frame_count = 0

                        # X -> 丢弃当前 episode
                        elif event.button == BUTTON_X:
                            if recording:
                                print(f"[REC] xxx 丢弃 episode {episode_idx}")
                                _safe_clear(dataset)
                                recording = False
                                frame_count = 0
                            else:
                                print("[REC] 当前未在录制, X 被忽略")

                        # Y -> 随机化方块
                        elif event.button == BUTTON_Y:
                            print("[Action] 随机化方块位置:")
                            randomize_rgb_cubes_in_fixed_zones(model, data)
                            mujoco.mj_forward(model, data)

                        # A -> 夹爪开
                        elif event.button == BUTTON_A:
                            gripper_ctrl = 0.0
                            data.ctrl[gripper_act_id] = gripper_ctrl

                        # B -> 夹爪闭
                        elif event.button == BUTTON_B:
                            gripper_ctrl = 255.0
                            data.ctrl[gripper_act_id] = gripper_ctrl

                # ================= 2. 读取连续摇杆 =================
                lx = apply_deadzone(safe_get_axis(js, AXIS_LX))
                ly = apply_deadzone(safe_get_axis(js, AXIS_LY))
                rx = apply_deadzone(safe_get_axis(js, AXIS_RX))
                ry = apply_deadzone(safe_get_axis(js, AXIS_RY))

                # ================= 3. 更新 mocap target =================
                data.mocap_pos[mocap_id, 0] += lx * LINEAR_SPEED * frame_dt
                data.mocap_pos[mocap_id, 1] += -ly * LINEAR_SPEED * frame_dt
                data.mocap_pos[mocap_id, 2] += -ry * LINEAR_SPEED * frame_dt
                data.mocap_pos[mocap_id, 0] = np.clip(data.mocap_pos[mocap_id, 0], *WS_X)
                data.mocap_pos[mocap_id, 1] = np.clip(data.mocap_pos[mocap_id, 1], *WS_Y)
                data.mocap_pos[mocap_id, 2] = np.clip(data.mocap_pos[mocap_id, 2], *WS_Z)

                q_cur = data.mocap_quat[mocap_id].copy()
                if abs(rx) > 0.0:
                    q_cur = quat_mul_axis_angle(q_cur, np.array([0, 0, 1]),
                                                -rx * ANGULAR_SPEED * frame_dt)
                if safe_get_button(js, BUTTON_LB):
                    q_cur = quat_mul_axis_angle(q_cur, np.array([1, 0, 0]),
                                                -ANGULAR_SPEED * frame_dt)
                if safe_get_button(js, BUTTON_RB):
                    q_cur = quat_mul_axis_angle(q_cur, np.array([1, 0, 0]),
                                                +ANGULAR_SPEED * frame_dt)
                data.mocap_quat[mocap_id] = q_cur

                # ================= 4. 多步仿真 + Diff-IK =================
                n_sub = int(min(MAX_SUBSTEPS, max(1, round(frame_dt / dt))))
                for _ in range(n_sub):
                    err_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos
                    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
                    mujoco.mju_negQuat(site_quat_conj, site_quat)
                    mujoco.mju_mulQuat(err_quat, data.mocap_quat[mocap_id], site_quat_conj)
                    mujoco.mju_quat2Vel(err_ori, err_quat, 1.0)

                    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
                    dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, err)

                    if max_angvel > 0:
                        dq_abs_max = np.abs(dq).max()
                        if dq_abs_max > max_angvel:
                            dq *= max_angvel / dq_abs_max

                    q = data.qpos.copy()
                    mujoco.mj_integratePos(model, q, dq, integration_dt)

                    q_target = q[qpos_ids]
                    jnt_ranges = model.jnt_range[dof_ids]
                    q_target = np.clip(q_target, jnt_ranges[:, 0], jnt_ranges[:, 1])
                    data.ctrl[actuator_ids] = q_target

                    mujoco.mj_step(model, data)

                viewer.sync()

                # ================= 5. 30Hz 渲染 + 数据采集 =================
                if now - last_render_time >= record_period:
                    images_rgb = render_three_cameras_rgb(renderer, data, CAMERA_NAMES)

                    # ------- 如果正在录制, 采集一帧 -------
                    if recording:
                        # state: 6 关节 qpos + 1 夹爪真实归一化开合度
                        gripper_state_norm = float(
                            np.clip(data.qpos[driver_qpos_adr] / DRIVER_RANGE_MAX, 0.0, 1.0)
                        )
                        state = np.concatenate([
                            data.qpos[qpos_ids].astype(np.float32),
                            np.array([gripper_state_norm], dtype=np.float32),
                        ])
                        # action: 6 关节目标 ctrl + 1 夹爪归一化 ctrl (0=open, 1=close)
                        action = np.concatenate([
                            data.ctrl[actuator_ids].astype(np.float32),
                            np.array([gripper_ctrl / 255.0], dtype=np.float32),
                        ])

                        frame = {
                            "observation.state": state,
                            "action": action,
                        }
                        for cam_name, img in zip(CAMERA_NAMES, images_rgb):
                            frame[f"observation.images.{cam_name}"] = img

                        try:
                            dataset.add_frame(frame, task=args.task)
                            frame_count += 1
                        except TypeError:
                            # 兼容旧版 API: task 在 frame 字典里
                            frame["task"] = args.task
                            dataset.add_frame(frame)
                            frame_count += 1
                        except Exception as e:
                            print(f"[REC] add_frame 失败: {e}")
                            recording = False

                    # ------- 显示 -------
                    canvas_bgr = stack_canvas_bgr(images_rgb)
                    canvas_bgr = draw_overlay(
                        canvas_bgr, recording, episode_idx, frame_count,
                        dataset.num_episodes, args.task,
                    )
                    cv2.imshow("MuJoCo Multi-Camera", canvas_bgr)
                    last_render_time = now

                # 即使没有新帧也要 waitKey 让 cv2 窗口响应
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    running = False

            # -------------------- 退出前清理 --------------------
            if recording:
                print("[REC] 退出时仍在录制, 尝试保存最后一个 episode...")
                if frame_count > 0:
                    try:
                        dataset.save_episode()
                        print(f"[REC] 已保存 episode {episode_idx}, 共 {frame_count} 帧")
                    except Exception as e:
                        print(f"[REC] 保存失败: {e}")
                        _safe_clear(dataset)
                else:
                    _safe_clear(dataset)

    except KeyboardInterrupt:
        print("\n[Main] 收到 Ctrl+C, 正在安全退出...")

    finally:
        # 关键: 无论怎么退出, 这里都会跑
        # 1) 如果退出时还在录制, 把已有帧保存成最后一条 episode (但不会保存没帧的空 episode)
        if recording:
            print("[REC] 退出时仍在录制...")
            if frame_count > 0:
                try:
                    dataset.save_episode()
                    print(f"[REC] 已保存最后一条 episode, 共 {frame_count} 帧")
                except Exception as e:
                    print(f"[REC] 最后一条保存失败: {e}")
                    _safe_clear(dataset)
            else:
                _safe_clear(dataset)

        # 2) v3.0 必须调用 finalize(), 否则 meta/episodes/*.parquet 不完整、数据集无法加载
        _safe_finalize(dataset)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass

        print(f"\n[Done] 数据集 episode 总数: {dataset.num_episodes}")
        print(f"       保存路径: {Path(args.root).resolve()}")
        print("       可以开始训练 ACT 了:")
        print(f"         lerobot-train \\")
        print(f"             --dataset.repo_id={args.repo_id} \\")
        print(f"             --dataset.root={args.root} \\")
        print(f"             --policy.type=act \\")
        print(f"             --output_dir=outputs/train/ur5e_act \\")
        print(f"             --job_name=ur5e_act \\")
        print(f"             --policy.device=cuda \\")
        print(f"             --policy.push_to_hub=false")


def _safe_clear(dataset):
    """丢弃当前 episode buffer，处理不同 LeRobot 版本的 API 差异。"""
    for attr in ("clear_episode_buffer", "clear_buffer", "reset_episode_buffer"):
        fn = getattr(dataset, attr, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception as e:
                print(f"[REC] {attr} 失败: {e}")
    print("[REC] 警告: 找不到清空 episode buffer 的方法, 请检查 lerobot 版本")


def _safe_finalize(dataset):
    """关闭数据集的 parquet writer, 写入元数据 footer。
       v3.0 格式必须调用; 老版本可能没有这个方法, 就跳过。
    """
    fn = getattr(dataset, "finalize", None)
    if callable(fn):
        try:
            fn()
            print("[Dataset] finalize() 完成")
        except Exception as e:
            print(f"[Dataset] finalize 失败: {e}")
    else:
        print("[Dataset] 老版本 LeRobot, 无需 finalize")


if __name__ == "__main__":
    main()