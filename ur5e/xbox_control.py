"""
使用游戏手柄(Xbox/PS 兼容)通过差分逆运动学(Diff-IK)
远程操控 UR5e + Robotiq 2F-85。

默认按键映射(Xbox 手柄):
    左摇杆 X/Y   -> 末端 XY 平移(世界系)
    右摇杆 Y     -> 末端 Z 平移
    右摇杆 X     -> 绕世界 Z 旋转(yaw)
    LB / RB      -> 绕 X 轴俯仰(pitch)
    LT / RT      -> 夹爪闭合 / 张开(模拟量)
    A 键         -> 复位到 "home" keyframe
    Back 键      -> 退出
    Start 键     -> 打印当前末端位姿(调试)

如果你的手柄轴/按钮编号不同,在运行时会打印轴数和按钮数,
按住某个轴或键观察变化后,调整下面 AXIS_* / BUTTON_* 常量即可。

依赖: pip install mujoco pygame numpy
"""

import time
import cv2
import numpy as np
import mujoco
import mujoco.viewer
import pygame


# =========================================================
# 仿真参数
# =========================================================
integration_dt: float = 1.0      # 关节速度积分时长
damping: float = 1e-4             # 伪逆阻尼
gravity_compensation: bool = True
dt: float = 0.002                 # 仿真步长
max_angvel: float = 3.14          # 单关节最大角速度(rad/s),0=不限

# =========================================================
# 手柄/遥操作参数
# =========================================================
LINEAR_SPEED = 0.2       # 末端线速度 m/s
ANGULAR_SPEED = 1.0       # 末端角速度 rad/s
GRIPPER_SPEED = 1000   # 夹爪 ctrl 变化速度 (ctrl range 0~255)
DEAD_ZONE = 0.12          # 摇杆死区

# 工作空间裁剪(防止目标跑出可达范围)
WS_X = (-0.8, 0.8)
WS_Y = (-0.8, 0.8)
WS_Z = (0.05, 1.2)

# =========================================================
# 手柄轴/按钮编号(Linux/Windows Xbox 控制器的常见映射)
# 如果你的手柄不同,改这里
# =========================================================
AXIS_LX = 0       # 左摇杆 X
AXIS_LY = 1       # 左摇杆 Y
AXIS_RX = 3       # 右摇杆 X (有些系统是 2)
AXIS_RY = 4       # 右摇杆 Y (有些系统是 3)
AXIS_LT = 2       # 左扳机   (有些系统是 4)
AXIS_RT = 5       # 右扳机   (有些系统是 5)

BUTTON_A     = 0
BUTTON_B     = 1
BUTTON_X     = 2
BUTTON_Y     = 3
BUTTON_LB    = 4
BUTTON_RB    = 5
BUTTON_BACK  = 6
BUTTON_START = 7


# =========================================================
# 工具函数
# =========================================================
def apply_deadzone(v: float, dz: float = DEAD_ZONE) -> float:
    """对摇杆输入施加死区并做线性重标定到 [-1, 1]。"""
    if abs(v) < dz:
        return 0.0
    return (v - np.sign(v) * dz) / (1.0 - dz)

def yaw_to_quat(yaw: float) -> np.ndarray:
    """仅绕世界 Z 轴旋转的四元数 [w, x, y, z]。"""
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def set_free_body_pose(model, data, body_name: str, pos: np.ndarray, quat: np.ndarray) -> None:
    """设置带 freejoint 的 body 位姿，并清零速度。"""
    body_id = model.body(body_name).id
    jnt_adr = model.body_jntadr[body_id]
    qpos_adr = model.jnt_qposadr[jnt_adr]
    qvel_adr = model.jnt_dofadr[jnt_adr]

    # freejoint qpos = [x, y, z, qw, qx, qy, qz]
    data.qpos[qpos_adr:qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3:qpos_adr + 7] = quat

    # freejoint qvel = [vx, vy, vz, wx, wy, wz]
    data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def sample_point_in_zone(center_xy, radius, cube_half, yaw_margin=0.0):
    """在圆形区域内随机采样一个点，给 cube 留出边缘余量。"""
    effective_r = max(0.0, radius - cube_half - yaw_margin)
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    r = effective_r * np.sqrt(np.random.uniform(0.0, 1.0))
    x = center_xy[0] + r * np.cos(theta)
    y = center_xy[1] + r * np.sin(theta)
    return np.array([x, y], dtype=np.float64)


def randomize_rgb_cubes_in_fixed_zones(model, data) -> None:
    """
    将红/绿/蓝三个方块分别随机放到各自固定区域内：
      red   -> 红区
      green -> 绿区
      blue  -> 蓝区
    """
    # ---------------------------
    # 桌面参数（按你的 XML 写死）
    # table body pos = (0.5, 0, 0.2)
    # table_top size = (0.3, 0.4, 0.02), pos = (0, 0, 0.2)
    # 桌面上表面 z = 0.2 + 0.2 + 0.02 = 0.42
    # ---------------------------
    table_top_z = 0.42

    # 方块半尺寸：size="0.02 0.02 0.02"
    cube_half = 0.02

    # 放到桌面上方一点点，避免初始嵌入
    z = table_top_z + cube_half + 0.002

    # 三个放置区在世界坐标下的中心
    zones = {
        "cube_red":   np.array([0.4, 0.0], dtype=np.float64),
        "cube_green": np.array([0.4,  0.00], dtype=np.float64),
        "cube_blue":  np.array([0.4,  0.0], dtype=np.float64),
    }

    # 你的 zone cylinder 半径是 0.04
    zone_radius = 0.2

    for body_name, center_xy in zones.items():
        xy = sample_point_in_zone(center_xy, zone_radius, cube_half)
        yaw = np.random.uniform(-np.pi, np.pi)
        quat = yaw_to_quat(yaw)

        pos = np.array([xy[0], xy[1], z], dtype=np.float64)
        set_free_body_pose(model, data, body_name, pos, quat)

        print(f"[{body_name}] x={xy[0]:.3f}, y={xy[1]:.3f}, z={z:.3f}, yaw={yaw:.3f}")



def render_three_cameras(renderer, data, camera_names):
    """渲染三个相机，并拼成 2x2 画布:
       [scene_cam | top_cam]
       [wrist_cam | blank   ]
    """
    images = []
    for cam_name in camera_names:
        renderer.update_scene(data, camera=cam_name)
        rgb = renderer.render()
        images.append(rgb.copy())   # 建议 copy，避免底层缓冲区复用导致显示异常

    top_row = np.hstack([images[0], images[1]])
    blank = np.zeros_like(images[2])
    bottom_row = np.hstack([images[2], blank])
    canvas = np.vstack([top_row, bottom_row])

    # MuJoCo 返回 RGB，OpenCV 需要 BGR
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    return canvas_bgr


def quat_mul_axis_angle(q: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """把当前四元数 q 绕给定轴再旋转 angle 弧度(左乘增量四元数)。"""
    half = angle * 0.5
    dq = np.array([np.cos(half),
                   axis[0] * np.sin(half),
                   axis[1] * np.sin(half),
                   axis[2] * np.sin(half)], dtype=np.float64)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, dq, q)
    # 归一化防漂
    out /= np.linalg.norm(out) + 1e-12
    return out


def safe_get_axis(js, idx: int) -> float:
    if idx < js.get_numaxes():
        return js.get_axis(idx)
    return 0.0


def safe_get_button(js, idx: int) -> int:
    if idx < js.get_numbuttons():
        return js.get_button(idx)
    return 0


# =========================================================
# 主循环
# =========================================================
def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "请升级到 mujoco >= 3.1.0"

    # ---------- 初始化手柄 ----------
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("未检测到手柄!请插入手柄后重试。")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"[Gamepad] 已连接: {js.get_name()}")
    print(f"[Gamepad] 轴数={js.get_numaxes()}, 按钮数={js.get_numbuttons()}, "
          f"帽={js.get_numhats()}")

    # ---------- 载入模型 ----------
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    # ---------- 多相机离屏渲染 ----------
    CAM_W, CAM_H = 320, 240
    camera_names = ["scene_cam", "top_cam", "wrist_cam"]
    renderer = mujoco.Renderer(model, width=CAM_W, height=CAM_H)

    # 检查相机是否存在
    for name in camera_names:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id == -1:
            raise ValueError(f"未找到相机: {name}")
        print(f"[Camera] {name} id={cam_id}")

    cv2.namedWindow("MuJoCo Multi-Camera", cv2.WINDOW_NORMAL)


    site_id  = model.site("pinch").id
    mocap_id = model.body("target").mocapid[0]
    key_id   = model.key("home").id

    # 重力补偿
    body_names = ["shoulder_link", "upper_arm_link", "forearm_link",
                  "wrist_1_link", "wrist_2_link", "wrist_3_link"]
    if gravity_compensation:
        for name in body_names:
            model.body_gravcomp[model.body(name).id] = 1.0

    # 机械臂关节/驱动器 id
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    dof_ids  = np.array([model.joint(n).id for n in joint_names])
    qpos_ids = model.jnt_qposadr[dof_ids]

    actuator_names = ["shoulder_pan", "shoulder_lift", "elbow",
                      "wrist_1", "wrist_2", "wrist_3"]
    actuator_ids = np.array([model.actuator(n).id for n in actuator_names])

    # 夹爪驱动器
    gripper_act_id = model.actuator("fingers_actuator").id
    gripper_ctrl = 0.0  # 当前夹爪指令 (0=张开, 255=闭合)

    # 预分配
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    err  = np.zeros(6)
    err_pos = err[:3]
    err_ori = err[3:]
    site_quat      = np.zeros(4)
    site_quat_conj = np.zeros(4)
    err_quat       = np.zeros(4)

    # ---------- 启动 viewer ----------
    with mujoco.viewer.launch_passive(
            model=model, data=data,
            show_left_ui=False, show_right_ui=False) as viewer:

        mujoco.mj_resetDataKeyframe(model, data, key_id)
        # 关键:reset 只写 qpos,派生量(site 位姿、body xpos 等)必须手动前向传播
        mujoco.mj_forward(model, data)

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # 把 mocap 初始化到当前末端位姿,避免上电瞬间猛拉
        data.mocap_pos[mocap_id] = data.site(site_id).xpos.copy()
        mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(site_id).xmat)

        # ---- 在进入主循环前,增加这些变量 ----
        RENDER_HZ = 30
        render_period = 1.0 / RENDER_HZ
        last_render_time = time.time()
        last_loop_time = time.time()

        # 一次外层循环里,把仿真"追"到墙钟 now 所需的步数上限(保护性)
        MAX_SUBSTEPS = 50

        running = True
        while viewer.is_running() and running:
            now = time.time()
            frame_dt = now - last_loop_time  # 距离上次循环真实经过的时间
            last_loop_time = now

            # ---------- 1. 手柄事件 ----------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.JOYBUTTONDOWN:
                    # Back: 退出
                    if event.button == BUTTON_BACK:
                        running = False

                    elif event.button == BUTTON_Y:
                        # mujoco.mj_resetDataKeyframe(model, data, key_id)

                        # 红绿蓝三个方块分别随机到各自固定区域
                        randomize_rgb_cubes_in_fixed_zones(model, data)

                        mujoco.mj_forward(model, data)

                        # mocap 同步到当前末端，避免复位后 target 还停在旧位置
                        # data.mocap_pos[mocap_id] = data.site(site_id).xpos.copy()
                        # mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(site_id).xmat)

                        # # 夹爪复位为张开
                        # gripper_ctrl = 0.0
                        # data.ctrl[gripper_act_id] = gripper_ctrl

                        print("[Action] Reset to home + randomized RGB cubes.")


                    # X: 打印当前关节值
                    elif event.button == BUTTON_X:
                        q_now = data.qpos[qpos_ids].copy()
                        print("[Joint qpos]", np.array2string(q_now, precision=4, suppress_small=True))

                    # A: 夹爪打开
                    elif event.button == BUTTON_A:
                        gripper_ctrl = 0.0
                        data.ctrl[gripper_act_id] = gripper_ctrl
                        print("[Gripper] Open")

                    # B: 夹爪闭合
                    elif event.button == BUTTON_B:
                        gripper_ctrl = 255.0
                        data.ctrl[gripper_act_id] = gripper_ctrl
                        print("[Gripper] Close")

                    # Start: 打印当前末端位姿
                    elif event.button == BUTTON_START:
                        ee_pos = data.site(site_id).xpos.copy()
                        ee_quat = np.zeros(4)
                        mujoco.mju_mat2Quat(ee_quat, data.site(site_id).xmat)
                        print("[EE pos ]", np.array2string(ee_pos, precision=4, suppress_small=True))
                        print("[EE quat]", np.array2string(ee_quat, precision=4, suppress_small=True))

            # ---------- 2. 读取手柄连续值 ----------
            lx = apply_deadzone(safe_get_axis(js, AXIS_LX))
            ly = apply_deadzone(safe_get_axis(js, AXIS_LY))
            rx = apply_deadzone(safe_get_axis(js, AXIS_RX))
            ry = apply_deadzone(safe_get_axis(js, AXIS_RY))
            lt = (safe_get_axis(js, AXIS_LT) + 1.0) * 0.5
            rt = (safe_get_axis(js, AXIS_RT) + 1.0) * 0.5

            # ---------- 3. 更新 mocap target: 用 frame_dt,不用 dt ----------
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

            # # ---------- 4. 夹爪 ----------
            # gripper_ctrl += (lt - rt) * GRIPPER_SPEED * frame_dt
            # gripper_ctrl = float(np.clip(gripper_ctrl, 0.0, 255.0))
            # data.ctrl[gripper_act_id] = gripper_ctrl

            # ---------- 5. 一次外层循环里跑多步仿真,追上墙钟 ----------
            n_sub = int(min(MAX_SUBSTEPS, max(1, round(frame_dt / dt))))
            for _ in range(n_sub):
                # ---- 差分 IK ----
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

            # ---------- 6. 渲染降频:只在到期时做 ----------
            if now - last_render_time >= render_period:
                canvas_bgr = render_three_cameras(renderer, data, camera_names)
                cv2.imshow("MuJoCo Multi-Camera", canvas_bgr)
                last_render_time = now

            # waitKey(1) 无论有没有新帧都要调,否则窗口不响应;但不再靠它做节拍
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                running = False



    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()