"""
UR5e + Robotiq 2F-85  ACT 推理脚本（基于 LeRobot v0.4.x）
==========================================================

用法
-----
    python mujoco_act_infer.py \
        --policy-path outputs/train/ur5e_act/checkpoints/last/pretrained_model \
        --dataset-repo-id local/ur5e_pick_place \
        --dataset-root ./data/ur5e_pick_place \
        --scene scene.xml \
        --device cuda

说明
-----
- 严格对齐 xbox_record_lerobot.py 里的 state/action 约定：
    state  = [6 joints qpos, gripper_norm(0..1)]
    action = [6 joints ctrl,  gripper_norm(0..1)]
- 图像：3 个相机 (scene_cam / top_cam / wrist_cam)，HWC uint8 RGB, 240x320
- 推理频率：30 Hz；MuJoCo 每两个 policy step 之间跑若干 substep
- 按 ESC 退出；按 r 重置环境 + policy（清空 ACT 的 action chunk 队列）
- 按 s 仅 reset policy（不 reset 环境）
- 按 c 随机化方块位置

关键点
-----
ACT 的 select_action 内部维护一个 action chunk 队列：
  - 队列空时，跑一次 forward 生成 chunk_size (默认 100) 个动作
  - 每次调用 pop 一个
  - 切换到新 episode 必须调用 policy.reset() 清空队列
"""

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch


# ============================================================
# 常量：必须与采集脚本保持一致
# ============================================================
RECORD_FPS = 30
IMG_H, IMG_W = 240, 320
CAMERA_NAMES = ["scene_cam", "top_cam", "wrist_cam"]

STATE_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3", "gripper",
]
ACTION_NAMES = list(STATE_NAMES)

DT = 0.002
DRIVER_RANGE_MAX = 0.8  # 与录制脚本保持一致


# ============================================================
# LeRobot 兼容 import
# ============================================================
def _import_lerobot():
    """
    兼容新老版本 LeRobot 的 import 路径。
    新版 (>= 0.3): lerobot.datasets.*, lerobot.policies.*
    老版:          lerobot.common.datasets.*, lerobot.common.policies.*
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        return LeRobotDatasetMetadata, ACTPolicy, make_pre_post_processors
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore
        from lerobot.common.policies.act.modeling_act import ACTPolicy  # type: ignore
        from lerobot.common.policies.factory import make_pre_post_processors  # type: ignore
        return LeRobotDatasetMetadata, ACTPolicy, make_pre_post_processors


# ============================================================
# MuJoCo 渲染 / 显示工具
# ============================================================
def render_three_cameras_rgb(renderer, data, camera_names):
    imgs = []
    for cam_name in camera_names:
        renderer.update_scene(data, camera=cam_name)
        imgs.append(renderer.render().copy())
    return imgs


def stack_canvas_bgr(images_rgb):
    top_row = np.hstack([images_rgb[0], images_rgb[1]])
    blank = np.zeros_like(images_rgb[2])
    bottom_row = np.hstack([images_rgb[2], blank])
    canvas = np.vstack([top_row, bottom_row])
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def draw_overlay(canvas_bgr, step, gripper_ctrl, fps):
    h, w = canvas_bgr.shape[:2]
    cv2.rectangle(canvas_bgr, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(
        canvas_bgr,
        f"step={step}  gripper={gripper_ctrl:.2f}  fps={fps:5.1f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA,
    )
    cv2.rectangle(canvas_bgr, (0, h - 22), (w, h), (0, 0, 0), -1)
    cv2.putText(
        canvas_bgr,
        "ESC: quit    r: reset env+policy    s: reset policy    c: randomize cubes",
        (10, h - 7),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 180), 1, cv2.LINE_AA,
    )
    return canvas_bgr


# ============================================================
# 辅助：随机化方块位置（和录制脚本一致）
# ============================================================
def yaw_to_quat(yaw):
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _set_free_body_pose(model, data, body_name, pos, quat):
    body_id = model.body(body_name).id
    jnt_adr = model.body_jntadr[body_id]
    qpos_adr = model.jnt_qposadr[jnt_adr]
    qvel_adr = model.jnt_dofadr[jnt_adr]
    data.qpos[qpos_adr:qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3:qpos_adr + 7] = quat
    data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def randomize_cubes(model, data):
    table_top_z = 0.42
    cube_half = 0.02
    z = table_top_z + cube_half + 0.002
    center_xy = np.array([0.4, 0.0])
    zone_radius = 0.2

    for body_name in ("cube_red", "cube_green", "cube_blue"):
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        r = (zone_radius - cube_half) * np.sqrt(np.random.uniform(0.0, 1.0))
        xy = center_xy + np.array([r * np.cos(theta), r * np.sin(theta)])
        yaw = np.random.uniform(-np.pi, np.pi)
        _set_free_body_pose(
            model, data, body_name,
            np.array([xy[0], xy[1], z]),
            yaw_to_quat(yaw),
        )
    mujoco.mj_forward(model, data)


# ============================================================
# 构造 raw observation  ->  "inference-ready" observation
# ============================================================
def _prepare_observation_for_inference_fallback(
    observation, device, task=None, robot_type=None,
):
    """
    LeRobot 官方 prepare_observation_for_inference 的本地实现，
    与 lerobot/utils/control_utils.py 里保持一致的语义。

    每个字段都会经过：
        numpy -> torch tensor
        image: type(float32) / 255, permute(H,W,C -> C,H,W)
        unsqueeze(0) 加 batch 维
        .to(device) 搬设备
    """
    out = {}
    for name, value in observation.items():
        if isinstance(value, np.ndarray):
            t = torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            t = value
        else:
            # 其他类型（例如 str）原样透传，后面单独处理
            out[name] = value
            continue

        if "image" in name:
            t = t.type(torch.float32).div(255.0)
            # HWC -> CHW（只有当最后一维是 3 的时候才转）
            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1).contiguous()

        if t.dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            # state 等都应该是 float；int / uint 会让 normalizer 崩
            t = t.float()

        t = t.unsqueeze(0).to(device)
        out[name] = t

    out["task"] = task if task is not None else ""
    out["robot_type"] = robot_type if robot_type is not None else ""
    return out


def _import_prepare_obs():
    """
    优先用 LeRobot 官方的 prepare_observation_for_inference，
    失败则用本地等价实现。
    """
    try:
        from lerobot.utils.control_utils import prepare_observation_for_inference  # type: ignore
        return prepare_observation_for_inference
    except Exception:
        pass
    try:
        from lerobot.common.utils.control_utils import prepare_observation_for_inference  # type: ignore
        return prepare_observation_for_inference
    except Exception:
        pass
    return _prepare_observation_for_inference_fallback


def build_raw_observation(state_vec, images_rgb):
    """
    返回 *未经处理* 的 raw observation（numpy arrays），
    之后会被送进 prepare_observation_for_inference(...) 做标准化处理。
    """
    obs = {
        "observation.state": np.asarray(state_vec, dtype=np.float32),
    }
    for cam_name, img in zip(CAMERA_NAMES, images_rgb):
        # HWC uint8，prepare_observation_for_inference 会自己转 float/255 + permute
        obs[f"observation.images.{cam_name}"] = np.ascontiguousarray(img)
    return obs


def _ensure_batched_action(action) -> np.ndarray:
    """
    后处理结果可能是 tensor、numpy 或 dict，统一成一维 numpy array。
    """
    if isinstance(action, dict):
        # 有些版本 postprocess 返回 dict (含 "action" key)
        if "action" in action:
            action = action["action"]
        else:
            # 取第一个 tensor 值
            action = next(v for v in action.values() if hasattr(v, "shape"))

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    action = np.asarray(action, dtype=np.float32).squeeze()
    if action.ndim == 0:
        raise RuntimeError(f"action 维度异常: {action.shape}")
    return action


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="scene.xml")
    parser.add_argument(
        "--policy-path", type=str, required=True,
        help="训练产出的 pretrained_model 目录，例如 "
             "outputs/train/ur5e_act/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--dataset-repo-id", type=str, default="local/ur5e_pick_place")
    parser.add_argument("--dataset-root", type=str, default="./data/ur5e_pick_place")
    parser.add_argument(
        "--task", type=str,
        default="pick and place the cubes onto the matching-color zones",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--randomize-cubes", action="store_true",
        help="启动时随机化方块位置",
    )
    args = parser.parse_args()

    assert mujoco.__version__ >= "3.1.0", "请升级 mujoco >= 3.1.0"

    LeRobotDatasetMetadata, ACTPolicy, make_pre_post_processors = _import_lerobot()

    # 设备
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA 不可用，回退到 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] {device}")

    # ------------------- 1) 加载 policy -------------------
    print(f"[Policy] 加载 ACT: {args.policy_path}")
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()

    # ------------------- 2) 构造 preprocessor / postprocessor -------------------
    # 优先从 policy path 本地加载保存好的 processors（包含训练时的归一化 stats）
    preprocess, postprocess = None, None
    try:
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            args.policy_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        print("[Processor] 从 policy_path 加载 pre/post processor")
    except Exception as e:
        print(f"[Processor] 从 policy_path 加载失败 ({e})，改从 dataset stats 构造")
        ds_meta = LeRobotDatasetMetadata(
            repo_id=args.dataset_repo_id, root=args.dataset_root,
        )
        # 不同版本的签名略有差异，尝试几种常见写法
        try:
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                dataset_stats=ds_meta.stats,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
        except TypeError:
            preprocess, postprocess = make_pre_post_processors(
                policy_cfg=policy.config,
                dataset_stats=ds_meta.stats,
            )
        print("[Processor] 从 dataset stats 构造 pre/post processor")

    # ------------------- 2.5) 找到官方的 prepare_observation_for_inference -------------------
    prepare_obs_fn = _import_prepare_obs()
    print(f"[Processor] prepare_observation_for_inference = {prepare_obs_fn.__module__}."
          f"{prepare_obs_fn.__name__}")

    # ------------------- 3) MuJoCo 初始化 -------------------
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    model.opt.timestep = DT

    key_id = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    if args.randomize_cubes:
        randomize_cubes(model, data)

    renderer = mujoco.Renderer(model, width=IMG_W, height=IMG_H)
    for name in CAMERA_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) == -1:
            raise RuntimeError(f"Camera not found: {name}")

    # 机械臂 joint / actuator 索引
    joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    dof_ids = np.array([model.joint(n).id for n in joint_names])
    qpos_ids = model.jnt_qposadr[dof_ids]
    jnt_ranges = model.jnt_range[dof_ids]

    actuator_names = [
        "shoulder_pan", "shoulder_lift", "elbow",
        "wrist_1", "wrist_2", "wrist_3",
    ]
    actuator_ids = np.array([model.actuator(n).id for n in actuator_names])
    gripper_act_id = model.actuator("fingers_actuator").id

    # 夹爪真实开合度（用 left_driver_joint qpos）
    driver_joint_id = model.joint("left_driver_joint").id
    driver_qpos_adr = model.jnt_qposadr[driver_joint_id]

    # 把启动关节角写到 ctrl 一次，避免第一步落差过大
    data.ctrl[actuator_ids] = data.qpos[qpos_ids]
    data.ctrl[gripper_act_id] = 0.0

    n_substeps = max(1, int(round((1.0 / RECORD_FPS) / DT)))
    record_period = 1.0 / RECORD_FPS

    # ------------------- 4) 启动 viewer + 推理主循环 -------------------
    cv2.namedWindow("ACT Inference", cv2.WINDOW_NORMAL)
    policy.reset()  # 清空 ACT 的 action chunk 队列

    with mujoco.viewer.launch_passive(
            model=model, data=data,
            show_left_ui=False, show_right_ui=False) as viewer:

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        step = 0
        last_loop_t = time.time()
        fps_ema = 0.0
        last_gripper_ctrl = 0.0

        while viewer.is_running() and step < args.max_steps:
            loop_t0 = time.time()

            # 4.1 渲染 3 路相机（和采集时完全一样的分辨率 / 顺序 / 色彩）
            images_rgb = render_three_cameras_rgb(renderer, data, CAMERA_NAMES)

            # 4.2 构造 state：6 关节 qpos + 夹爪归一化
            gripper_state_norm = float(
                np.clip(data.qpos[driver_qpos_adr] / DRIVER_RANGE_MAX, 0.0, 1.0)
            )
            state_vec = np.concatenate([
                data.qpos[qpos_ids].astype(np.float32),
                np.array([gripper_state_norm], dtype=np.float32),
            ])

            # 4.3 推理
            #   raw_obs: numpy dict（HWC uint8 图像 + float32 state）
            #   -> prepare_observation_for_inference 负责:
            #        numpy->torch, image/255, HWC->CHW, unsqueeze(0), .to(device)
            #   -> preprocess 负责: dataset stats 归一化, 以及可能的 rename / tokenize
            raw_obs = build_raw_observation(state_vec, images_rgb)
            inference_obs = prepare_obs_fn(
                raw_obs,
                device=device,
                task=args.task,
                robot_type="ur5e_2f85",
            )

            if step == 0:
                # 第一步打印各字段的 dtype / shape，方便确认是否 float32
                print("[Debug] inference-ready observation:")
                for k, v in inference_obs.items():
                    if isinstance(v, torch.Tensor):
                        print(f"   {k}: shape={tuple(v.shape)}, "
                              f"dtype={v.dtype}, device={v.device}, "
                              f"range=[{v.min().item():.3f}, {v.max().item():.3f}]")
                    else:
                        print(f"   {k}: {type(v).__name__} = {v!r}")

            with torch.inference_mode():
                batch = preprocess(inference_obs)
                action = policy.select_action(batch)
                action = postprocess(action)
            action_np = _ensure_batched_action(action)

            if action_np.shape[0] < 7:
                raise RuntimeError(
                    f"action 维度不足 7: shape={action_np.shape}，"
                    f"检查训练时的 feature 定义是否和 STATE_NAMES 一致"
                )

            # 4.4 写回 MuJoCo actuator
            arm_action = np.clip(action_np[:6], jnt_ranges[:, 0], jnt_ranges[:, 1])
            gripper_action = float(np.clip(action_np[6], 0.0, 1.0) * 255.0)

            data.ctrl[actuator_ids] = arm_action.astype(np.float64)
            data.ctrl[gripper_act_id] = gripper_action
            last_gripper_ctrl = gripper_action

            if step % 30 == 0:
                print(
                    f"[step {step:4d}] "
                    f"state={np.round(state_vec, 3).tolist()} "
                    f"action={np.round(action_np[:7], 3).tolist()}"
                )

            # 4.5 在两个 policy step 之间跑若干 MuJoCo substep
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)
            viewer.sync()

            # 4.6 显示
            instant_dt = time.time() - loop_t0 + 1e-6
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / instant_dt) if fps_ema > 0 else 1.0 / instant_dt
            canvas_bgr = stack_canvas_bgr(images_rgb)
            canvas_bgr = draw_overlay(canvas_bgr, step, last_gripper_ctrl, fps_ema)
            cv2.imshow("ACT Inference", canvas_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("r"):
                mujoco.mj_resetDataKeyframe(model, data, key_id)
                mujoco.mj_forward(model, data)
                policy.reset()
                step = 0
                print("[Reset] 环境 + policy 全部 reset")
                continue
            elif key == ord("s"):
                policy.reset()
                print("[Reset] policy reset (action chunk 队列清空)")
            elif key == ord("c"):
                randomize_cubes(model, data)
                policy.reset()
                print("[Reset] 方块随机化，policy reset")

            step += 1

            # 控制到 ~30Hz
            elapsed = time.time() - loop_t0
            sleep_t = record_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            last_loop_t = time.time()

    cv2.destroyAllWindows()
    print(f"[Done] 推理结束，共 {step} 步")


if __name__ == "__main__":
    main()