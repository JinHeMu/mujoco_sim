import cv2
import numpy as np
import mujoco

XML_PATH = "scene.xml"   # 你的主场景文件
W, H = 640, 480          # 单个相机画面尺寸

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 创建渲染器
renderer = mujoco.Renderer(model, height=H, width=W)

camera_names = ["scene_cam", "top_cam", "wrist_cam"]

while True:
    mujoco.mj_step(model, data)

    images = []
    for cam_name in camera_names:
        renderer.update_scene(data, camera=cam_name)
        img = renderer.render()   # RGB, shape=(H, W, 3)
        images.append(img)

    # 拼成一张图：上面两个，下面一个
    top_row = np.hstack([images[0], images[1]])
    blank = np.zeros_like(images[2])
    bottom_row = np.hstack([images[2], blank])
    canvas = np.vstack([top_row, bottom_row])

    # MuJoCo 渲染结果通常是 RGB，OpenCV 显示要转 BGR
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imshow("MuJoCo Multi-Camera", canvas_bgr)

    key = cv2.waitKey(1)
    if key == 27:   # ESC 退出
        break

cv2.destroyAllWindows()
