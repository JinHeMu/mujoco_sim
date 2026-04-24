# mujoco_sim + ACT

本项目使用ur5e + 2f_85夹爪，在mujoco中进行仿真

使用lerobot环境训练ACT策略，在mujoco中进行推理策略仿真

使用方法：

1. 采集数据集

```bash
  python xbox_record_lerobot.py \
      --repo-id local/ur5e_pick_place \
      --root ./data/ur5e_pick_place \
      --task "pick and place the cubes onto the matching-color zones" \
      --scene scene.xml
```

2. 使用lerobot环境进行训练

```bash
lerobot-train \
  --dataset.repo_id=local/ur5e_pick_place \
  --dataset.root=./data/ur5e_pick_place \
  --policy.type=act \
  --output_dir=outputs/train/ur5e_act \
  --job_name=ur5e_act \
  --policy.device=cuda \
  --policy.push_to_hub=false
```

3. 仿真推理

```bash
    python mujoco_act_infer.py \
        --policy-path outputs/train/ur5e_act/checkpoints/last/pretrained_model \
        --dataset-repo-id local/ur5e_pick_place \
        --dataset-root ./data/ur5e_pick_place \
        --scene scene.xml \
        --device cuda
```

