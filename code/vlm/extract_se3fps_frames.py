"""
SE(3)-FPS Oracle Frame Extraction
==================================
使用 SE(3) 最远点采样（Farthest Point Sampling）从相机外参轨迹中
提取关键帧，替代基于阈值的 Kinematic Oracle 方案。

核心思想：
    将每一帧的相机位姿 T_i ∈ SE(3) 视为黎曼流形上的一个点，
    使用贪心 FPS 算法迭代选取"距已选帧集合最远"的帧，
    从而保证所选帧在物理空间（平移+旋转）中分布最均匀。
"""

import os
import cv2
import json
import glob
import numpy as np



# ---- 路径配置 ------------------------------------------------
# 包含所有以 _hq 结尾的场景文件夹的根目录
BASE_DIR = os.environ.get("VLM_VIDEO_ROOT", "./videos")

# 所有关键帧图片的统一输出目录
OUTPUT_FRAMES_DIR = os.environ.get("VLM_OUTPUT_FRAMES_DIR", "./frames_se3fps")

# ---- 采样参数 ------------------------------------------------
# 每个视频最多抽取的关键帧数（受 VLM 显存上限约束，通常 8）
MAX_FRAMES = 8

# α：平移距离的权重，(1-α) 为旋转距离的权重，取值范围 (0, 1)
#
# 物理含义：
#   α=1.0 → 纯粹按位移大小选帧，完全忽略朝向变化
#   α=0.0 → 纯粹按旋转角度选帧，完全忽略空间移动
#   α=0.7 → 平移优先，但旋转仍有 30% 的贡献
#
# 选取建议：
#   - 你们的任务核心是 Δd（物理位移），推荐 α=0.7
#   - 若场景以旋转运动为主（如全景扫描），可降至 α=0.5
#   - 注意：归一化后两项量纲一致，α 的效果才真正显现
#     （见下方 D_max 归一化说明）
ALPHA = 0.7

# D_max 的计算方式：用于归一化平移距离
#
# 两种选择：
#   "percentile" → 用轨迹内所有帧对距离的第 95 百分位数作为基准
#                  优点：对异常远帧（跳变）鲁棒
#                  适合：Fast/Teleport 类型轨迹
#   "max"        → 用轨迹内最大帧对距离
#                  优点：简单，保证所有距离都归一化到 [0,1]
#                  缺点：受单个异常帧影响大
#
# 推荐：使用 "percentile"
D_MAX_MODE = "percentile"          # "percentile" 或 "max"
D_MAX_PERCENTILE = 95              # 仅在 D_MAX_MODE="percentile" 时生效

# ==============================================================
# 核心算法模块
# ==============================================================

def compute_D_max(extrinsics: list) -> float:
    """
    计算轨迹内平移距离的归一化基准值 D_max。

    目的：让平移项和旋转项在数值上可比较（都落在 [0,1] 附近），
    否则室内场景中帧间平移动辄几米，会数值上压制归一化到 [0,1] 的旋转项，
    导致 α 的调节实际上失效。

    Args:
        extrinsics: camera-to-world 4x4 矩阵列表

    Returns:
        D_max: 用于归一化的平移基准距离（米）
    """
    translations = np.array([T[:3, 3] for T in extrinsics])  # (N, 3)

    # 计算所有帧对之间的平移距离矩阵
    # diff[i,j] = ||t_i - t_j||_2
    diff = translations[:, None, :] - translations[None, :, :]   # (N, N, 3)
    dist_matrix = np.linalg.norm(diff, axis=-1)                  # (N, N)

    # 只取上三角（避免重复），排除对角线（自身距离为 0）
    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

    if len(upper) == 0:
        return 1.0  # 安全退路：仅一帧时返回 1

    if D_MAX_MODE == "percentile":
        return float(np.percentile(upper, D_MAX_PERCENTILE)) + 1e-6
    else:  # "max"
        return float(np.max(upper)) + 1e-6


def se3_distance(T1: np.ndarray, T2: np.ndarray,
                 alpha: float, D_max: float) -> float:
    """
    计算两帧相机位姿在 SE(3) 上的加权距离：

        d(T_i, T_j) = α · ||t_i - t_j||₂ / D_max
                    + (1-α) · arccos((tr(R_i^T R_j) - 1) / 2) / π

    两项均归一化到 [0, 1]：
        - 平移项：Euclidean 距离除以 D_max
        - 旋转项：旋转角（0°~180°）除以 π（即 180°）

    Args:
        T1, T2  : 4×4 camera-to-world 变换矩阵
        alpha   : 平移权重
        D_max   : 平移归一化基准（由 compute_D_max 计算）

    Returns:
        标量距离值，范围大致在 [0, 1]
    """
    # ---- 平移距离 ------------------------------------------------
    t_dist = np.linalg.norm(T1[:3, 3] - T2[:3, 3]) / D_max
    t_dist = min(t_dist, 1.0)   # clip：防止跳变帧超出 1

    # ---- 旋转距离（SO(3) 测地线角度）-----------------------------
    # R_diff = R_i^T · R_j：从姿态 i 到姿态 j 的相对旋转
    R_diff  = T1[:3, :3].T @ T2[:3, :3]

    # 旋转角公式：cos θ = (tr(R) - 1) / 2
    # clip 防止数值误差导致 arccos 输入超出 [-1, 1]
    cos_ang = float(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
    r_dist  = np.arccos(cos_ang) / np.pi    # 归一化：除以 π，范围 [0, 1]

    return alpha * t_dist + (1.0 - alpha) * r_dist


def fps_se3(extrinsics: list,
            max_frames: int = MAX_FRAMES,
            alpha: float    = ALPHA) -> list:
    """
    SE(3) 流形上的贪心最远点采样（Farthest Point Sampling, FPS）。

    算法流程：
        1. 初始化：强制选第 0 帧和最后一帧，计算所有帧到首尾帧集合的距离
           → minDist[i] = min(d(T_i, T_0), d(T_i, T_{N-1}))
        2. 每轮迭代：
           a. 选 minDist 最大的未选帧 f_new（距已选集合最远）
           b. 将 f_new 加入已选集合
           c. 增量更新：minDist[i] = min(minDist[i], d(T_i, T_f_new))
              ↑ 只需 N 次计算，无需重新计算全部帧对
        3. 重复直到选出 k 帧

    复杂度：O(N·k)，无超参数搜索，无需深度图。

    Args:
        extrinsics  : camera-to-world 4x4 矩阵列表，长度 N
        max_frames  : 目标帧数 k
        alpha       : 平移/旋转权重（见 se3_distance）

    Returns:
        已排序的关键帧索引列表，长度 ≤ max_frames
    """
    N = len(extrinsics)
    if N == 0:
        return []
    if max_frames <= 1:
        return [0]
    if N <= max_frames:
        return list(range(N))   # 帧数不超上限，全部返回
    if max_frames == 2:
        return [0, N - 1]

    # 预计算归一化基准
    D_max = compute_D_max(extrinsics)

    # ---- Step 0：初始化，强制选第 0 帧和最后一帧 ----------------
    selected = [0, N - 1]
    dist_to_first = np.array([
        se3_distance(extrinsics[i], extrinsics[0], alpha, D_max)
        for i in range(N)
    ])
    dist_to_last = np.array([
        se3_distance(extrinsics[i], extrinsics[N - 1], alpha, D_max)
        for i in range(N)
    ])
    min_dists = np.minimum(dist_to_first, dist_to_last)
    min_dists[0] = -np.inf
    min_dists[N - 1] = -np.inf

    # ---- 迭代选帧 -----------------------------------------------
    while len(selected) < max_frames:
        # 选距离当前已选集合最远的帧
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

        # 增量更新：只更新与新选帧的距离
        new_dists = np.array([
            se3_distance(extrinsics[i], extrinsics[next_idx], alpha, D_max)
            for i in range(N)
        ])
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[next_idx] = -np.inf  # 屏蔽刚选的帧

    return sorted(selected)


# ==============================================================
# 数据 I/O 模块
# ==============================================================

def load_extrinsics(pose_file_path: str) -> list:
    """
    读取 video_xxx.json 中的 camera_to_world 外参序列。

    期望的 JSON 格式（与 extract_oracle_frames.py 保持一致）：
    {
        "frames": [
            {"camera_to_world": [[...4x4 矩阵...]]},
            ...
        ]
    }

    ⚠️ 如果你的数据格式不同（例如使用 "transform_matrix" 字段），
       请在此处修改字段名。
    """
    with open(pose_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extrinsics = []
    for frame in data.get('frames', []):
        # ↓ 如果字段名不是 'camera_to_world'，在此修改
        matrix = np.array(frame['camera_to_world'], dtype=np.float64)
        extrinsics.append(matrix)

    return extrinsics


def process_video_to_keyframes(video_path: str, pose_path: str,
                                output_dir: str, scene_id: str) -> list:
    """
    完整的单视频处理流水线：
        读取外参 → SE(3)-FPS 选帧 → 从视频中提取帧 → 保存图片

    Returns:
        保存成功的图片路径列表
    """
    print(f"  🎬 处理: {scene_id}")

    # Step 1：读取外参
    try:
        extrinsics = load_extrinsics(pose_path)
    except Exception as e:
        print(f"  ❌ 外参读取失败: {e}")
        return []

    if len(extrinsics) == 0:
        print(f"  ⚠️  外参列表为空，跳过")
        return []

    # Step 2：SE(3)-FPS 选帧
    keyframe_indices = fps_se3(extrinsics, max_frames=MAX_FRAMES, alpha=ALPHA)
    print(f"  📐 SE(3)-FPS 选帧索引: {keyframe_indices}  "
          f"(共 {len(extrinsics)} 帧 → 选 {len(keyframe_indices)} 帧)")

    # Step 3：创建输出目录
    scene_output_dir = os.path.join(output_dir, scene_id)
    os.makedirs(scene_output_dir, exist_ok=True)

    # Step 4：逐帧读取视频并保存关键帧
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ 无法打开视频: {video_path}")
        return []

    saved_paths = []
    target_set  = set(keyframe_indices)
    max_target  = max(keyframe_indices)
    cur_idx     = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cur_idx > max_target:
            break
        if cur_idx in target_set:
            # 文件名包含原始帧号，方便后续与外参对齐
            save_path = os.path.join(scene_output_dir, f"frame_{cur_idx:04d}.jpg")
            cv2.imwrite(save_path, frame)
            saved_paths.append(save_path)
        cur_idx += 1

    cap.release()

    if len(saved_paths) == len(keyframe_indices):
        print(f"  ✅ 成功保存 {len(saved_paths)} 帧")
    else:
        print(f"  ⚠️  期望 {len(keyframe_indices)} 帧，实际保存 {len(saved_paths)} 帧（视频帧数不足？）")

    return saved_paths


# ==============================================================
# 主执行入口
# ==============================================================

if __name__ == "__main__":
    scene_dirs = glob.glob(os.path.join(BASE_DIR, "*_hq"))

    if not scene_dirs:
        print(f"❌ 在 {BASE_DIR} 下未找到任何 *_hq 文件夹，请检查路径")
        exit(1)

    print(f"🔍 发现 {len(scene_dirs)} 个场景")
    print(f"⚙️  配置: MAX_FRAMES={MAX_FRAMES}, ALPHA={ALPHA}, "
          f"D_MAX_MODE={D_MAX_MODE}\n")

    all_saved_data = {}

    for scene_dir in sorted(scene_dirs):
        scene_id = os.path.basename(scene_dir).replace('_hq', '')
        print(f"{'='*50}\n🏙️  场景: {scene_id}")

        scene_data = {}
        json_files = sorted(glob.glob(os.path.join(scene_dir, "video_*.json")))
        json_files = [
            path for path in json_files
            if not os.path.basename(path).replace(".json", "").endswith("_raw")
        ]

        if not json_files:
            print(f"  ⚠️  未找到 video_*.json 文件，跳过")
            continue

        for pose_path in json_files:
            video_path = pose_path.replace('.json', '.mp4')
            traj_name  = os.path.basename(pose_path).replace('.json', '')

            if not os.path.exists(video_path):
                print(f"  ⚠️  找不到对应视频，跳过: {os.path.basename(video_path)}")
                continue

            saved = process_video_to_keyframes(
                video_path = video_path,
                pose_path  = pose_path,
                output_dir = OUTPUT_FRAMES_DIR,
                scene_id   = f"{scene_id}/{traj_name}"
            )
            scene_data[traj_name] = saved

        all_saved_data[scene_id] = scene_data

    # 保存汇总 JSON（记录所有场景的图片路径，供后续 QA 生成使用）
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_FRAMES_DIR, "se3fps_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_saved_data, f, indent=2, ensure_ascii=False)

    total_scenes = len(all_saved_data)
    total_frames = sum(
        len(paths)
        for scene in all_saved_data.values()
        for paths in scene.values()
    )
    print(f"\n🎉 全部完成！共处理 {total_scenes} 个场景，提取 {total_frames} 帧")
    print(f"📦 汇总文件: {summary_path}")
