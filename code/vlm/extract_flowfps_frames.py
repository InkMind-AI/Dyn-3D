"""
Optical-Flow Proxy FPS Frame Extraction
======================================
在没有相机外参的推理设置下，用光流代理距离替代 SE(3) 距离，再执行
最远点采样。脚本与 extract_se3fps_frames.py 保持相同的目录遍历和输出
结构，并强制保留首帧和末帧。

代理距离：
    d_hat(i,j) = alpha * mean(||F_{i->j}||)_center / M
               + (1-alpha) * std(div F_{i->j}) / D

其中 M 和 D 是单个视频内用首帧到候选帧光流估计得到的 95 分位数尺度，
用于让光流幅值项和散度项在数值上可比较。
"""

import os
import cv2
import json
import glob
import time
import numpy as np



BASE_DIR = os.environ.get("VLM_VIDEO_ROOT", "./videos")
OUTPUT_FRAMES_DIR = os.environ.get(
    "VLM_OUTPUT_FRAMES_DIR",
    "./frames_flowfps",
)

MAX_FRAMES = int(os.environ.get("VLM_MAX_FRAMES", "8"))

# 与 SE(3)-FPS 中 alpha 含义一致：更大时更偏向平移代理，更小时更偏向旋转/发散代理。
ALPHA = float(os.environ.get("FLOW_ALPHA", "0.7"))

# 光流在低分辨率上计算，保证大规模抽帧可运行。
RESIZE_MAX_SIDE = int(os.environ.get("FLOW_RESIZE_MAX_SIDE", "224"))

# 中心区域比例，0.5 表示取中心 50% 高宽区域估计平移代理。
CENTER_CROP_RATIO = float(os.environ.get("FLOW_CENTER_CROP_RATIO", "0.5"))

# 尺度估计使用的分位数。
SCALE_PERCENTILE = float(os.environ.get("FLOW_SCALE_PERCENTILE", "95"))

# 默认先把候选帧压到 96 个再做 FPS。对于每段视频只选 8 帧，
# 这比在 900 帧慢视频上全量计算两两光流快很多；首尾帧仍会强制保留。
CANDIDATE_STRIDE = max(1, int(os.environ.get("FLOW_CANDIDATE_STRIDE", "1")))
MAX_CANDIDATES = int(os.environ.get("FLOW_MAX_CANDIDATES", "96"))

# 光流算法：默认 DIS 比 Farneback 快。若当前 OpenCV 不支持 DIS，会自动退回 Farneback。
FLOW_METHOD = os.environ.get("FLOW_METHOD", "dis").strip().lower()

# 默认只计算时间正向光流，速度约为双向对称距离的 2 倍。
# 如需更接近严格对称距离，设置 FLOW_SYMMETRIC=1。
FLOW_SYMMETRIC = os.environ.get("FLOW_SYMMETRIC", "0").strip().lower() in {
    "1", "true", "yes", "y"
}

# 断点续跑：默认开启。完整输出会跳过；半成品会清理 frame_*.jpg 后重跑。
FLOW_RESUME = os.environ.get("FLOW_RESUME", "1").strip().lower() in {
    "1", "true", "yes", "y"
}
FLOW_OVERWRITE = os.environ.get("FLOW_OVERWRITE", "0").strip().lower() in {
    "1", "true", "yes", "y"
}
FLOW_MANIFEST_NAME = os.environ.get("FLOW_MANIFEST_NAME", "flowfps_manifest.json")

_DIS_ESTIMATOR = None


def write_json_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_json_if_exists(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def expected_saved_count(frame_count: int) -> int:
    if frame_count <= 0:
        return 0
    return min(MAX_FRAMES, frame_count)


def output_manifest_path(scene_output_dir: str) -> str:
    return os.path.join(scene_output_dir, FLOW_MANIFEST_NAME)


def list_saved_jpegs(scene_output_dir: str) -> list:
    return sorted(glob.glob(os.path.join(scene_output_dir, "frame_*.jpg")))


def output_is_complete(scene_output_dir: str, frame_count: int) -> tuple:
    """
    判断某个 video 的输出是否已经完整。
    优先相信 manifest；如果没有 manifest，则用帧数量和首尾帧兜底判断。
    """
    expected_count = expected_saved_count(frame_count)
    if expected_count <= 0:
        return False, []

    manifest = load_json_if_exists(output_manifest_path(scene_output_dir))
    if manifest.get("status") == "completed":
        saved_paths = manifest.get("saved_paths") or []
        keyframe_indices = manifest.get("keyframe_indices") or []
        manifest_count_ok = len(saved_paths) == len(keyframe_indices) == expected_count
        files_ok = saved_paths and all(os.path.exists(path) for path in saved_paths)
        if manifest_count_ok and files_ok:
            return True, saved_paths

    jpgs = list_saved_jpegs(scene_output_dir)
    if len(jpgs) != expected_count:
        return False, jpgs

    first_ok = os.path.exists(os.path.join(scene_output_dir, "frame_0000.jpg"))
    last_ok = True
    if frame_count > 1 and expected_count > 1:
        last_ok = os.path.exists(os.path.join(scene_output_dir, f"frame_{frame_count - 1:04d}.jpg"))
    if first_ok and last_ok:
        return True, jpgs
    return False, jpgs


def clear_partial_output(scene_output_dir: str) -> None:
    """
    清理中断留下的半成品，避免新旧图片混在一起。
    只删除当前 video 输出目录下的 frame_*.jpg 和 manifest。
    """
    for path in list_saved_jpegs(scene_output_dir):
        try:
            os.remove(path)
        except OSError:
            pass
    manifest_path = output_manifest_path(scene_output_dir)
    if os.path.exists(manifest_path):
        try:
            os.remove(manifest_path)
        except OSError:
            pass


def get_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def build_candidate_indices(frame_count: int) -> list:
    if frame_count <= 0:
        return []

    candidates = list(range(0, frame_count, CANDIDATE_STRIDE))
    if candidates[-1] != frame_count - 1:
        candidates.append(frame_count - 1)

    if MAX_CANDIDATES > 0 and len(candidates) > MAX_CANDIDATES:
        pick = np.linspace(0, len(candidates) - 1, num=MAX_CANDIDATES)
        candidates = sorted(set(candidates[int(round(i))] for i in pick))
        if candidates[0] != 0:
            candidates.insert(0, 0)
        if candidates[-1] != frame_count - 1:
            candidates.append(frame_count - 1)

    return sorted(set(candidates))


def resize_gray(frame: np.ndarray, max_side: int = RESIZE_MAX_SIDE) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return gray


def compute_optical_flow(gray_a: np.ndarray, gray_b: np.ndarray) -> np.ndarray:
    """
    计算光流。默认优先使用 DIS FAST，速度明显快于 Farneback；
    如果 OpenCV 构建不包含 DISOpticalFlow，则自动回退到 Farneback。
    """
    global _DIS_ESTIMATOR

    if FLOW_METHOD == "dis" and hasattr(cv2, "DISOpticalFlow_create"):
        if _DIS_ESTIMATOR is None:
            _DIS_ESTIMATOR = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        return _DIS_ESTIMATOR.calc(gray_a, gray_b, None)

    return cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def load_candidate_gray_frames(video_path: str, candidate_indices: list) -> dict:
    """
    顺序读取视频，只缓存候选帧的低分辨率灰度图。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    targets = set(candidate_indices)
    max_target = max(candidate_indices) if candidate_indices else -1
    frames = {}
    cur_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cur_idx > max_target:
            break
        if cur_idx in targets:
            frames[cur_idx] = resize_gray(frame)
        cur_idx += 1

    cap.release()
    return frames


def optical_flow_components(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple:
    """
    返回两个光流代理分量：
        1. 中心区域平均光流幅值，近似平移强度；
        2. 光流散度的标准差，近似旋转/视角变化强度。
    """
    flow = compute_optical_flow(gray_a, gray_b)

    h, w = gray_a.shape[:2]
    crop_h = max(1, int(round(h * CENTER_CROP_RATIO)))
    crop_w = max(1, int(round(w * CENTER_CROP_RATIO)))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    center_flow = flow[y0:y0 + crop_h, x0:x0 + crop_w]
    center_mag = float(np.mean(np.linalg.norm(center_flow, axis=-1)))

    du_dx = np.gradient(flow[..., 0], axis=1)
    dv_dy = np.gradient(flow[..., 1], axis=0)
    divergence = du_dx + dv_dy
    div_std = float(np.std(divergence))

    return center_mag, div_std


def estimate_flow_scales(frames: dict, ordered_indices: list) -> tuple:
    """
    用首帧到候选帧的光流分量估计单视频归一化尺度。
    """
    if len(ordered_indices) <= 1:
        return 1.0, 1.0

    first_idx = ordered_indices[0]
    first_frame = frames[first_idx]
    mags, divs = [], []

    for idx in ordered_indices[1:]:
        if idx not in frames:
            continue
        mag, div = optical_flow_components(first_frame, frames[idx])
        mags.append(mag)
        divs.append(div)

    mag_scale = float(np.percentile(mags, SCALE_PERCENTILE)) if mags else 1.0
    div_scale = float(np.percentile(divs, SCALE_PERCENTILE)) if divs else 1.0
    return max(mag_scale, 1e-6), max(div_scale, 1e-6)


def flow_proxy_distance(idx_a: int, idx_b: int, frames: dict,
                        mag_scale: float, div_scale: float,
                        cache: dict) -> float:
    """
    光流代理距离。默认使用时间正向光流以提高速度；如启用
    FLOW_SYMMETRIC=1，则使用 i->j 与 j->i 两个方向分量的平均值。
    """
    if idx_a == idx_b:
        return 0.0

    key = tuple(sorted((idx_a, idx_b)))
    if key in cache:
        return cache[key]

    if FLOW_SYMMETRIC:
        mag_ab, div_ab = optical_flow_components(frames[idx_a], frames[idx_b])
        mag_ba, div_ba = optical_flow_components(frames[idx_b], frames[idx_a])
        mag = 0.5 * (mag_ab + mag_ba)
        div = 0.5 * (div_ab + div_ba)
    else:
        src, dst = key
        mag, div = optical_flow_components(frames[src], frames[dst])

    mag_norm = min(mag / mag_scale, 1.0)
    div_norm = min(div / div_scale, 1.0)
    dist = ALPHA * mag_norm + (1.0 - ALPHA) * div_norm
    cache[key] = float(dist)
    return cache[key]


def fps_flow(frames: dict, max_frames: int = MAX_FRAMES) -> list:
    """
    在光流代理距离上执行 FPS，并强制保留首尾帧。
    """
    ordered_indices = sorted(frames.keys())
    N = len(ordered_indices)
    if N == 0:
        return []
    if max_frames <= 1:
        return [ordered_indices[0]]
    if N <= max_frames:
        return ordered_indices
    if max_frames == 2:
        return [ordered_indices[0], ordered_indices[-1]]

    first_idx = ordered_indices[0]
    last_idx = ordered_indices[-1]
    selected = [first_idx, last_idx]

    mag_scale, div_scale = estimate_flow_scales(frames, ordered_indices)
    cache = {}

    min_dists = {}
    for idx in ordered_indices:
        if idx in selected:
            min_dists[idx] = -np.inf
        else:
            d_first = flow_proxy_distance(idx, first_idx, frames, mag_scale, div_scale, cache)
            d_last = flow_proxy_distance(idx, last_idx, frames, mag_scale, div_scale, cache)
            min_dists[idx] = min(d_first, d_last)

    while len(selected) < max_frames:
        next_idx = max(min_dists, key=min_dists.get)
        if not np.isfinite(min_dists[next_idx]):
            break
        selected.append(next_idx)
        min_dists[next_idx] = -np.inf

        for idx in ordered_indices:
            if idx in selected:
                continue
            new_dist = flow_proxy_distance(idx, next_idx, frames, mag_scale, div_scale, cache)
            min_dists[idx] = min(min_dists[idx], new_dist)

    return sorted(selected)


def save_selected_frames(video_path: str, keyframe_indices: list,
                         scene_output_dir: str) -> list:
    os.makedirs(scene_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ 无法打开视频: {video_path}")
        return []

    saved_paths = []
    target_set = set(keyframe_indices)
    max_target = max(keyframe_indices) if keyframe_indices else -1
    cur_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cur_idx > max_target:
            break
        if cur_idx in target_set:
            save_path = os.path.join(scene_output_dir, f"frame_{cur_idx:04d}.jpg")
            cv2.imwrite(save_path, frame)
            saved_paths.append(save_path)
        cur_idx += 1

    cap.release()
    return saved_paths


def process_video_to_keyframes(video_path: str, output_dir: str,
                               scene_id: str) -> list:
    print(f"  🎬 处理: {scene_id}")

    frame_count = get_video_frame_count(video_path)
    if frame_count <= 0:
        print(f"  ⚠️  无法读取帧数，跳过: {video_path}")
        return []

    scene_output_dir = os.path.join(output_dir, scene_id)
    if FLOW_RESUME and not FLOW_OVERWRITE:
        is_complete, saved_paths = output_is_complete(scene_output_dir, frame_count)
        if is_complete:
            print(f"  ⏭️  已完成，跳过: {len(saved_paths)} 帧")
            return saved_paths
        if saved_paths:
            print(f"  ♻️  检测到半成品 {len(saved_paths)} 帧，清理后重跑")
            clear_partial_output(scene_output_dir)
    elif FLOW_OVERWRITE:
        clear_partial_output(scene_output_dir)

    candidate_indices = build_candidate_indices(frame_count)
    frames = load_candidate_gray_frames(video_path, candidate_indices)
    if not frames:
        print(f"  ⚠️  无可用候选帧，跳过")
        return []

    keyframe_indices = fps_flow(frames, MAX_FRAMES)
    print(f"  🌊 Flow-FPS 选帧索引: {keyframe_indices} "
          f"(共 {frame_count} 帧，候选 {len(candidate_indices)} 帧 "
          f"→ 选 {len(keyframe_indices)} 帧，固定首尾)")

    saved_paths = save_selected_frames(video_path, keyframe_indices, scene_output_dir)

    if len(saved_paths) == len(keyframe_indices):
        print(f"  ✅ 成功保存 {len(saved_paths)} 帧")
        manifest = {
            "status": "completed",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_path": video_path,
            "output_dir": scene_output_dir,
            "frame_count": frame_count,
            "candidate_count": len(candidate_indices),
            "keyframe_indices": keyframe_indices,
            "saved_paths": saved_paths,
            "config": {
                "max_frames": MAX_FRAMES,
                "alpha": ALPHA,
                "resize_max_side": RESIZE_MAX_SIDE,
                "center_crop_ratio": CENTER_CROP_RATIO,
                "scale_percentile": SCALE_PERCENTILE,
                "candidate_stride": CANDIDATE_STRIDE,
                "max_candidates": MAX_CANDIDATES,
                "flow_method": FLOW_METHOD,
                "flow_symmetric": FLOW_SYMMETRIC,
            },
        }
        write_json_atomic(output_manifest_path(scene_output_dir), manifest)
    else:
        print(f"  ⚠️  期望 {len(keyframe_indices)} 帧，实际保存 {len(saved_paths)} 帧")

    return saved_paths


if __name__ == "__main__":
    scene_dirs = glob.glob(os.path.join(BASE_DIR, "*_hq"))

    if not scene_dirs:
        print(f"❌ 在 {BASE_DIR} 下未找到任何 *_hq 文件夹，请检查路径")
        exit(1)

    print(f"🔍 发现 {len(scene_dirs)} 个场景")
    print(f"⚙️  配置: MAX_FRAMES={MAX_FRAMES}, ALPHA={ALPHA}, "
          f"RESIZE_MAX_SIDE={RESIZE_MAX_SIDE}, "
          f"FLOW_METHOD={FLOW_METHOD}, FLOW_SYMMETRIC={FLOW_SYMMETRIC}, "
          f"CANDIDATE_STRIDE={CANDIDATE_STRIDE}, "
          f"MAX_CANDIDATES={MAX_CANDIDATES}, "
          f"RESUME={FLOW_RESUME}, OVERWRITE={FLOW_OVERWRITE}, "
          f"OUTPUT={OUTPUT_FRAMES_DIR}\n")

    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_FRAMES_DIR, "flowfps_summary.json")
    all_saved_data = load_json_if_exists(summary_path)

    for scene_dir in sorted(scene_dirs):
        scene_id = os.path.basename(scene_dir).replace("_hq", "")
        print(f"{'='*50}\n🏙️  场景: {scene_id}")

        scene_data = all_saved_data.get(scene_id, {})
        video_files = sorted(glob.glob(os.path.join(scene_dir, "video_*.mp4")))
        video_files = [
            path for path in video_files
            if not os.path.basename(path).replace(".mp4", "").endswith("_raw")
        ]

        if not video_files:
            print(f"  ⚠️  未找到 video_*.mp4 文件，跳过")
            continue

        for video_path in video_files:
            traj_name = os.path.basename(video_path).replace(".mp4", "")
            saved = process_video_to_keyframes(
                video_path=video_path,
                output_dir=OUTPUT_FRAMES_DIR,
                scene_id=f"{scene_id}/{traj_name}",
            )
            scene_data[traj_name] = saved
            all_saved_data[scene_id] = scene_data
            write_json_atomic(summary_path, all_saved_data)

        all_saved_data[scene_id] = scene_data

    write_json_atomic(summary_path, all_saved_data)

    total_scenes = len(all_saved_data)
    total_frames = sum(
        len(paths)
        for scene in all_saved_data.values()
        for paths in scene.values()
    )
    print(f"\n🎉 全部完成！共处理 {total_scenes} 个场景，提取 {total_frames} 帧")
    print(f"📦 汇总文件: {summary_path}")
