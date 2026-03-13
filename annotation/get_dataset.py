"""
get_dataset.py

HuggingFace 데이터셋 `eunjuri/pick_and_place` 에서
RGB 비디오(cam_third)와 hand tactile 이미지를 다운로드하고,
일정 간격으로 프레임을 샘플링하여 저장합니다.

저장 구조:
    frames/
        episode_000000/
            rgb/
                frame_000000.jpg
                frame_000000.jpg
                ...
            tactile/
                left_tactile_thumb_tip/
                    frame_000000.png
                    frame_000000.png
                ...
            tactile_merged/
                frame_000000_left.png
                frame_000000_right.png
                ...

Usage:
    python get_dataset.py --out_dir frames --frame_step 10 --max_episodes 5
    python get_dataset.py --modalities rgb,tactile --frame_step 10
"""

import argparse
import json
import os
from pathlib import Path

import av
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from tqdm import tqdm

from consts import IMAGE_PATH, TACTILE_TO_IMAGE_SHAPE, VERTICES, split_vertice

REPO_ID = "eunjuri/pick_and_place"
CAM_KEY = "observation.images.cam_third"


def get_episode_list(max_episodes: int | None = None) -> list[dict]:
    """episodes.jsonl 에서 에피소드 목록을 가져온다."""
    path = hf_hub_download(repo_id=REPO_ID, filename="meta/episodes.jsonl", repo_type="dataset")
    episodes = []
    with open(path) as f:
        for line in f:
            episodes.append(json.loads(line))
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    return episodes


def get_task_map() -> dict[int, str]:
    """task_index -> task 문자열 매핑을 반환한다."""
    path = hf_hub_download(repo_id=REPO_ID, filename="meta/tasks.jsonl", repo_type="dataset")
    task_map = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            task_map[obj["task_index"]] = obj["task"]
    return task_map


def decode_video_frames(video_path: str, frame_step: int = 10) -> list[tuple[int, Image.Image]]:
    """
    av 라이브러리로 mp4 파일을 디코딩하여 frame_step 간격으로 프레임을 추출한다.
    Returns:
        list of (frame_index, PIL.Image)
    """
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i % frame_step == 0:
                img = frame.to_image()  # PIL.Image (RGB)
                frames.append((i, img))
    return frames


def episode_chunk(episode_index: int, chunks_size: int = 1000) -> int:
    return episode_index // chunks_size


def get_tactile_feature_keys() -> list[str]:
    """info.json 에서 tactile 이미지 feature 키 목록을 가져온다."""
    path = hf_hub_download(repo_id=REPO_ID, filename="meta/info.json", repo_type="dataset")
    with open(path) as f:
        info = json.load(f)

    features = info.get("features", {})
    return sorted(
        [
            key
            for key, spec in features.items()
            if key.startswith("observation.images.")
            and "tactile" in key
            and spec.get("dtype") == "image"
        ]
    )


def parse_modalities(value: str) -> set[str]:
    mods = {v.strip().lower() for v in value.split(",") if v.strip()}
    allowed = {"rgb", "tactile"}
    invalid = mods - allowed
    if invalid:
        raise ValueError(f"Unsupported modalities: {sorted(invalid)} (allowed: {sorted(allowed)})")
    if not mods:
        raise ValueError("At least one modality is required.")
    return mods


def parse_optional_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [v.strip() for v in value.split(",") if v.strip()]
    return parsed if parsed else None


def load_tactile_dataframe(episode_index: int) -> pd.DataFrame:
    """에피소드 parquet 파일에서 tactile image 컬럼이 포함된 프레임 테이블을 로드한다."""
    chunk = episode_chunk(episode_index)
    parquet_filename = f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    parquet_path = hf_hub_download(repo_id=REPO_ID, filename=parquet_filename, repo_type="dataset")
    return pd.read_parquet(parquet_path)


def _sensor_sort_key(sensor_name: str) -> tuple[int, int, str]:
    finger_order = {
        "thumb": 0,
        "index": 1,
        "middle": 2,
        "ring": 3,
        "little": 4,
        "palm": 5,
    }
    part_order = {
        "tip": 0,
        "nail": 1,
        "middle": 2,
        "pad": 3,
        "palm": 4,
    }

    if "_tactile_" not in sensor_name:
        return (99, 99, sensor_name)

    tail = sensor_name.split("_tactile_", 1)[1]
    parts = tail.split("_")
    finger = parts[0] if parts else ""
    part = "_".join(parts[1:]) if len(parts) > 1 else ""
    return (finger_order.get(finger, 99), part_order.get(part, 99), sensor_name)


# 좌표 기반 tactile 배치를 위한 hand local grid 좌표 (x, y)
# 키는 '<finger>_<part>' 형태 (예: thumb_tip, index_nail, palm)
HAND_LOCAL_COORDS: dict[str, tuple[int, int]] = {
    "thumb_tip": (0, 2),
    "thumb_nail": (0, 1),
    "thumb_middle": (1, 2),
    "thumb_pad": (1, 3),
    "index_tip": (3, 0),
    "index_nail": (3, 1),
    "index_pad": (3, 2),
    "middle_tip": (4, 0),
    "middle_nail": (4, 1),
    "middle_pad": (4, 2),
    "ring_tip": (5, 0),
    "ring_nail": (5, 1),
    "ring_pad": (5, 2),
    "little_tip": (6, 0),
    "little_nail": (6, 1),
    "little_pad": (6, 2),
    "palm": (4, 4),
}

HAND_GRID_W = max(x for x, _ in HAND_LOCAL_COORDS.values()) + 1
HAND_GRID_H = max(y for _, y in HAND_LOCAL_COORDS.values()) + 1


def _parse_side_and_local_key(sensor_name: str) -> tuple[str, str] | None:
    # sensor_name 예시: left_tactile_thumb_tip
    if "_tactile_" not in sensor_name:
        return None
    side, tail = sensor_name.split("_tactile_", 1)
    if side not in {"left", "right"}:
        return None
    return side, tail


def _enhance_gray_image(img: Image.Image, contrast: float = 1.8) -> Image.Image:
    """tactile 셀 강도를 보기 쉽게 grayscale+고대비로 보정한다."""
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=1)
    if contrast != 1.0:
        gray = ImageEnhance.Contrast(gray).enhance(contrast)
    return gray


def _contact_color(v: int) -> tuple[int, int, int]:
    """강도값을 배경과 구분되는 색으로 매핑한다 (높을수록 밝은 붉은색)."""
    v = max(0, min(255, int(v)))
    # 저강도는 어두운 회색으로 유지
    threshold = 120
    if v < threshold:
        d = int(v * 0.28)
        return (d, d, d)

    # 고강도는 주황 -> 밝은 빨강으로 전환
    a = (v - threshold) / (255.0 - threshold)
    r = int(210 + 45 * a)
    g = int(70 * (1.0 - a))
    b = int(45 * (1.0 - a))
    return (r, g, b)


def _get_canvas_from_background() -> Image.Image:
    # 저장 시점에 background를 포함한 완성 이미지를 만든다.
    if os.path.exists(IMAGE_PATH):
        return Image.open(IMAGE_PATH).convert("RGB")

    # background가 없으면 vertices 범위 기준 흰 캔버스 생성
    max_x = max(max(v[0] for v in verts) for verts in VERTICES.values()) + 1
    max_y = max(max(v[1] for v in verts) for verts in VERTICES.values()) + 1
    return Image.new("RGB", (max_x, max_y), (255, 255, 255))


def _render_sensor_to_canvas(canvas: Image.Image, sensor_name: str, sensor_path: str) -> None:
    if sensor_name not in VERTICES or sensor_name not in TACTILE_TO_IMAGE_SHAPE:
        return
    if not os.path.exists(sensor_path):
        return

    c, h, w = TACTILE_TO_IMAGE_SHAPE[sensor_name]
    if c != 3:
        return

    with Image.open(sensor_path) as src:
        src_gray = _enhance_gray_image(src).resize((w, h), Image.Resampling.BILINEAR)

    quads = split_vertice(VERTICES[sensor_name], (c, h, w))
    draw = ImageDraw.Draw(canvas)

    # visualize_tactile와 동일하게 셀 단위 색을 영역에 채움
    for r in range(h):
        for col in range(w):
            v = int(src_gray.getpixel((col, r)))
            draw.polygon(quads[r][col], fill=_contact_color(v))


def _draw_sensor_outlines(
    canvas: Image.Image,
    line_color: tuple[int, int, int] = (40, 40, 40),
    line_width: int = 2,
) -> None:
    """센서 영역 외곽선을 덧그려 손/손가락 윤곽을 시각적으로 복원한다."""
    draw = ImageDraw.Draw(canvas)
    for verts in VERTICES.values():
        draw.polygon(verts, outline=line_color, width=line_width)


def save_merged_tactile_image(
    sensor_paths: dict[str, str],
    output_dir: Path,
    frame_index: int,
    scale: int,
    hand_gap_cells: int = 2,
    margin_cells: int = 1,
) -> str:
    """consts.py 좌표/크기 규격으로 tactile 이미지를 손바닥 레이아웃 1장으로 렌더링한다."""
    # legacy 인자는 시그니처 호환을 위해 남겨둠
    _ = (scale, hand_gap_cells, margin_cells)

    valid = [k for k in sensor_paths if k in VERTICES and k in TACTILE_TO_IMAGE_SHAPE]
    if not valid:
        return ""

    canvas = _get_canvas_from_background()
    for sensor_name in sorted(valid):
        _render_sensor_to_canvas(canvas, sensor_name, sensor_paths[sensor_name])

    # tactile 채움 후 외곽선을 덧그려 손가락 윤곽을 명확히 표시
    _draw_sensor_outlines(canvas)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"frame_{int(frame_index):06d}_tactile.png"
    canvas.save(out_path)
    return str(out_path)


def save_tactile_frames(
    df: pd.DataFrame,
    ep_dir: Path,
    frame_indices: list[int],
    tactile_keys: list[str],
    tactile_merge_scale: int,
) -> list[dict]:
    """선택된 frame_indices 에 대해 tactile 이미지 바이트를 센서별 PNG로 저장한다."""
    if "frame_index" not in df.columns:
        raise ValueError("Parquet does not contain 'frame_index' column.")

    by_frame = df.set_index("frame_index", drop=False)
    records: list[dict] = []

    for frame_i in frame_indices:
        if frame_i not in by_frame.index:
            continue

        row = by_frame.loc[frame_i]
        # 혹시 중복 인덱스가 있으면 첫 행만 사용
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        sensor_paths: dict[str, str] = {}
        for key in tactile_keys:
            if key not in row:
                continue

            cell = row[key]
            if not isinstance(cell, dict) or "bytes" not in cell:
                continue

            sensor = key.replace("observation.images.", "")
            sensor_dir = ep_dir / "tactile" / sensor
            sensor_dir.mkdir(parents=True, exist_ok=True)

            ext = Path(cell.get("path", "frame.png")).suffix or ".png"
            img_path = sensor_dir / f"frame_{int(frame_i):06d}{ext}"
            if not img_path.exists():
                with open(img_path, "wb") as f:
                    f.write(cell["bytes"])
            sensor_paths[sensor] = str(img_path)

        if sensor_paths:
            merged_dir = ep_dir / "tactile_merged"
            combined_merged = save_merged_tactile_image(
                sensor_paths=sensor_paths,
                output_dir=merged_dir,
                frame_index=int(frame_i),
                scale=tactile_merge_scale,
            )
            records.append(
                {
                    "frame_index": int(frame_i),
                    "sensors": sensor_paths,
                    "merged": {
                        "combined": combined_merged,
                    },
                }
            )

    return records


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    modalities = parse_modalities(args.modalities)

    # 태스크 매핑 로드
    task_map = get_task_map()

    tactile_keys: list[str] = []
    if "tactile" in modalities:
        tactile_keys = get_tactile_feature_keys()
        requested_tactile_keys = parse_optional_csv(args.tactile_keys)
        if requested_tactile_keys is not None:
            missing = [k for k in requested_tactile_keys if k not in tactile_keys]
            if missing:
                raise ValueError(f"Unknown tactile keys: {missing}")
            tactile_keys = requested_tactile_keys

    # 에피소드 목록 로드
    print(f"Loading episode list from {REPO_ID} ...")
    episodes = get_episode_list(args.max_episodes)
    print(f"  -> {len(episodes)} episodes to process")
    print(f"  -> modalities: {sorted(modalities)}")
    if tactile_keys:
        print(f"  -> tactile sensors: {len(tactile_keys)}")

    # 에피소드별 메타 저장용 (prepare_batch_jsonl.py에서 사용)
    meta_records = []

    for ep in tqdm(episodes, desc="Downloading & sampling frames"):
        ep_idx = ep["episode_index"]
        task_idx = ep.get("task_index", 0)
        task_str = task_map.get(task_idx, "unknown task")
        chunk = episode_chunk(ep_idx)

        frame_paths = []
        sampled_indices: list[int] = []
        tactile_records: list[dict] = []

        ep_dir = Path(args.out_dir) / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        if "rgb" in modalities:
            video_filename = f"videos/chunk-{chunk:03d}/{CAM_KEY}/episode_{ep_idx:06d}.mp4"
            try:
                video_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=video_filename,
                    repo_type="dataset",
                )
            except Exception as e:
                print(f"  [WARN] episode {ep_idx} rgb download failed: {e}")
                video_path = None

            if video_path is not None:
                sampled = decode_video_frames(video_path, frame_step=args.frame_step)
                rgb_dir = ep_dir / "rgb"
                rgb_dir.mkdir(parents=True, exist_ok=True)
                for frame_i, img in sampled:
                    img_path = rgb_dir / f"frame_{frame_i:06d}.jpg"
                    if not img_path.exists():
                        img.save(img_path, quality=90)
                    frame_paths.append({"frame_index": frame_i, "path": str(img_path)})
                sampled_indices = [x[0] for x in sampled]

        if "tactile" in modalities:
            try:
                tactile_df = load_tactile_dataframe(ep_idx)
                if not sampled_indices:
                    sampled_indices = [int(v) for v in tactile_df["frame_index"].tolist()[:: args.frame_step]]
                tactile_records = save_tactile_frames(
                    df=tactile_df,
                    ep_dir=ep_dir,
                    frame_indices=sampled_indices,
                    tactile_keys=tactile_keys,
                    tactile_merge_scale=args.tactile_merge_scale,
                )
            except Exception as e:
                print(f"  [WARN] episode {ep_idx} tactile export failed: {e}")

        meta_records.append({
            "episode_index": ep_idx,
            "task_index": task_idx,
            "task": task_str,
            "modalities": sorted(modalities),
            "frames": frame_paths,
            "tactile_frames": tactile_records,
        })

    # 메타 정보 저장
    meta_path = Path(args.out_dir) / "episodes_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_records, f, indent=2)

    print(f"\nDone! Frames saved to '{args.out_dir}/'")
    print(f"   Meta info: {meta_path}")
    print(f"   Total episodes: {len(meta_records)}")
    total_rgb_frames = sum(len(ep["frames"]) for ep in meta_records)
    total_tactile_frames = sum(len(ep["tactile_frames"]) for ep in meta_records)
    print(f"   Total sampled RGB frames: {total_rgb_frames}")
    print(f"   Total sampled tactile frame-records: {total_tactile_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RGB and tactile frames from eunjuri/pick_and_place")
    parser.add_argument("--out_dir", type=str, default="frames",
                        help="프레임 저장 디렉토리 (default: frames)")
    parser.add_argument("--frame_step", type=int, default=10,
                        help="몇 프레임마다 1장 샘플링 (default: 10, 30fps -> 3fps)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="처리할 최대 에피소드 수 (default: 전체)")
    parser.add_argument(
        "--modalities",
        type=str,
        default="rgb,tactile",
        help="저장할 모달리티. 콤마로 구분 (rgb,tactile). default: rgb,tactile",
    )
    parser.add_argument(
        "--tactile_keys",
        type=str,
        default=None,
        help=(
            "저장할 tactile feature 키 목록(콤마 구분). "
            "예: observation.images.left_tactile_thumb_tip,observation.images.right_tactile_thumb_tip"
        ),
    )
    parser.add_argument(
        "--tactile_merge_scale",
        type=int,
        default=56,
        help="합성 tactile 이미지 확대 배율(원본 3x3 기준, default: 56)",
    )
    args = parser.parse_args()
    main(args)
