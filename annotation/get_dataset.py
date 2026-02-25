"""
get_dataset.py

HuggingFace 데이터셋 `eunjuri/pick_and_place` 에서
cam_third (3rd person camera) 비디오를 다운로드하고
일정 간격으로 프레임을 샘플링하여 JPEG 이미지로 저장합니다.

저장 구조:
    frames/
        episode_000000/
            frame_000000.jpg
            frame_000010.jpg
            ...
        episode_000001/
            ...

Usage:
    python get_dataset.py --out_dir frames --frame_step 10 --max_episodes 5
"""

import argparse
import json
import os
from pathlib import Path

import av
from huggingface_hub import hf_hub_download, HfApi
from PIL import Image
from tqdm import tqdm

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


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # 태스크 매핑 로드
    task_map = get_task_map()

    # 에피소드 목록 로드
    print(f"Loading episode list from {REPO_ID} ...")
    episodes = get_episode_list(args.max_episodes)
    print(f"  → {len(episodes)} episodes to process")

    # 에피소드별 메타 저장용 (prepare_batch_jsonl.py에서 사용)
    meta_records = []

    for ep in tqdm(episodes, desc="Downloading & sampling frames"):
        ep_idx = ep["episode_index"]
        task_idx = ep.get("task_index", 0)
        task_str = task_map.get(task_idx, "unknown task")
        chunk = episode_chunk(ep_idx)

        video_filename = f"videos/chunk-{chunk:03d}/{CAM_KEY}/episode_{ep_idx:06d}.mp4"

        try:
            video_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=video_filename,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  [WARN] episode {ep_idx}: {e}")
            continue

        # 프레임 디렉토리
        ep_dir = Path(args.out_dir) / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # 프레임 샘플링 & 저장
        sampled = decode_video_frames(video_path, frame_step=args.frame_step)
        frame_paths = []
        for frame_i, img in sampled:
            img_path = ep_dir / f"frame_{frame_i:06d}.jpg"
            if not img_path.exists():
                img.save(img_path, quality=90)
            frame_paths.append({"frame_index": frame_i, "path": str(img_path)})

        meta_records.append({
            "episode_index": ep_idx,
            "task_index": task_idx,
            "task": task_str,
            "frames": frame_paths,
        })

    # 메타 정보 저장
    meta_path = Path(args.out_dir) / "episodes_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_records, f, indent=2)

    print(f"\n✅ Done! Frames saved to '{args.out_dir}/'")
    print(f"   Meta info: {meta_path}")
    print(f"   Total episodes: {len(meta_records)}")
    total_frames = sum(len(ep["frames"]) for ep in meta_records)
    print(f"   Total sampled frames: {total_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download cam_third frames from eunjuri/pick_and_place")
    parser.add_argument("--out_dir", type=str, default="frames",
                        help="프레임 저장 디렉토리 (default: frames)")
    parser.add_argument("--frame_step", type=int, default=10,
                        help="몇 프레임마다 1장 샘플링 (default: 10, 30fps → 3fps)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="처리할 최대 에피소드 수 (default: 전체)")
    args = parser.parse_args()
    main(args)
