"""
prepare_batch_jsonl_tactile_gt_from_rgb.py

Task + RGB 연속 프레임 스트립을 보고,
"현재 시점에서 어떤 tactile 패턴이 정답(ground truth)으로 기대되는지"를
텍스트/구조화 JSON으로 어노테이션하기 위한 OpenAI Batch 입력 JSONL을 생성합니다.

출력 파일:
    batch/batch_input_tactile_gt_from_rgb.jsonl

Usage:
    python prepare_batch_jsonl_tactile_gt_from_rgb.py
    python prepare_batch_jsonl_tactile_gt_from_rgb.py --strip_size 5 --scale 0.7
"""

import argparse
import base64
import io
import json
import os

from PIL import Image
from tqdm import tqdm

SYSTEM_MESSAGE = """\
You are an expert robotic manipulation analyst.
You are given consecutive RGB frames from a third-person camera and the task description.
Infer the expected hand tactile ground-truth at the moment of the middle frame.

Return ONLY a JSON object following the provided schema.
"""

USER_MESSAGE = """\
Task: {task}
Episode progress: {progress_label}

You are seeing {n} consecutive RGB frames in chronological order.
Focus on the middle frame (frame {mid}) while using surrounding frames for motion context.

Write ONE concise sentence that describes the expected tactile ground truth at this moment.
The sentence should mention key fingers/hands only if relevant.
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "tactile_gt_from_rgb",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "gt_sentence": {"type": "string"},
            },
            "required": ["gt_sentence"],
            "additionalProperties": False,
        },
    },
}


def make_strip_base64(image_paths: list[str], scale: float | None = None) -> tuple[str, str]:
    images = [Image.open(p).convert("RGB") for p in image_paths]

    if scale is not None and scale != 1.0:
        images = [
            im.resize(
                (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                Image.Resampling.LANCZOS,
            )
            for im in images
        ]

    max_h = max(im.height for im in images)
    resized = []
    for im in images:
        if im.height != max_h:
            w = int(im.width * max_h / im.height)
            im = im.resize((w, max_h), Image.Resampling.LANCZOS)
        resized.append(im)

    total_w = sum(im.width for im in resized)
    strip = Image.new("RGB", (total_w, max_h))

    x = 0
    for im in resized:
        strip.paste(im, (x, 0))
        x += im.width

    buf = io.BytesIO()
    strip.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


def progress_label(i: int, total: int) -> str:
    if total <= 1:
        return "single-step (100%)"

    ratio = i / (total - 1)
    pct = int(ratio * 100)
    if ratio < 0.2:
        stage = "early"
    elif ratio < 0.8:
        stage = "middle"
    else:
        stage = "late"
    return f"{stage} stage ({pct}%)"


def build_request(
    episode_index: int,
    center_frame_index: int,
    image_paths: list[str],
    task: str,
    model: str,
    strip_size: int,
    p_label: str,
    scale: float | None = None,
) -> dict:
    b64, mime = make_strip_base64(image_paths, scale=scale)
    mid_label = (strip_size // 2) + 1

    return {
        "custom_id": f"ep{episode_index:06d}_fr{center_frame_index:06d}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": USER_MESSAGE.format(
                                task=task,
                                progress_label=p_label,
                                n=strip_size,
                                mid=mid_label,
                            ),
                        },
                    ],
                },
            ],
            "response_format": RESPONSE_FORMAT,
        },
    }


def main(args):
    with open(args.frames_meta) as f:
        episodes_meta: list[dict] = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, args.output_name)

    n = args.strip_size
    if n % 2 == 0:
        raise ValueError("--strip_size must be odd.")
    half = n // 2

    total_frames = sum(len(ep.get("frames", [])) for ep in episodes_meta)
    print(f"Episodes   : {len(episodes_meta)}")
    print(f"RGB Frames : {total_frames}")
    print(f"Strip size : {n}")
    print(f"Output     : {jsonl_path}")

    count = 0
    with open(jsonl_path, "w") as fout:
        for ep in tqdm(episodes_meta, desc="Building JSONL (Task+RGB->Tactile GT)"):
            ep_idx = ep["episode_index"]
            task = ep.get("task", "")
            frames = ep.get("frames", [])
            if not frames:
                continue

            for i, frame_info in enumerate(frames):
                center_fr_idx = frame_info["frame_index"]
                indices = [max(0, min(i + d, len(frames) - 1)) for d in range(-half, half + 1)]
                strip_paths = [frames[j]["path"] for j in indices]
                missing = [p for p in strip_paths if not os.path.exists(p)]
                if missing:
                    continue

                req = build_request(
                    episode_index=ep_idx,
                    center_frame_index=center_fr_idx,
                    image_paths=strip_paths,
                    task=task,
                    model=args.model,
                    strip_size=n,
                    p_label=progress_label(i, len(frames)),
                    scale=args.scale,
                )
                fout.write(json.dumps(req) + "\n")
                count += 1

    print(f"\nDone: {count} requests written to {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Batch JSONL: Task+RGB -> tactile GT text")
    parser.add_argument("--frames_meta", type=str, default="frames/episodes_meta.json")
    parser.add_argument("--out_dir", type=str, default="batch")
    parser.add_argument("--output_name", type=str, default="batch_input_tactile_gt_from_rgb.jsonl")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--strip_size", type=int, default=3)
    parser.add_argument("--scale", type=float, default=None)
    main(parser.parse_args())

