"""
prepare_batch_jsonl_tactile_only_active_fingers.py

연속 tactile 합성 이미지(strip)만 보고,
"어떤 손가락/손바닥이 활성화(active)되었는지" 어노테이션하는
OpenAI Batch 입력 JSONL을 생성합니다.

입력 우선순위:
1) tactile_frames[].merged.combined
2) 없으면 스킵

출력 파일:
    batch/batch_input_tactile_only_active_fingers.jsonl

Usage:
    python prepare_batch_jsonl_tactile_only_active_fingers.py
    python prepare_batch_jsonl_tactile_only_active_fingers.py --strip_size 5 --scale 0.8
"""

import argparse
import base64
import io
import json
import os

from PIL import Image
from tqdm import tqdm

SYSTEM_MESSAGE = """\
You are a tactile perception expert.
You are given only hand tactile visualization frames (no RGB camera).
Infer activation/contact pattern at the middle frame using temporal context.

Return ONLY a JSON object that strictly follows the schema.
"""

USER_MESSAGE = """\
You are seeing {n} consecutive tactile visualization frames in chronological order.
Focus on the middle frame (frame {mid}) and use neighboring frames only for temporal disambiguation.

Write ONE concise sentence describing which hand/finger regions appear active at the middle frame.
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "tactile_only_annotation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "annotation_sentence": {"type": "string"},
            },
            "required": ["annotation_sentence"],
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


def build_request(
    episode_index: int,
    center_frame_index: int,
    image_paths: list[str],
    model: str,
    strip_size: int,
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
                            "text": USER_MESSAGE.format(n=strip_size, mid=mid_label),
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

    total_tactile_frames = sum(len(ep.get("tactile_frames", [])) for ep in episodes_meta)
    print(f"Episodes         : {len(episodes_meta)}")
    print(f"Tactile frames   : {total_tactile_frames}")
    print(f"Strip size       : {n}")
    print(f"Output           : {jsonl_path}")

    count = 0
    with open(jsonl_path, "w") as fout:
        for ep in tqdm(episodes_meta, desc="Building JSONL (Tactile only)"):
            ep_idx = ep["episode_index"]
            tactile_frames = ep.get("tactile_frames", [])
            if not tactile_frames:
                continue

            merged_items = []
            for item in tactile_frames:
                merged_path = item.get("merged", {}).get("combined", "")
                if merged_path and os.path.exists(merged_path):
                    merged_items.append({
                        "frame_index": int(item["frame_index"]),
                        "path": merged_path,
                    })

            if not merged_items:
                continue

            for i, item in enumerate(merged_items):
                center_fr_idx = item["frame_index"]
                indices = [max(0, min(i + d, len(merged_items) - 1)) for d in range(-half, half + 1)]
                strip_paths = [merged_items[j]["path"] for j in indices]

                missing = [p for p in strip_paths if not os.path.exists(p)]
                if missing:
                    continue

                req = build_request(
                    episode_index=ep_idx,
                    center_frame_index=center_fr_idx,
                    image_paths=strip_paths,
                    model=args.model,
                    strip_size=n,
                    scale=args.scale,
                )
                fout.write(json.dumps(req) + "\n")
                count += 1

    print(f"\nDone: {count} requests written to {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Batch JSONL: tactile only -> active fingers")
    parser.add_argument("--frames_meta", type=str, default="frames/episodes_meta.json")
    parser.add_argument("--out_dir", type=str, default="batch")
    parser.add_argument("--output_name", type=str, default="batch_input_tactile_only_active_fingers.jsonl")
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--strip_size", type=int, default=3)
    parser.add_argument("--scale", type=float, default=None)
    main(parser.parse_args())
