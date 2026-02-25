"""
prepare_batch_jsonl.py

get_dataset.py 로 저장한 cam_third 프레임들을 읽어,
GPT-4o Vision으로 "로봇 손이 물체에 닿아 있는가(contact)" 를 판단하는
OpenAI Batch API 입력 JSONL 파일을 생성합니다.

프레임을 --strip_size 개(기본 3) 씩 묶어 가로로 이어붙인 스트립 이미지 1장을
하나의 요청으로 전송합니다.  어노테이션 결과는 스트립의 **중간 프레임** 기준으로
기록됩니다.

각 JSONL 라인의 custom_id 포맷:
    ep{episode_index:06d}_fr{frame_index:06d}   ← 중간 프레임 인덱스

Output label schema (JSON):
    {
        "left_hand_contact":  true | false | null,
        "right_hand_contact": true | false | null,
        "contact_object":     "<object name or null>",
        "confidence":         "high" | "medium" | "low",
        "reason":             "<brief explanation>"
    }

Usage:
    python prepare_batch_jsonl.py \\
        --frames_meta frames/episodes_meta.json \\
        --out_dir batch \\
        --model gpt-4o \\
        --strip_size 3
"""

import argparse
import base64
import io
import json
import os

from PIL import Image
from tqdm import tqdm

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = """\
You are a robot manipulation expert and computer vision analyst.
You will be shown a strip of {n} consecutive frames (left → right, chronological order)
from a third-person (external) camera recording a robot performing a pick-and-place task.
The robot is a Unitree G1 humanoid robot with dexterous Inspire hands.

Your job is to determine whether the robot's hands are in physical contact
with any object at the moment captured in the **middle frame** of the strip.
Use the surrounding frames only as temporal context to resolve ambiguity.

Return ONLY a JSON object — no extra text, no markdown fences.\
"""

USER_MESSAGE = """\
Task the robot is performing: {task}

The image contains {n} consecutive frames arranged left-to-right.
Focus your answer on the **middle (frame {mid})** of the strip.

Analyze the strip and answer the following questions about the middle frame:

1. Is the robot's **left hand** currently touching / grasping / holding any object?
2. Is the robot's **right hand** currently touching / grasping / holding any object?
3. If any hand is in contact, what object is being touched?

Reply with this exact JSON schema:
{{
  "left_hand_contact":  <true | false | null>,
  "right_hand_contact": <true | false | null>,
  "contact_object":     "<object name, or null if neither hand is in contact>",
  "confidence":         "<high | medium | low>",
  "reason":             "<one short sentence explaining your decision>"
}}

Rules:
- Use `null` for a hand if it is not visible in the middle frame.
- `confidence` reflects how certain you are given image clarity and occlusion.
- Keep `reason` under 30 words.\
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "contact_annotation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "left_hand_contact":  {"type": ["boolean", "null"]},
                "right_hand_contact": {"type": ["boolean", "null"]},
                "contact_object":     {"type": ["string", "null"]},
                "confidence":         {"type": "string", "enum": ["high", "medium", "low"]},
                "reason":             {"type": "string"},
            },
            "required": [
                "left_hand_contact",
                "right_hand_contact",
                "contact_object",
                "confidence",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────


def make_strip_base64(image_paths: list[str], scale: float | None = None) -> tuple[str, str]:
    """여러 이미지를 가로로 이어붙여 base64 JPEG 문자열로 반환합니다.

    Args:
        image_paths: 합성할 이미지 경로 목록
        scale: 0.0 < scale <= 1.0 비율로 각 프레임을 축소. None이면 원본 크기 유지.
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]

    # ── 개별 프레임 리사이즈 (비율 유지) ──────────────────────────────────────
    if scale is not None and scale != 1.0:
        images = [
            im.resize(
                (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                Image.Resampling.LANCZOS,
            )
            for im in images
        ]

    max_h = max(im.height for im in images)
    # 비율 유지하며 높이 통일
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


def build_request(episode_index: int, center_frame_index: int,
                  image_paths: list[str], task: str,
                  model: str,
                  strip_size: int, scale: float | None = None) -> dict:
    b64, mime = make_strip_base64(image_paths, scale=scale)
    mid_label = (strip_size // 2) + 1   # 1-based (e.g. "2" for strip_size=3)

    return {
        "custom_id": f"ep{episode_index:06d}_fr{center_frame_index:06d}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE.format(n=strip_size),
                },
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
    # 메타 로드
    with open(args.frames_meta) as f:
        episodes_meta: list[dict] = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, "batch_input.jsonl")

    total_frames = sum(len(ep["frames"]) for ep in episodes_meta)
    n = args.strip_size
    half = n // 2
    size_info = f"x{args.scale} (비율 축소)" if args.scale and args.scale != 1.0 else "원본 크기"
    print(f"Episodes   : {len(episodes_meta)}")
    print(f"Frames     : {total_frames}")
    print(f"Strip size : {n} (annotation on middle frame)")
    print(f"Frame size : {size_info}")
    print(f"Output     : {jsonl_path}")

    count = 0
    with open(jsonl_path, "w") as fout:
        for ep in tqdm(episodes_meta, desc="Building JSONL"):
            ep_idx = ep["episode_index"]
            task   = ep["task"]
            frames = ep["frames"]  # list of {"frame_index": int, "path": str}

            for i, frame_info in enumerate(frames):
                center_fr_idx = frame_info["frame_index"]

                # 앞뒤 half 개 프레임 수집 (경계는 클리핑)
                indices = [max(0, min(i + d, len(frames) - 1)) for d in range(-half, half + 1)]
                strip_paths = [frames[j]["path"] for j in indices]

                # 하나라도 누락된 경우 스킵
                missing = [p for p in strip_paths if not os.path.exists(p)]
                if missing:
                    for p in missing:
                        print(f"  [WARN] missing frame: {p}")
                    continue

                req = build_request(
                    episode_index=ep_idx,
                    center_frame_index=center_fr_idx,
                    image_paths=strip_paths,
                    task=task,
                    model=args.model,
                    strip_size=n,
                    scale=args.scale,
                )
                fout.write(json.dumps(req) + "\n")
                count += 1

    print(f"\n✅ {count} requests written to {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare OpenAI Batch JSONL for robot hand-contact annotation"
    )
    parser.add_argument(
        "--frames_meta", type=str, default="frames/episodes_meta.json",
        help="get_dataset.py 가 생성한 episodes_meta.json 경로"
    )
    parser.add_argument(
        "--out_dir", type=str, default="batch",
        help="JSONL 출력 디렉토리 (default: batch)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5-mini",
        help="OpenAI 모델 이름 (default: gpt-4o)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="샘플링 temperature (default: 0.0, 결정론적)"
    )
    parser.add_argument(
        "--strip_size", type=int, default=3,
        help="한 요청에 묶을 연속 프레임 수 (홀수 권장, default: 3)"
    )
    parser.add_argument(
        "--scale", type=float, default=None,
        metavar="RATIO",
        help="각 프레임의 축소 비율 (0.0 < RATIO <= 1.0). "
             "예: --scale 0.5 → 가로·세로 절반. 지정하지 않으면 원본 크기 유지."
    )
    args = parser.parse_args()
    main(args)
