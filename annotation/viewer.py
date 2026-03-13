"""
viewer.py - Batch JSONL + Dataset 뷰어

기능:
1) batch 모드: 기존 batch_input.jsonl 뷰어
2) dataset 모드: episodes_meta.json 기반 RGB + 좌/우 손 tactile 시각화
   - tactile은 센서 이미지를 좌/우 손별로 이어붙여 2장으로 표시

Usage:
    python annotation/viewer.py --mode dataset
    python annotation/viewer.py --mode dataset --frames_meta annotation/frames/episodes_meta.json
    python annotation/viewer.py --mode batch --jsonl annotation/batch/batch_input.jsonl
"""

import argparse
import base64
import io
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from consts import IMAGE_PATH, TACTILE_TO_IMAGE_SHAPE, VERTICES, split_vertice

# ── 전역 상태 ──────────────────────────────────────────────────────────────────
RECORDS: list[dict] = []
VIEW_MODE = "dataset"
TACTILE_DISPLAY_SCALE = 0.8
GT_TEXT_BY_ID: dict[str, str] = {}
ANNOTATION_TEXT_BY_ID: dict[str, str] = {}
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENABLE_SIMILARITY = True
_EMBEDDER = None
_EMBEDDER_ERROR = ""
TEXT_SIMILARITY_BY_ID: dict[str, float] = {}


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dataset_records(frames_meta_path: str) -> list[dict]:
    """episodes_meta.json 을 프레임 단위 레코드로 펼친다."""
    with open(frames_meta_path, encoding="utf-8") as f:
        episodes = json.load(f)

    records: list[dict] = []
    for ep in episodes:
        ep_idx = ep.get("episode_index", -1)
        task = ep.get("task", "")

        rgb_by_idx = {
            int(item["frame_index"]): item.get("path", "")
            for item in ep.get("frames", [])
            if "frame_index" in item
        }
        tactile_by_idx = {
            int(item["frame_index"]): item.get("sensors", {})
            for item in ep.get("tactile_frames", [])
            if "frame_index" in item
        }
        tactile_merged_by_idx = {
            int(item["frame_index"]): item.get("merged", {})
            for item in ep.get("tactile_frames", [])
            if "frame_index" in item
        }

        gt_by_idx = _build_optional_text_by_idx(ep, "gt_sentence")
        anno_by_idx = _build_optional_text_by_idx(ep, "annotation_sentence")

        frame_indices = sorted(set(rgb_by_idx.keys()) | set(tactile_by_idx.keys()) | set(tactile_merged_by_idx.keys()))
        for frame_idx in frame_indices:
            custom_id = f"ep{ep_idx:06d}_fr{frame_idx:06d}"
            records.append(
                {
                    "mode": "dataset",
                    "custom_id": custom_id,
                    "episode_index": ep_idx,
                    "frame_index": frame_idx,
                    "task": task,
                    "rgb_path": rgb_by_idx.get(frame_idx, ""),
                    "tactile_sensors": tactile_by_idx.get(frame_idx, {}),
                    "tactile_merged": tactile_merged_by_idx.get(frame_idx, {}),
                    "gt_text": GT_TEXT_BY_ID.get(custom_id, gt_by_idx.get(frame_idx, "")),
                    "annotation_text": ANNOTATION_TEXT_BY_ID.get(custom_id, anno_by_idx.get(frame_idx, "")),
                }
            )

    return records


def _enhance_gray_image(img: Image.Image, contrast: float = 1.8) -> Image.Image:
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=1)
    if contrast != 1.0:
        gray = ImageEnhance.Contrast(gray).enhance(contrast)
    return gray


def _contact_color(v: int) -> tuple[int, int, int]:
    v = max(0, min(255, int(v)))
    threshold = 120
    if v < threshold:
        d = int(v * 0.28)
        return (d, d, d)
    a = (v - threshold) / (255.0 - threshold)
    r = int(210 + 45 * a)
    g = int(70 * (1.0 - a))
    b = int(45 * (1.0 - a))
    return (r, g, b)


def _draw_sensor_outlines(
    canvas: Image.Image,
    line_color: tuple[int, int, int] = (40, 40, 40),
    line_width: int = 2,
) -> None:
    draw = ImageDraw.Draw(canvas)
    for verts in VERTICES.values():
        draw.polygon(verts, outline=line_color, width=line_width)


def _background_canvas(size: tuple[int, int]) -> Image.Image:
    if os.path.exists(IMAGE_PATH):
        bg = Image.open(IMAGE_PATH).convert("RGB")
        if bg.size != size:
            bg = bg.resize(size, Image.Resampling.BILINEAR)
        return bg
    return Image.new("RGB", size, (255, 255, 255))


def _to_contact_highlight_rgb(img: Image.Image) -> Image.Image:
    """background.png 위에 tactile polygon만 접촉 강조 색으로 칠한다."""
    out = _background_canvas(img.size)
    gray = _enhance_gray_image(img)
    draw = ImageDraw.Draw(out)

    for sensor_name, verts in VERTICES.items():
        if sensor_name not in TACTILE_TO_IMAGE_SHAPE:
            continue
        c, h, w = TACTILE_TO_IMAGE_SHAPE[sensor_name]
        if c != 3:
            continue

        quads = split_vertice(verts, (c, h, w))
        for r in range(h):
            for col in range(w):
                quad = quads[r][col]
                cx = int(sum(p[0] for p in quad) / 4)
                cy = int(sum(p[1] for p in quad) / 4)
                cx = max(0, min(cx, gray.width - 1))
                cy = max(0, min(cy, gray.height - 1))
                v = int(gray.getpixel((cx, cy)))
                draw.polygon(quad, fill=_contact_color(v))

    _draw_sensor_outlines(out)
    return out


def image_file_to_data_url(path: str, as_gray_highlight: bool = False) -> tuple[str, str]:
    if not path or not os.path.exists(path):
        return "", "image/jpeg"

    with Image.open(path) as im:
        if as_gray_highlight:
            out = _to_contact_highlight_rgb(im)
            fmt = "PNG"
            mime = "image/png"
            buf = io.BytesIO()
            out.save(buf, format=fmt)
            return base64.b64encode(buf.getvalue()).decode("utf-8"), mime

    ext = Path(path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


def _sensor_sort_key(sensor_name: str) -> tuple[int, int, str]:
    # legacy helper (unused)
    return (0, 0, sensor_name)


def build_hand_strip(sensor_paths: dict[str, str], side: str, scale: int = 56) -> tuple[str, str, int]:
    # legacy helper (unused)
    _ = (sensor_paths, side, scale)
    return "", "image/png", 0


def _make_tactile_canvas() -> Image.Image:
    max_x = max(max(v[0] for v in verts) for verts in VERTICES.values()) + 1
    max_y = max(max(v[1] for v in verts) for verts in VERTICES.values()) + 1
    return _background_canvas((max_x, max_y))


def build_combined_tactile_image(sensor_paths: dict[str, str], scale: int = 56) -> tuple[str, str, int]:
    """consts.py 기준(좌표/shape)으로 tactile 이미지를 단일 손바닥 레이아웃으로 합성한다."""
    _ = scale  # consts 기반 렌더에서는 scale 미사용

    valid = [name for name, path in sensor_paths.items() if name in VERTICES and name in TACTILE_TO_IMAGE_SHAPE and os.path.exists(path)]
    if not valid:
        return "", "image/png", 0

    canvas = _make_tactile_canvas()
    draw = ImageDraw.Draw(canvas)

    for sensor_name in sorted(valid):
        c, h, w = TACTILE_TO_IMAGE_SHAPE[sensor_name]
        if c != 3:
            continue

        with Image.open(sensor_paths[sensor_name]) as src:
            src_gray = _enhance_gray_image(src).resize((w, h), Image.Resampling.BILINEAR)

        quads = split_vertice(VERTICES[sensor_name], (c, h, w))
        for r in range(h):
            for col in range(w):
                v = int(src_gray.getpixel((col, r)))
                draw.polygon(quads[r][col], fill=_contact_color(v))

    _draw_sensor_outlines(canvas)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/png", len(valid)


def _parse_ep_fr_from_custom_id(custom_id: str) -> tuple[str, str]:
    # 형식: ep000123_fr000456
    try:
        parts = custom_id.split("_")
        ep = parts[0][2:]
        fr = parts[1][2:]
        if ep.isdigit() and fr.isdigit():
            return str(int(ep)), str(int(fr))
    except Exception:
        pass
    return "-", "-"


def _extract_sentence_from_record(obj: dict, preferred_key: str) -> str:
    # 1) 단순 스키마: {custom_id, gt_sentence/annotation_sentence}
    val = obj.get(preferred_key)
    if isinstance(val, str) and val.strip():
        return val.strip()

    # 2) OpenAI batch output line: response.body.choices[0].message.content
    try:
        content = obj["response"]["body"]["choices"][0]["message"]["content"]
        if isinstance(content, str):
            parsed = json.loads(content)
            inner = parsed.get(preferred_key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    except Exception:
        pass

    return ""


def load_text_map_jsonl(path: str, preferred_key: str) -> dict[str, str]:
    if not path or not os.path.exists(path):
        return {}

    mapped: dict[str, str] = {}
    for rec in load_jsonl(path):
        custom_id = rec.get("custom_id", "")
        if not custom_id:
            continue
        text = _extract_sentence_from_record(rec, preferred_key)
        if text:
            mapped[custom_id] = text
    return mapped


def _build_optional_text_by_idx(ep: dict, key: str) -> dict[int, str]:
    by_idx: dict[int, str] = {}
    for section in ("frames", "tactile_frames"):
        for item in ep.get(section, []):
            if "frame_index" not in item:
                continue
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                by_idx[int(item["frame_index"])] = v.strip()
    return by_idx


def extract_batch_info(record: dict) -> dict:
    custom_id = record.get("custom_id", "")
    parsed_ep, parsed_fr = _parse_ep_fr_from_custom_id(custom_id)
    body = record.get("body", {})
    model = body.get("model", "")
    messages = body.get("messages", [])

    system_text = ""
    image_b64 = ""
    image_mime = "image/jpeg"
    user_text = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_text = content if isinstance(content, str) else ""
        elif role == "user":
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            header, data = url.split(",", 1)
                            image_mime = header.split(":")[1].split(";")[0]
                            image_b64 = data
                    elif part.get("type") == "text":
                        user_text = part.get("text", "")
            else:
                user_text = str(content)

    return {
        "mode": "batch",
        "custom_id": custom_id,
        "model": model,
        "task": "",
        "episode_index": parsed_ep,
        "frame_index": parsed_fr,
        "rgb_b64": image_b64,
        "rgb_mime": image_mime,
        "left_b64": "",
        "left_mime": "image/png",
        "right_b64": "",
        "right_mime": "image/png",
        "tactile_b64": "",
        "tactile_mime": "image/png",
        "tactile_count": 0,
        "user_text": user_text,
        "system_text": system_text,
    }


def _get_embedder():
    global _EMBEDDER, _EMBEDDER_ERROR
    if _EMBEDDER is not None:
        return _EMBEDDER
    if _EMBEDDER_ERROR:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return _EMBEDDER
    except Exception as e:
        _EMBEDDER_ERROR = f"{type(e).__name__}: {e}"
        print(f"[WARN] embedding model load failed: {_EMBEDDER_ERROR}")
        return None


def _compute_text_similarity(gt_text: str, annotation_text: str) -> float | None:
    if not ENABLE_SIMILARITY:
        return None
    if not gt_text or not annotation_text:
        return None

    embedder = _get_embedder()
    if embedder is None:
        return None

    try:
        vectors = embedder.encode([gt_text, annotation_text], normalize_embeddings=True)
        a = vectors[0]
        b = vectors[1]
        # normalize_embeddings=True 이므로 dot product == cosine similarity
        sim = float(sum(float(x) * float(y) for x, y in zip(a, b)))
        return max(-1.0, min(1.0, sim))
    except Exception as e:
        print(f"[WARN] similarity compute failed: {type(e).__name__}: {e}")
        return None


def extract_dataset_info(record: dict) -> dict:
    rgb_b64, rgb_mime = image_file_to_data_url(record.get("rgb_path", ""))
    parsed_ep, parsed_fr = _parse_ep_fr_from_custom_id(record.get("custom_id", ""))
    merged = record.get("tactile_merged", {})
    # 저장된 merged 이미지는 그대로 표시 (뷰어에서 추가 가공하지 않음)
    tactile_b64, tactile_mime = image_file_to_data_url(merged.get("combined", ""))

    sensors = record.get("tactile_sensors", {})
    tactile_count = len(sensors)

    # combined이 없으면 센서 원본으로 좌표 기반 합성
    if not tactile_b64:
        tactile_b64, tactile_mime, fallback_count = build_combined_tactile_image(sensors)
        if tactile_count == 0:
            tactile_count = fallback_count

    gt_text = record.get("gt_text", "") if isinstance(record.get("gt_text", ""), str) else ""
    annotation_text = record.get("annotation_text", "") if isinstance(record.get("annotation_text", ""), str) else ""

    sim = TEXT_SIMILARITY_BY_ID.get(record.get("custom_id", ""))
    if sim is None:
        sim = _compute_text_similarity(gt_text, annotation_text)
        if sim is not None:
            TEXT_SIMILARITY_BY_ID[record.get("custom_id", "")] = sim

    return {
        "mode": "dataset",
        "custom_id": record.get("custom_id", ""),
        "model": "-",
        "task": record.get("task", ""),
        "episode_index": record.get("episode_index", parsed_ep),
        "frame_index": record.get("frame_index", parsed_fr),
        "rgb_b64": rgb_b64,
        "rgb_mime": rgb_mime,
        "tactile_b64": tactile_b64,
        "tactile_mime": tactile_mime,
        "tactile_count": tactile_count,
        "user_text": "",
        "system_text": "",
        "gt_text": gt_text,
        "annotation_text": annotation_text,
        "has_text_annotation": bool(gt_text or annotation_text),
        "text_similarity": sim,
    }


# ── HTML 빌더 ──────────────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; }
header {
    background: #16213e;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    border-bottom: 2px solid #0f3460;
    position: sticky; top: 0; z-index: 100;
}
header h1 { font-size: 1.15rem; color: #e94560; }
.episode-chip {
    font-size: 0.85rem;
    color: #ffd166;
    background: #0d1b2a;
    border: 1px solid #0f3460;
    border-radius: 999px;
    padding: 4px 10px;
}
.nav-btn {
    padding: 6px 18px; border-radius: 6px; border: none;
    background: #0f3460; color: #e0e0e0; cursor: pointer; font-size: 0.9rem;
}
.nav-btn:hover { background: #e94560; }
.nav-btn:disabled { opacity: 0.3; cursor: default; }
#idx-display { font-size: 0.9rem; min-width: 120px; text-align: center; }
#jump-input {
    width: 80px; padding: 5px 8px; border-radius: 6px;
    border: 1px solid #0f3460; background: #16213e; color: #e0e0e0;
}
.main { max-width: 1500px; margin: 24px auto; padding: 0 24px; }
.media-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; }
.card {
    background: #16213e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    border: 1px solid #0f3460;
}
.card h2 { font-size: 0.85rem; color: #e94560; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
.meta-grid { display: flex; gap: 24px; flex-wrap: wrap; }
.meta-item label { font-size: 0.75rem; color: #888; display: block; }
.meta-item span  { font-size: 0.95rem; color: #fff; font-weight: 600; }
.img-wrap { overflow-x: auto; text-align: center; padding: 8px 0; }
.img-wrap img {
    max-height: 520px;
    border-radius: 8px;
    border: 2px solid #0f3460;
    cursor: pointer;
}
.img-wrap img:hover { border-color: #e94560; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
pre {
    white-space: pre-wrap; word-break: break-word;
    background: #0d1b2a; padding: 14px; border-radius: 8px;
    font-size: 0.82rem; line-height: 1.6; color: #b0c4de;
    max-height: 260px; overflow-y: auto;
}
/* GT/Annotation 박스는 항상 같은 높이 + 스크롤바 공간 고정 */
#gt-text, #annotation-text {
    height: 120px;
    max-height: 120px;
    min-height: 120px;
    overflow-y: scroll;
}
#lb { display:none; position:fixed; inset:0; background:rgba(0,0,0,.85); z-index:200; align-items:center; justify-content:center; }
#lb.open { display:flex; }
#lb img { max-width:95vw; max-height:95vh; border-radius:8px; }
#lb-close { position:fixed; top:16px; right:24px; font-size:2rem; cursor:pointer; color:#fff; }
#progress { height: 4px; background: #0f3460; position: fixed; top: 0; left: 0; z-index: 200; transition: width 0.3s; }
@media (max-width: 1100px) { .media-row { grid-template-columns: 1fr; } }
@media (max-width: 1000px) { .grid-2 { grid-template-columns: 1fr; } }
"""

JS = """
let idx = 0;
const total = window.__TOTAL__;

function updateProgress() {
    document.getElementById('progress').style.width = ((idx + 1) / total * 100) + '%';
}

function setImage(elId, b64, mime) {
    const el = document.getElementById(elId);
    if (!b64) {
        el.style.display = 'none';
        return;
    }
    el.style.display = '';
    el.src = `data:${mime};base64,${b64}`;
}

function applyDisplayScale() {
    const tactile = document.getElementById('tactile-img');
    const scale = window.__TACTILE_DISPLAY_SCALE__;
    tactile.style.width = (scale * 100) + '%';
    tactile.style.height = 'auto';
}

function load(i) {
    if (i < 0 || i >= total) return;
    idx = i;
    updateProgress();

    fetch('/record?idx=' + i)
        .then(r => r.json())
        .then(data => {
            document.getElementById('idx-display').textContent = (i + 1) + ' / ' + total;
            document.getElementById('custom-id').textContent = data.custom_id;
            document.getElementById('mode-id').textContent = data.mode;
            document.getElementById('task-id').textContent = data.task || '-';
            document.getElementById('episode-id').textContent = data.episode_index;
            document.getElementById('episode-chip').textContent = 'EP ' + data.episode_index;
            document.getElementById('frame-id').textContent = data.frame_index;
            document.getElementById('model-id').textContent = data.model;
            document.getElementById('tactile-count').textContent = data.tactile_count;

            setImage('rgb-img', data.rgb_b64, data.rgb_mime);
            setImage('tactile-img', data.tactile_b64, data.tactile_mime);

            const isBatch = data.mode === 'batch';
            document.getElementById('prompt-section').style.display = isBatch ? '' : 'none';
            document.getElementById('user-text').textContent = data.user_text || '';
            document.getElementById('system-text').textContent = data.system_text || '';

            const hasText = !!data.has_text_annotation;
            document.getElementById('text-annotation-section').style.display = (data.mode === 'dataset') ? '' : 'none';
            document.getElementById('gt-text').textContent = data.gt_text || '-';
            document.getElementById('annotation-text').textContent = data.annotation_text || '-';

            const simRaw = (typeof data.text_similarity === 'number')
                ? Math.max(-1.0, Math.min(1.0, data.text_similarity))
                : null;
            document.getElementById('similarity-wrap').style.display = (data.mode === 'dataset') ? '' : 'none';

            const leftFill = document.getElementById('similarity-left-fill');
            const rightFill = document.getElementById('similarity-right-fill');
            leftFill.style.width = '0%';
            rightFill.style.width = '0%';

            if (hasText && simRaw !== null) {
                const halfPct = (Math.abs(simRaw) * 50.0).toFixed(2) + '%';
                if (simRaw >= 0) {
                    rightFill.style.width = halfPct;
                    rightFill.style.background = '#16c47f';
                } else {
                    leftFill.style.width = halfPct;
                    leftFill.style.background = '#ff5d73';
                }
                document.getElementById('similarity-label').textContent = simRaw.toFixed(3);
            } else {
                document.getElementById('similarity-label').textContent = 'N/A';
            }

            document.getElementById('jump-input').value = i + 1;
            document.getElementById('btn-prev').disabled = (i === 0);
            document.getElementById('btn-next').disabled = (i === total - 1);
        });
}

document.addEventListener('DOMContentLoaded', () => {
    applyDisplayScale();
    load(0);
    document.getElementById('btn-prev').onclick = () => load(idx - 1);
    document.getElementById('btn-next').onclick = () => load(idx + 1);
    document.getElementById('jump-input').addEventListener('change', e => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v)) load(Math.max(0, Math.min(total - 1, v - 1)));
    });
    document.addEventListener('keydown', e => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') load(idx + 1);
        if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   load(idx - 1);
    });

    document.querySelectorAll('.zoomable').forEach((img) => {
        img.addEventListener('click', () => {
            if (!img.src) return;
            document.getElementById('lb-img').src = img.src;
            document.getElementById('lb').classList.add('open');
        });
    });
    document.getElementById('lb-close').onclick = () =>
        document.getElementById('lb').classList.remove('open');
    document.getElementById('lb').onclick = (e) => {
        if (e.target === document.getElementById('lb'))
            document.getElementById('lb').classList.remove('open');
    };
});
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Viewer</title>
<style>{css}</style>
</head>
<body>
<div id="progress"></div>
<div id="lb"><span id="lb-close">X</span><img id="lb-img" src="" alt="full"></div>

<header>
  <h1>RGB + Hand Tactile Viewer</h1>
  <span id="episode-chip" class="episode-chip">EP -</span>
  <button class="nav-btn" id="btn-prev">Prev</button>
  <span id="idx-display">- / {total}</span>
  <button class="nav-btn" id="btn-next">Next</button>
  <input id="jump-input" type="number" min="1" max="{total}" title="Jump to index" />
  <span style="font-size:0.8rem;color:#888">총 {total}개</span>
</header>

<div class="main">
  <div class="card">
    <h2>Record Info</h2>
    <div class="meta-grid">
      <div class="meta-item"><label>Mode</label><span id="mode-id">-</span></div>
      <div class="meta-item"><label>Custom ID</label><span id="custom-id">-</span></div>
      <div class="meta-item"><label>Task</label><span id="task-id">-</span></div>
      <div class="meta-item"><label>Episode</label><span id="episode-id">-</span></div>
      <div class="meta-item"><label>Frame</label><span id="frame-id">-</span></div>
      <div class="meta-item"><label>Model(batch)</label><span id="model-id">-</span></div>
      <div class="meta-item"><label>Tactile Sensors</label><span id="tactile-count">0</span></div>
    </div>
  </div>

  <div class="media-row">
    <div class="card">
      <h2>RGB Camera</h2>
      <div class="img-wrap">
        <img id="rgb-img" class="zoomable" src="" alt="rgb">
      </div>
    </div>

    <div class="card">
      <h2>Tactile (Palm Layout, Combined)</h2>
      <div class="img-wrap"><img id="tactile-img" class="zoomable" src="" alt="tactile combined"></div>
    </div>
  </div>

  <div class="card" id="text-annotation-section" style="display:none">
    <h2>Text Annotation</h2>
    <h2 style="margin-top:8px">GT Sentence</h2>
    <pre id="gt-text"></pre>
    <h2 style="margin-top:16px">Annotation Sentence</h2>
    <pre id="annotation-text"></pre>
    <div id="similarity-wrap" style="display:none; margin-top:14px;">
      <h2 style="margin-bottom:8px">Embedding Cosine Similarity (-1 ~ 1, centered at 0)</h2>
      <div style="position:relative;background:#0d1b2a;border:1px solid #0f3460;border-radius:999px;height:14px;overflow:hidden;">
        <div id="similarity-left-fill" style="position:absolute;right:50%;top:0;height:100%;width:0%;background:#ff5d73;"></div>
        <div id="similarity-right-fill" style="position:absolute;left:50%;top:0;height:100%;width:0%;background:#16c47f;"></div>
        <div style="position:absolute;left:50%;top:0;transform:translateX(-50%);height:100%;width:2px;background:#9fb4d1;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:0.75rem;color:#9fb4d1;">
        <span>-1.0</span><span>0.0</span><span>1.0</span>
      </div>
      <div style="margin-top:4px;font-size:0.85rem;color:#c6d2ea;">score: <span id="similarity-label">N/A</span></div>
    </div>
  </div>

  <div class="card" id="prompt-section" style="display:none">
    <h2>Batch Prompt</h2>
    <h2 style="margin-top:8px">User</h2>
    <pre id="user-text"></pre>
    <h2 style="margin-top:16px">System</h2>
    <pre id="system-text"></pre>
  </div>
</div>

<script>window.__TOTAL__ = {total};</script>
<script>window.__TACTILE_DISPLAY_SCALE__ = {tactile_scale};</script>
<script>{js}</script>
</body>
</html>
"""


def build_html(total: int, tactile_scale: float) -> str:
    return HTML_TEMPLATE.format(css=CSS, js=JS, total=total, tactile_scale=tactile_scale)


# ── HTTP 핸들러 ────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in {"/", "/index.html"}:
            self.send_html(build_html(len(RECORDS), TACTILE_DISPLAY_SCALE))
            return

        if path == "/record":
            idx = int(query.get("idx", ["0"])[0])
            idx = max(0, min(idx, len(RECORDS) - 1))
            if VIEW_MODE == "batch":
                info = extract_batch_info(RECORDS[idx])
            else:
                info = extract_dataset_info(RECORDS[idx])
            self.send_json(info)
            return

        self.send_response(404)
        self.end_headers()


# ── 메인 ──────────────────────────────────────────────────────────────────────
def _latest_annotation_jsonl_defaults() -> tuple[str, str]:
    """기본 GT/annotation JSONL 경로를 자동 탐색한다.

    우선순위:
    1) annotation/annotations/ 직접 파일
    2) annotation/annotations_parallel 최신 하위폴더
    """
    base = Path(__file__).parent

    candidates: list[tuple[float, str, str]] = []

    # 1) 단일 폴더 모드
    annotations_dir = base / "annotations"
    gt_direct = annotations_dir / "gt_sentences.jsonl"
    anno_direct = annotations_dir / "annotation_sentences.jsonl"
    if gt_direct.exists() or anno_direct.exists():
        mtime = max(
            gt_direct.stat().st_mtime if gt_direct.exists() else 0.0,
            anno_direct.stat().st_mtime if anno_direct.exists() else 0.0,
        )
        candidates.append((mtime, str(gt_direct) if gt_direct.exists() else "", str(anno_direct) if anno_direct.exists() else ""))

    # 2) 기존 parallel 폴더 모드
    parallel_root = base / "annotations_parallel"
    if parallel_root.exists() and parallel_root.is_dir():
        for d in parallel_root.iterdir():
            if not d.is_dir():
                continue
            gt_path = d / "gt_sentences.jsonl"
            anno_path = d / "annotation_sentences.jsonl"
            if not gt_path.exists() and not anno_path.exists():
                continue
            mtime = max(
                gt_path.stat().st_mtime if gt_path.exists() else 0.0,
                anno_path.stat().st_mtime if anno_path.exists() else 0.0,
                d.stat().st_mtime,
            )
            candidates.append((mtime, str(gt_path) if gt_path.exists() else "", str(anno_path) if anno_path.exists() else ""))

    if not candidates:
        return "", ""

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, gt_default, anno_default = candidates[0]
    return gt_default, anno_default


def main():
    parser = argparse.ArgumentParser(description="Batch + Dataset Viewer")
    default_gt_jsonl, default_annotation_jsonl = _latest_annotation_jsonl_defaults()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "dataset"],
        default="dataset",
        help="뷰어 모드 (default: dataset)",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(Path(__file__).parent / "batch" / "batch_input.jsonl"),
        help="batch 모드에서 열람할 JSONL 경로",
    )
    parser.add_argument(
        "--frames_meta",
        type=str,
        default=str(Path(__file__).parent / "frames" / "episodes_meta.json"),
        help="dataset 모드에서 열람할 episodes_meta.json 경로",
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP 포트 (default: 8765)")
    parser.add_argument(
        "--tactile_display_scale",
        type=float,
        default=0.8,
        help="뷰어에서 tactile 이미지 표시 배율 (0.1~1.0, default: 0.8)",
    )
    parser.add_argument("--dry_run", action="store_true", help="레코드 로딩만 확인하고 종료")
    parser.add_argument(
        "--open_browser",
        action="store_true",
        help="브라우저 자동 열기 활성화 (기본: 비활성화)",
    )
    parser.add_argument(
        "--no_open_browser",
        action="store_true",
        help="(호환용) 브라우저 자동 열기 비활성화",
    )
    parser.add_argument(
        "--gt_jsonl",
        type=str,
        default=default_gt_jsonl,
        help="custom_id -> gt_sentence 매핑 JSONL (기본: 최신 annotations_parallel 폴더)",
    )
    parser.add_argument(
        "--annotation_jsonl",
        type=str,
        default=default_annotation_jsonl,
        help="custom_id -> annotation_sentence 매핑 JSONL (기본: 최신 annotations_parallel 폴더)",
    )
    parser.add_argument(
        "--disable_similarity",
        action="store_true",
        help="GT/annotation 임베딩 유사도 계산 비활성화",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace 공개 임베딩 모델 이름",
    )
    args = parser.parse_args()

    global RECORDS, VIEW_MODE
    VIEW_MODE = args.mode

    global GT_TEXT_BY_ID, ANNOTATION_TEXT_BY_ID
    GT_TEXT_BY_ID = load_text_map_jsonl(args.gt_jsonl, "gt_sentence")
    ANNOTATION_TEXT_BY_ID = load_text_map_jsonl(args.annotation_jsonl, "annotation_sentence")

    global TACTILE_DISPLAY_SCALE
    TACTILE_DISPLAY_SCALE = max(0.1, min(1.0, args.tactile_display_scale))

    global ENABLE_SIMILARITY, EMBEDDING_MODEL_NAME
    ENABLE_SIMILARITY = not args.disable_similarity
    EMBEDDING_MODEL_NAME = args.embedding_model

    if args.mode == "batch":
        if not os.path.exists(args.jsonl):
            print(f"[ERROR] JSONL 파일을 찾을 수 없습니다: {args.jsonl}")
            return
        RECORDS = load_jsonl(args.jsonl)
        print(f"Loaded {len(RECORDS)} batch records from {args.jsonl}")
    else:
        if not os.path.exists(args.frames_meta):
            print(f"[ERROR] frames meta 파일을 찾을 수 없습니다: {args.frames_meta}")
            return
        RECORDS = load_dataset_records(args.frames_meta)
        print(f"Loaded {len(RECORDS)} dataset frame records from {args.frames_meta}")

    if not RECORDS:
        print("[WARN] 표시할 레코드가 없습니다.")
        return

    if args.dry_run:
        print("Dry-run 완료.")
        return

    url = f"http://localhost:{args.port}"
    print(f"Open: {url}")
    print("Stop: Ctrl+C")

    if args.open_browser and not args.no_open_browser:
        import threading
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    server = HTTPServer(("localhost", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료.")


if __name__ == "__main__":
    main()

