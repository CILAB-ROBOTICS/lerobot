"""
viewer.py  –  batch_input.jsonl 간단 뷰어
────────────────────────────────────────
표준 라이브러리만 사용 (Pillow 제외, 선택적).
각 요청의 스트립 이미지 + custom_id + task + prompt 를 브라우저에서 확인.

Usage:
    python annotation/viewer.py                          # 기본값 사용
    python annotation/viewer.py --jsonl batch/batch_input.jsonl --port 8765
"""

import argparse
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── 전역 상태 ──────────────────────────────────────────────────────────────────
RECORDS: list = []   # JSONL 전체 파싱 결과


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_info(record: dict) -> dict:
    """record에서 뷰어에 필요한 정보를 추출합니다."""
    custom_id = record.get("custom_id", "")
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
                        # data:{mime};base64,{data}
                        if url.startswith("data:"):
                            header, data = url.split(",", 1)
                            image_mime = header.split(":")[1].split(";")[0]
                            image_b64 = data
                    elif part.get("type") == "text":
                        user_text = part.get("text", "")
            else:
                user_text = str(content)

    return {
        "custom_id": custom_id,
        "model": model,
        "system_text": system_text,
        "user_text": user_text,
        "image_b64": image_b64,
        "image_mime": image_mime,
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
header h1 { font-size: 1.2rem; color: #e94560; flex: 1; }
.nav-btn {
    padding: 6px 18px; border-radius: 6px; border: none;
    background: #0f3460; color: #e0e0e0; cursor: pointer; font-size: 0.9rem;
    transition: background 0.2s;
}
.nav-btn:hover { background: #e94560; }
.nav-btn:disabled { opacity: 0.3; cursor: default; }
#idx-display { font-size: 0.9rem; min-width: 120px; text-align: center; }
#jump-input {
    width: 70px; padding: 5px 8px; border-radius: 6px;
    border: 1px solid #0f3460; background: #16213e; color: #e0e0e0;
    font-size: 0.9rem;
}
.main { max-width: 1400px; margin: 24px auto; padding: 0 24px; }
.card {
    background: #16213e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    border: 1px solid #0f3460;
}
.card h2 { font-size: 0.85rem; color: #e94560; text-transform: uppercase;
           letter-spacing: 1px; margin-bottom: 10px; }
.meta-grid { display: flex; gap: 24px; flex-wrap: wrap; }
.meta-item label { font-size: 0.75rem; color: #888; display: block; }
.meta-item span  { font-size: 0.95rem; color: #fff; font-weight: 600; }
.strip-wrap {
    overflow-x: auto; text-align: center; padding: 8px 0;
}
.strip-wrap img {
    max-height: 400px;
    border-radius: 8px;
    border: 2px solid #0f3460;
    cursor: pointer;
}
.strip-wrap img:hover { border-color: #e94560; }
/* lightbox */
#lb { display:none; position:fixed; inset:0; background:rgba(0,0,0,.85);
      z-index:200; align-items:center; justify-content:center; }
#lb.open { display:flex; }
#lb img { max-width:95vw; max-height:95vh; border-radius:8px; }
#lb-close { position:fixed; top:16px; right:24px; font-size:2rem; cursor:pointer; color:#fff; }
pre {
    white-space: pre-wrap; word-break: break-word;
    background: #0d1b2a; padding: 14px; border-radius: 8px;
    font-size: 0.82rem; line-height: 1.6; color: #b0c4de;
    max-height: 260px; overflow-y: auto;
}
#progress {
    height: 4px; background: #0f3460;
    position: fixed; top: 0; left: 0; z-index: 200;
    transition: width 0.3s;
}
"""

JS = """
let idx = 0;
const total = window.__TOTAL__;

function updateProgress() {
    document.getElementById('progress').style.width = ((idx+1)/total*100)+'%';
}

function load(i) {
    if (i < 0 || i >= total) return;
    idx = i;
    updateProgress();
    fetch('/record?idx=' + i)
        .then(r => r.json())
        .then(data => {
            document.getElementById('idx-display').textContent = (i+1) + ' / ' + total;
            document.getElementById('custom-id').textContent  = data.custom_id;
            document.getElementById('model-id').textContent   = data.model;
            if (data.image_b64) {
                document.getElementById('strip-img').src = 'data:' + data.image_mime + ';base64,' + data.image_b64;
                document.getElementById('strip-section').style.display = '';
            } else {
                document.getElementById('strip-section').style.display = 'none';
            }
            document.getElementById('user-text').textContent   = data.user_text;
            document.getElementById('system-text').textContent = data.system_text;
            document.getElementById('jump-input').value = i + 1;
            document.getElementById('btn-prev').disabled = (i === 0);
            document.getElementById('btn-next').disabled = (i === total - 1);
        });
}

document.addEventListener('DOMContentLoaded', () => {
    load(0);
    document.getElementById('btn-prev').onclick = () => load(idx - 1);
    document.getElementById('btn-next').onclick = () => load(idx + 1);
    document.getElementById('jump-input').addEventListener('change', e => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v)) load(Math.max(0, Math.min(total-1, v-1)));
    });
    document.addEventListener('keydown', e => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') load(idx + 1);
        if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   load(idx - 1);
    });
    // lightbox
    document.getElementById('strip-img').onclick = () => {
        document.getElementById('lb-img').src = document.getElementById('strip-img').src;
        document.getElementById('lb').classList.add('open');
    };
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
<title>Batch JSONL Viewer</title>
<style>{css}</style>
</head>
<body>
<div id="progress"></div>

<!-- Lightbox -->
<div id="lb"><span id="lb-close">✕</span><img id="lb-img" src="" alt="full"></div>

<header>
  <h1>🤖 Batch JSONL Viewer</h1>
  <button class="nav-btn" id="btn-prev">◀ Prev</button>
  <span id="idx-display">— / {total}</span>
  <button class="nav-btn" id="btn-next">Next ▶</button>
  <input id="jump-input" type="number" min="1" max="{total}" title="Jump to index" />
  <span style="font-size:0.8rem;color:#888">총 {total}개</span>
</header>

<div class="main">
  <!-- 메타 -->
  <div class="card">
    <h2>Request Info</h2>
    <div class="meta-grid">
      <div class="meta-item"><label>Custom ID</label><span id="custom-id">-</span></div>
      <div class="meta-item"><label>Model</label><span id="model-id">-</span></div>
    </div>
  </div>

  <!-- 스트립 이미지 -->
  <div class="card" id="strip-section">
    <h2>Strip Image &nbsp;<small style="color:#888;font-size:0.75rem">(클릭하면 확대)</small></h2>
    <div class="strip-wrap">
      <img id="strip-img" src="" alt="strip">
    </div>
  </div>

  <!-- 유저 프롬프트 -->
  <div class="card">
    <h2>User Prompt</h2>
    <pre id="user-text"></pre>
  </div>

  <!-- 시스템 프롬프트 -->
  <div class="card">
    <h2>System Prompt</h2>
    <pre id="system-text"></pre>
  </div>
</div>

<script>window.__TOTAL__ = {total};</script>
<script>{js}</script>
</body>
</html>
"""


def build_html(total: int) -> str:
    return HTML_TEMPLATE.format(css=CSS, js=JS, total=total)


# ── HTTP 핸들러 ────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 로그 억제
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
        path   = parsed.path
        query  = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_html(build_html(len(RECORDS)))

        elif path == "/record":
            idx = int(query.get("idx", ["0"])[0])
            idx = max(0, min(idx, len(RECORDS) - 1))
            info = extract_info(RECORDS[idx])  # type: ignore[arg-type]
            # image_b64 는 크기가 크므로 그대로 전송
            self.send_json(info)

        else:
            self.send_response(404)
            self.end_headers()


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch JSONL Viewer")
    parser.add_argument(
        "--jsonl", type=str,
        default=str(Path(__file__).parent / "batch" / "batch_input.jsonl"),
        help="열람할 JSONL 파일 경로",
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP 포트 (default: 8765)")
    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"[ERROR] JSONL 파일을 찾을 수 없습니다: {args.jsonl}")
        return

    global RECORDS
    RECORDS = load_jsonl(args.jsonl)
    print(f"✅ {len(RECORDS)}개 레코드 로드: {args.jsonl}")

    if not RECORDS:
        print("[WARN] JSONL이 비어 있습니다. 파일을 먼저 생성하세요.")
        print("       python annotation/prepare_batch_jsonl.py")
        return

    url = f"http://localhost:{args.port}"
    print(f"🌐 브라우저에서 열기: {url}")
    print("   종료: Ctrl+C")

    import webbrowser, threading
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    server = HTTPServer(("localhost", args.port), Handler)  # type: ignore[arg-type]
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료.")


if __name__ == "__main__":
    main()

