# 🤖 Robot Hand-Contact Annotation Pipeline

Unitree G1 로봇의 pick-and-place 영상을 GPT-4o Vision으로 어노테이션하는 파이프라인입니다.  
각 프레임마다 **로봇 손이 물체에 닿아 있는지(contact)** 를 자동으로 판별합니다.

---

## 📁 디렉토리 구조

```
annotation/
├── get_dataset.py          # Step 1 · HuggingFace에서 프레임 다운로드
├── prepare_batch_jsonl.py  # Step 2 · Batch API 입력 JSONL 생성
├── submit_batch_openai.py  # Step 3 · OpenAI Batch 제출 + 상태 모니터링
├── watch_batch.py          # Step 3-1 · 별도 상태 모니터링
├── process_batch_output.py # Step 4 · 결과 다운로드 & 파싱
├── viewer.py               # 보조 · JSONL 브라우저 뷰어
├── batch/
│   ├── batch_input.jsonl   # Step 2 출력
│   └── batch_id.txt        # Step 3 출력
└── frames/
    ├── episodes_meta.json  # Step 1 출력
    └── episode_000000/
        ├── frame_000000.jpg
        └── ...
```

---

## ⚙️ 설치

```bash
pip install openai pillow tqdm python-dotenv av huggingface_hub
```

`.env` 파일에 OpenAI API 키를 설정합니다.

```bash
# annotation/.env  (또는 프로젝트 루트)
OPENAI_API_KEY=sk-...
```

---

## 🚀 전체 워크플로우

```
Step 1            Step 2                   Step 3              Step 4
get_dataset  →  prepare_batch_jsonl  →  submit_batch  →  process_output
(프레임 저장)    (JSONL 생성)            (Batch 제출)       (결과 파싱)
```

---

## Step 1 · 프레임 다운로드

HuggingFace `eunjuri/pick_and_place` 데이터셋에서 `cam_third` 영상을 받아
일정 간격으로 프레임을 JPEG로 저장합니다.

```bash
python get_dataset.py
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--out_dir` | `frames` | 프레임 저장 디렉토리 |
| `--frame_step` | `10` | N 프레임마다 1장 샘플링 (30fps 기준 10 → 3fps) |
| `--max_episodes` | 전체 | 처리할 최대 에피소드 수 |

```bash
# 예시: 5 에피소드만, 5 프레임마다 샘플링
python get_dataset.py --max_episodes 5 --frame_step 5
```

**출력:**
```
frames/
    episodes_meta.json        ← Step 2에서 사용
    episode_000000/
        frame_000000.jpg
        frame_000010.jpg
        ...
```

---

## Step 2 · Batch JSONL 생성

저장된 프레임들을 N장씩 묶어 가로 스트립 이미지로 합성하고,
OpenAI Batch API 입력 JSONL을 생성합니다.  
어노테이션은 스트립의 **중간 프레임** 기준으로 기록됩니다.

```bash
python prepare_batch_jsonl.py
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--frames_meta` | `frames/episodes_meta.json` | Step 1 출력 메타 파일 경로 |
| `--out_dir` | `batch` | JSONL 출력 디렉토리 |
| `--model` | `gpt-4o` | 사용할 OpenAI 모델 |
| `--temperature` | `0.0` | 샘플링 temperature (0 = 결정론적) |
| `--strip_size` | `3` | 한 요청에 묶을 연속 프레임 수 (홀수 권장) |
| `--scale` | `None` (원본) | 각 프레임 축소 비율 `0.0 < RATIO ≤ 1.0` |

```bash
# 예시: 3프레임 스트립, 50% 축소
python prepare_batch_jsonl.py --strip_size 3 --scale 0.5

# 예시: 5프레임 스트립, 75% 축소, gpt-4o-mini 사용
python prepare_batch_jsonl.py --strip_size 5 --scale 0.75 --model gpt-4o-mini
```

**출력:** `batch/batch_input.jsonl`

> 💡 **`--scale` 가이드**  
> 원본 해상도가 클수록 API 비용이 높아집니다.  
> `--scale 0.5`면 1920×1080 → 960×540 (스트립 3장 = 2880×540)

---

## Step 2.5 · JSONL 뷰어 (선택)

생성된 JSONL을 제출 전에 브라우저에서 확인합니다.

```bash
python viewer.py
# 또는
python viewer.py --jsonl batch/batch_input.jsonl --port 8765
```

브라우저가 자동으로 열립니다. `←` / `→` 키로 레코드 탐색.

---

## Step 3 · Batch 제출 + 모니터링

JSONL을 OpenAI Batch API에 제출하고 완료까지 상태를 폴링합니다.

```bash
python submit_batch_openai.py
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--jsonl_path` | `batch/batch_input.jsonl` | 제출할 JSONL 경로 |
| `--out_dir` | `batch` | `batch_id.txt` 저장 위치 |
| `--sleep` | `60` | 상태 폴링 간격(초) |
| `--no_watch` | `False` | 제출만 하고 모니터링 없이 즉시 종료 |

```bash
# 제출 후 자동으로 30초 간격 모니터링
python submit_batch_openai.py --sleep 30

# 제출만 하고 나중에 확인
python submit_batch_openai.py --no_watch
```

**출력:** `batch/batch_id.txt` (Batch ID 저장)

---

## Step 3-1 · 상태 모니터링만 (별도 실행)

이미 제출된 Batch의 상태를 별도로 확인합니다.

```bash
python watch_batch.py
# 또는 ID 직접 지정
python watch_batch.py --batch_id batch_xxxxxxxxxx
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--batch_id` | `batch/batch_id.txt` 에서 자동 로드 | 확인할 Batch ID |
| `--sleep` | `60` | 폴링 간격(초) |
| `--repeat` | `-1` (무한) | 최대 폴링 횟수 (`-1` = 완료/실패까지) |

**출력 예시:**
```
🧾 Batch ID: batch_xxxxxxxxxx
📦 Status: in_progress
📊 Progress : [████████░░░░░░░░░░░░░░░░░░░░░░] 40/100 (40.0%)  ❌ failed: 0
```

Batch가 `completed / failed / cancelled / expired` 상태가 되면 자동 종료됩니다.

---

## Step 4 · 결과 파싱 & 저장

완료된 Batch 결과를 다운로드해 에피소드/프레임별 JSON 및 CSV로 저장합니다.

```bash
python process_batch_output.py
# 또는 ID 직접 지정
python process_batch_output.py --batch_id batch_xxxxxxxxxx --out_dir annotations
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--out_dir` | `annotations` | 결과 저장 디렉토리 |
| `--batch_id` | `batch/batch_id.txt` 에서 자동 로드 | 결과를 받아올 Batch ID |

**출력:**
```
annotations/
    episodes_contact.json            ← 전체 결과 통합 JSON
    episode_000000_contact.json      ← 에피소드별 JSON
    episode_000001_contact.json
    ...
    summary.csv                      ← 빠른 분석용 CSV
```

**레이블 스키마:**
```json
{
  "episode_index": 0,
  "frame_index": 10,
  "left_hand_contact":  true,
  "right_hand_contact": false,
  "contact_object":     "T-shirt",
  "confidence":         "high",
  "reason":             "Left hand is clearly gripping the T-shirt."
}
```

---

## Step 2-B · Task+RGB 기반 Tactile GT 어노테이션 JSONL

Task 설명 + RGB 연속 프레임 스트립을 보고,
현재 시점의 tactile GT를 **한 문장(`gt_sentence`)** 으로 생성하는 배치 입력을 만듭니다.

```bash
python prepare_batch_jsonl_tactile_gt_from_rgb.py
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--frames_meta` | `frames/episodes_meta.json` | 메타 파일 |
| `--out_dir` | `batch` | 출력 디렉토리 |
| `--output_name` | `batch_input_tactile_gt_from_rgb.jsonl` | 출력 JSONL 파일명 |
| `--strip_size` | `3` | 연속 프레임 수(홀수) |
| `--scale` | `None` | 프레임 축소 비율 |

생성 후 전송:

```bash
python submit_batch_openai.py --jsonl_path batch/batch_input_tactile_gt_from_rgb.jsonl --out_dir batch
```

## Step 2-C · Tactile-only 활성 어노테이션 JSONL

연속 tactile 합성 이미지(strip)만 보고,
중앙 프레임에서 tactile 상태를 **한 문장(`annotation_sentence`)** 으로 생성하는 배치 입력을 만듭니다.

```bash
python prepare_batch_jsonl_tactile_only_active_fingers.py
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--frames_meta` | `frames/episodes_meta.json` | 메타 파일 |
| `--out_dir` | `batch` | 출력 디렉토리 |
| `--output_name` | `batch_input_tactile_only_active_fingers.jsonl` | 출력 JSONL 파일명 |
| `--strip_size` | `3` | 연속 tactile 프레임 수(홀수) |
| `--scale` | `None` | 이미지 축소 비율 |

생성 후 전송:

```bash
python submit_batch_openai.py --jsonl_path batch/batch_input_tactile_only_active_fingers.jsonl --out_dir batch
```

---

## 🔁 한 번에 실행하기

```bash
cd annotation

# 1. 프레임 다운로드
python get_dataset.py --max_episodes 10 --frame_step 10

# 2. JSONL 생성 (3프레임 스트립, 50% 축소)
python prepare_batch_jsonl.py --strip_size 3 --scale 0.5

# 3. 제출 (30초 간격 모니터링)
python submit_batch_openai.py --sleep 30

# 4. 결과 파싱 (Batch 완료 후)
python process_batch_output.py --out_dir annotations
```

---

## 💰 비용 추정

OpenAI Batch API는 일반 API 대비 **50% 할인**이 적용됩니다.

| 변수 | 예시 값 |
|---|---|
| 프레임 수 | 1,000장 |
| 스트립 크기 | 3장 |
| 프레임 해상도 (축소 후) | 960×540 |
| 스트립 해상도 | 2880×540 |
| 요청 수 | 1,000회 |

> `--scale` 값을 낮출수록 이미지 토큰이 줄어 비용이 절감됩니다.  
> OpenAI [이미지 토큰 계산기](https://platform.openai.com/docs/guides/vision) 참고.

---

## Step 2-D · 준비부터 제출+결과처리까지 한 번에 (Bash)

JSONL 생성 + 다중 Batch 제출 + 완료 즉시 `process_batch_output.py` 실행을 한 번에 수행합니다.

```bash
bash annotation/run_tactile_annotation_batches.sh --mode gt
bash annotation/run_tactile_annotation_batches.sh --mode annotation
bash annotation/run_tactile_annotation_batches.sh --mode both --strip_size 5 --scale 0.7
```

| 인자 | 설명 |
|---|---|
| `--mode` | `gt` / `annotation` / `both` |
| `--frames_meta` | 입력 메타 파일 경로 |
| `--out_dir` | JSONL/배치 ID 매핑 저장 경로 |
| `--processed_root` | batch 결과 파싱 출력 루트 디렉토리 |
| `--model` | OpenAI 모델명 |
| `--strip_size` | 연속 프레임 수(홀수) |
| `--scale` | 이미지 축소 비율(선택) |
| `--sleep` | 다중 배치 상태 폴링 간격(초) |
