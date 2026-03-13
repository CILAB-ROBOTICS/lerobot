#!/usr/bin/env bash
set -euo pipefail

# 준비(JSONL 생성) + 다중 배치 제출 + 완료 즉시 process까지 자동 실행
# mode:
#   gt          -> Task+RGB 기반 gt_sentence
#   annotation  -> Tactile-only annotation_sentence
#   both        -> 위 두 가지 모두 생성 후 병렬 제출/처리

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="both"
FRAMES_META="${SCRIPT_DIR}/frames/episodes_meta.json"
OUT_DIR="${SCRIPT_DIR}/batch"
MODEL="gpt-5.2"
STRIP_SIZE="3"
SCALE=""
SLEEP="10"
PROCESSED_ROOT="${SCRIPT_DIR}/annotations"
CLEAN_OUTPUT="1"

usage() {
  cat <<'EOF'
Usage:
  bash annotation/run_tactile_annotation_batches.sh [options]

Options:
  --mode <gt|annotation|both>      default: both
  --frames_meta <path>             default: annotation/frames/episodes_meta.json
  --out_dir <path>                 default: annotation/batch
  --processed_root <path>          default: annotation/annotations
  --model <name>                   default: gpt-5-mini
  --strip_size <odd_int>           default: 3
  --scale <ratio>                  optional, e.g. 0.7
  --sleep <sec>                    상태 폴링 간격, default: 10
  --no_clean_output                실행 전 processed_root 정리 비활성화
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --frames_meta) FRAMES_META="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --processed_root) PROCESSED_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --strip_size) STRIP_SIZE="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --sleep) SLEEP="$2"; shift 2 ;;
    --no_clean_output) CLEAN_OUTPUT="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR"
if [[ "$CLEAN_OUTPUT" == "1" ]]; then
  rm -rf "$PROCESSED_ROOT"
fi
mkdir -p "$PROCESSED_ROOT"

JSONLS=()

if [[ "$MODE" == "gt" || "$MODE" == "both" ]]; then
  GT_JSONL="$OUT_DIR/batch_input_tactile_gt_from_rgb.jsonl"
  GT_CMD=(
    python "$SCRIPT_DIR/prepare_batch_jsonl_tactile_gt_from_rgb.py"
    --frames_meta "$FRAMES_META"
    --out_dir "$OUT_DIR"
    --output_name "batch_input_tactile_gt_from_rgb.jsonl"
    --model "$MODEL"
    --strip_size "$STRIP_SIZE"
  )
  if [[ -n "$SCALE" ]]; then
    GT_CMD+=(--scale "$SCALE")
  fi
  echo "[PREPARE-GT] ${GT_CMD[*]}"
  "${GT_CMD[@]}"
  JSONLS+=("$GT_JSONL")
fi

if [[ "$MODE" == "annotation" || "$MODE" == "both" ]]; then
  ANNO_JSONL="$OUT_DIR/batch_input_tactile_only_active_fingers.jsonl"
  ANNO_CMD=(
    python "$SCRIPT_DIR/prepare_batch_jsonl_tactile_only_active_fingers.py"
    --frames_meta "$FRAMES_META"
    --out_dir "$OUT_DIR"
    --output_name "batch_input_tactile_only_active_fingers.jsonl"
    --model "$MODEL"
    --strip_size "$STRIP_SIZE"
  )
  if [[ -n "$SCALE" ]]; then
    ANNO_CMD+=(--scale "$SCALE")
  fi
  echo "[PREPARE-ANNOTATION] ${ANNO_CMD[*]}"
  "${ANNO_CMD[@]}"
  JSONLS+=("$ANNO_JSONL")
fi

if [[ ${#JSONLS[@]} -eq 0 ]]; then
  echo "No JSONL prepared. mode=$MODE"
  exit 1
fi

JSONL_CSV="$(IFS=,; echo "${JSONLS[*]}")"
PAR_CMD=(
  python "$SCRIPT_DIR/submit_and_process_batches_parallel.py"
  --jsonl_paths "$JSONL_CSV"
  --out_dir "$OUT_DIR"
  --processed_root "$PROCESSED_ROOT"
  --sleep "$SLEEP"
)

echo "[SUBMIT+PROCESS] ${PAR_CMD[*]}"
"${PAR_CMD[@]}"

echo "Done."
