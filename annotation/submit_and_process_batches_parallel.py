"""
submit_and_process_batches_parallel.py

여러 JSONL을 OpenAI Batch API에 빠르게 제출하고,
각 배치가 완료되는 즉시 `process_batch_output.py`를 호출해 결과를 저장합니다.

핵심 포인트:
- 제출은 watch 없이 연속 수행 (Batch API 측 병렬 처리 유도)
- 상태는 다중 batch를 한 루프에서 폴링
- completed 상태가 되면 즉시 process 실행

Usage:
    python submit_and_process_batches_parallel.py \
      --jsonl_paths annotation/batch/a.jsonl,annotation/batch/b.jsonl \
      --out_dir annotation/batch \
      --processed_root annotation/annotations_parallel
"""

import argparse
import json
import os
import subprocess
import time
import shutil
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

TERMINAL_STATES = {"completed", "failed", "cancelled", "expired"}


def submit_batch(client: OpenAI, jsonl_path: str) -> tuple[str, str]:
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return file_obj.id, batch.id


def process_single_batch(batch_id: str, process_script: str, out_dir: str) -> int:
    cmd = [
        "python",
        process_script,
        "--batch_id",
        batch_id,
        "--out_dir",
        out_dir,
    ]
    print(f"[PROCESS] {' '.join(cmd)}")
    return subprocess.call(cmd)


def parse_jsonl_paths(value: str) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise ValueError("--jsonl_paths is empty")
    return items


def _load_sentence_map(path: str, key: str) -> dict[str, str]:
    mapped: dict[str, str] = {}
    if not os.path.exists(path):
        return mapped
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            custom_id = obj.get("custom_id", "")
            text = obj.get(key, "")
            if isinstance(custom_id, str) and custom_id and isinstance(text, str) and text.strip():
                mapped[custom_id] = text.strip()
    return mapped


def _write_sentence_map(path: str, key: str, mapped: dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for custom_id in sorted(mapped.keys()):
            row = {"custom_id": custom_id, key: mapped[custom_id]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_processed_outputs(
    entries: list[dict],
    processed_root: str,
    merged_dirname: str,
    flat_output: bool,
) -> str:
    gt_merged: dict[str, str] = {}
    anno_merged: dict[str, str] = {}

    for e in entries:
        out_dir = e.get("processed_out_dir", "")
        if not out_dir:
            stem = Path(e.get("jsonl_path", "")).stem
            batch_id = e.get("batch_id", "")
            if stem and batch_id:
                candidate = os.path.join(processed_root, f"{stem}__{batch_id}")
                if os.path.isdir(candidate):
                    out_dir = candidate
        if not out_dir or not os.path.isdir(out_dir):
            continue
        gt_map = _load_sentence_map(os.path.join(out_dir, "gt_sentences.jsonl"), "gt_sentence")
        anno_map = _load_sentence_map(os.path.join(out_dir, "annotation_sentences.jsonl"), "annotation_sentence")
        gt_merged.update(gt_map)
        anno_merged.update(anno_map)

    merged_dir = processed_root if flat_output else os.path.join(processed_root, merged_dirname)
    os.makedirs(merged_dir, exist_ok=True)

    gt_out = os.path.join(merged_dir, "gt_sentences.jsonl")
    anno_out = os.path.join(merged_dir, "annotation_sentences.jsonl")
    _write_sentence_map(gt_out, "gt_sentence", gt_merged)
    _write_sentence_map(anno_out, "annotation_sentence", anno_merged)

    meta = {
        "gt_count": len(gt_merged),
        "annotation_count": len(anno_merged),
        "source_batches": [
            {
                "batch_id": e.get("batch_id"),
                "jsonl_path": e.get("jsonl_path"),
                "status": e.get("status"),
                "processed_out_dir": e.get("processed_out_dir", ""),
            }
            for e in entries
        ],
    }
    with open(os.path.join(merged_dir, "merged_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[MERGE] gt_sentences: {len(gt_merged)}")
    print(f"[MERGE] annotation_sentences: {len(anno_merged)}")
    print(f"[MERGE] merged directory: {merged_dir}")
    return merged_dir


def _cleanup_processed_dirs(entries: list[dict]) -> None:
    for e in entries:
        d = e.get("processed_out_dir", "")
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def main(args):
    load_dotenv()
    client = OpenAI()

    jsonl_paths = parse_jsonl_paths(args.jsonl_paths)
    for p in jsonl_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"JSONL not found: {p}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.processed_root, exist_ok=True)

    # 1) 다중 배치 제출
    entries: list[dict] = []
    print(f"Submitting {len(jsonl_paths)} batches...")
    for p in jsonl_paths:
        print(f"[SUBMIT] {p}")
        file_id, batch_id = submit_batch(client, p)
        entry = {
            "jsonl_path": p,
            "file_id": file_id,
            "batch_id": batch_id,
            "status": "submitted",
            "processed": False,
            "process_ok": None,
            "processed_out_dir": "",
        }
        entries.append(entry)
        print(f"  -> file_id={file_id}, batch_id={batch_id}")

    ids_path = os.path.join(args.out_dir, args.ids_filename)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Saved batch map: {ids_path}")

    # 2) 상태 폴링 + 완료 즉시 process
    remaining = {e["batch_id"] for e in entries}
    print(f"\nWatching {len(remaining)} batch(es) every {args.sleep}s...")

    while remaining:
        for entry in entries:
            batch_id = entry["batch_id"]
            if batch_id not in remaining:
                continue

            batch = client.batches.retrieve(batch_id)
            status = batch.status
            entry["status"] = status
            print(f"[STATUS] {batch_id}: {status}")

            if status == "completed":
                if not entry["processed"]:
                    stem = Path(entry["jsonl_path"]).stem
                    batch_work_root = args.processed_root
                    if (not args.keep_per_batch_dirs) and (not args.nested_output):
                        batch_work_root = os.path.join(args.processed_root, "_tmp_batches")
                    os.makedirs(batch_work_root, exist_ok=True)
                    out_dir = os.path.join(batch_work_root, f"{stem}__{batch_id}")
                    os.makedirs(out_dir, exist_ok=True)
                    ret = process_single_batch(
                        batch_id=batch_id,
                        process_script=args.process_script,
                        out_dir=out_dir,
                    )
                    entry["processed"] = True
                    entry["process_ok"] = (ret == 0)
                    entry["processed_out_dir"] = out_dir
                remaining.remove(batch_id)
            elif status in TERMINAL_STATES:
                # completed 외 종료 상태
                remaining.remove(batch_id)

        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

        if remaining:
            time.sleep(args.sleep)

    merged_dir = merge_processed_outputs(
        entries,
        args.processed_root,
        args.merged_dirname,
        flat_output=(not args.nested_output),
    )

    if (not args.keep_per_batch_dirs) and (not args.nested_output):
        _cleanup_processed_dirs(entries)
        tmp_root = os.path.join(args.processed_root, "_tmp_batches")
        if os.path.isdir(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)

    print("\nDone. Final statuses:")
    for e in entries:
        print(f"- {e['batch_id']} | {e['status']} | processed={e['processed']} | ok={e['process_ok']}")
    print(f"- merged: {merged_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple batches and process outputs when completed")
    parser.add_argument(
        "--jsonl_paths",
        type=str,
        required=True,
        help="콤마 구분 JSONL 경로 목록",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="batch",
        help="배치 ID 매핑 파일 저장 디렉토리",
    )
    parser.add_argument(
        "--ids_filename",
        type=str,
        default="batch_ids_multi.json",
        help="배치 매핑 JSON 파일명",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        default="annotations_parallel",
        help="process 결과 루트 디렉토리",
    )
    parser.add_argument(
        "--process_script",
        type=str,
        default=str(Path(__file__).parent / "process_batch_output.py"),
        help="단일 배치 결과 파싱 스크립트 경로",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=10,
        help="상태 폴링 간격(초)",
    )
    parser.add_argument(
        "--merged_dirname",
        type=str,
        default="merged_latest",
        help="통합 GT/annotation JSONL 저장 폴더명 (nested_output일 때만 사용)",
    )
    parser.add_argument(
        "--nested_output",
        action="store_true",
        help="통합 결과를 processed_root/merged_dirname 형태로 저장 (기본: processed_root 바로 저장)",
    )
    parser.add_argument(
        "--keep_per_batch_dirs",
        action="store_true",
        help="배치별 process 결과 폴더를 삭제하지 않고 유지",
    )
    main(parser.parse_args())

