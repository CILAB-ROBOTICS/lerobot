"""
process_batch_output.py

OpenAI Batch API 결과를 내려받아 로봇 손-물체 접촉(contact) 어노테이션을
파싱하고 에피소드/프레임 단위로 저장합니다.

저장 구조:
    annotations/
        episodes_contact.json          ← 전체 결과 (episode→frame→label)
        episode_000000_contact.json    ← 에피소드별 파일
        ...
        summary.csv                    ← 빠른 분석용 CSV

Usage:
    python process_batch_output.py --out_dir annotations
    python process_batch_output.py --out_dir annotations --batch_id file-xxxxx
"""

import argparse
import json
import os
import csv
from collections import defaultdict
from os.path import join
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# ─── helpers ──────────────────────────────────────────────────────────────────

def parse_custom_id(custom_id: str) -> tuple[int, int]:
    """
    custom_id 포맷: ep{episode:06d}_fr{frame:06d}
    Returns (episode_index, frame_index)
    """
    parts = custom_id.split("_")
    ep_idx = int(parts[0][2:])   # 'ep000000' → 0
    fr_idx = int(parts[1][2:])   # 'fr000010' → 10
    return ep_idx, fr_idx


def retrieve_batch_results(batch_id: str, client: OpenAI) -> tuple[list[dict], list[dict]]:
    """배치 결과 JSONL 을 다운로드해 파싱된 레코드 리스트로 반환한다."""
    batch = client.batches.retrieve(batch_id)
    print(f"Batch status : {batch.status}")

    if batch.status != "completed":
        raise RuntimeError(
            f"Batch is not completed yet (status={batch.status}). "
            "Run watch_batch.py to wait for completion."
        )

    if batch.output_file_id is None:
        raise RuntimeError("Batch has no output file. It may have failed entirely.")

    print(f"Output file  : {batch.output_file_id}")
    file_content = client.files.content(batch.output_file_id)

    records = []
    for line in file_content.text.splitlines():
        if line.strip():
            records.append(json.loads(line))

    # 오류 파일도 확인
    if batch.error_file_id:
        err_content = client.files.content(batch.error_file_id)
        errors = [json.loads(l) for l in err_content.text.splitlines() if l.strip()]
        print(f"⚠️  {len(errors)} error(s) in batch — check annotations/errors.jsonl")
    else:
        errors = []

    return records, errors  # type: ignore[return-value]


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args):
    load_dotenv()
    client = OpenAI()

    # batch_id 결정
    batch_id = args.batch_id
    if batch_id is None:
        id_file = join("batch", "batch_id.txt")
        if not os.path.exists(id_file):
            raise FileNotFoundError(
                "batch_id.txt not found. Pass --batch_id explicitly "
                "or run submit_batch_openai.py first."
            )
        with open(id_file) as f:
            batch_id = f.read().strip()
    print(f"Batch ID     : {batch_id}")

    # 결과 다운로드
    records, errors = retrieve_batch_results(batch_id, client)
    print(f"Total records: {len(records)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 오류 저장
    if errors:
        with open(join(args.out_dir, "errors.jsonl"), "w") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")

    # episode → { frame_index → label_dict } 구조로 수집
    episode_data: dict[int, dict[int, dict]] = defaultdict(dict)
    failed = 0

    # viewer용 텍스트 매핑 수집
    gt_sentence_rows: list[dict[str, str]] = []
    annotation_sentence_rows: list[dict[str, str]] = []

    for rec in records:
        custom_id = rec.get("custom_id", "")

        # 응답 파싱
        try:
            content = rec["response"]["body"]["choices"][0]["message"]["content"]
            label: dict[str, Any] = json.loads(content)
        except Exception as e:
            print(f"  [WARN] parse error for {custom_id}: {e}")
            label = {
                "left_hand_contact":  None,
                "right_hand_contact": None,
                "contact_object":     None,
                "confidence":         "low",
                "reason":             f"parse error: {e}",
            }
            failed += 1

        # viewer가 바로 읽을 수 있는 sentence 매핑 추출
        gt_sentence = label.get("gt_sentence")
        if custom_id and isinstance(gt_sentence, str) and gt_sentence.strip():
            gt_sentence_rows.append({
                "custom_id": custom_id,
                "gt_sentence": gt_sentence.strip(),
            })

        annotation_sentence = label.get("annotation_sentence")
        if custom_id and isinstance(annotation_sentence, str) and annotation_sentence.strip():
            annotation_sentence_rows.append({
                "custom_id": custom_id,
                "annotation_sentence": annotation_sentence.strip(),
            })

        # 기존 contact 포맷 저장 경로 (custom_id 파싱 가능한 경우에만)
        try:
            ep_idx, fr_idx = parse_custom_id(custom_id)
        except Exception:
            if custom_id:
                print(f"  [WARN] cannot parse custom_id (skip episode/frame aggregation): {custom_id}")
            failed += 1
            continue

        label["frame_index"] = fr_idx
        label["episode_index"] = ep_idx
        episode_data[ep_idx][fr_idx] = label

    print(f"Parsed OK    : {len(records) - failed}")
    print(f"Failed       : {failed}")

    # ── 에피소드별 JSON 저장 ─────────────────────────────────────────────────
    for ep_idx, frames in sorted(episode_data.items()):
        ep_records = [frames[fi] for fi in sorted(frames.keys())]
        ep_path = join(args.out_dir, f"episode_{ep_idx:06d}_contact.json")
        with open(ep_path, "w") as f:
            json.dump({
                "episode_index": ep_idx,
                "frames": ep_records,
            }, f, indent=2)

    # ── 전체 결합 JSON 저장 ──────────────────────────────────────────────────
    all_episodes = []
    for ep_idx, frames in sorted(episode_data.items()):
        all_episodes.append({
            "episode_index": ep_idx,
            "frames": [frames[fi] for fi in sorted(frames.keys())],
        })

    combined_path = join(args.out_dir, "episodes_contact.json")
    with open(combined_path, "w") as f:
        json.dump(all_episodes, f, indent=2)

    # ── summary CSV ─────────────────────────────────────────────────────────
    csv_path = join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "episode_index", "frame_index",
            "left_hand_contact", "right_hand_contact",
            "contact_object", "confidence", "reason",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for ep_idx, frames in sorted(episode_data.items()):
            for fi in sorted(frames.keys()):
                row = frames[fi]
                writer.writerow({k: row.get(k) for k in fieldnames})

    # ── viewer용 sentence 매핑 JSONL 저장 ───────────────────────────────────
    gt_jsonl = join(args.out_dir, "gt_sentences.jsonl")
    with open(gt_jsonl, "w") as f:
        for row in gt_sentence_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    annotation_jsonl = join(args.out_dir, "annotation_sentences.jsonl")
    with open(annotation_jsonl, "w") as f:
        for row in annotation_sentence_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ Done!")
    print(f"   Episodes annotated : {len(episode_data)}")
    print(f"   Combined JSON      : {combined_path}")
    print(f"   Summary CSV        : {csv_path}")
    print(f"   GT JSONL (viewer)  : {gt_jsonl} ({len(gt_sentence_rows)} rows)")
    print(f"   ANNO JSONL(viewer) : {annotation_jsonl} ({len(annotation_sentence_rows)} rows)")
    print(f"   Per-episode JSONs  : {args.out_dir}/episode_XXXXXX_contact.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download & parse OpenAI Batch contact annotation results"
    )
    parser.add_argument(
        "--out_dir", type=str, default="annotations",
        help="어노테이션 저장 디렉토리 (default: annotations)"
    )
    parser.add_argument(
        "--batch_id", type=str, default=None,
        help="OpenAI Batch ID (미지정 시 batch/batch_id.txt 에서 읽음)"
    )
    args = parser.parse_args()
    main(args)
