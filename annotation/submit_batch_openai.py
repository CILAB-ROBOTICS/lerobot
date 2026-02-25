"""
submit_batch_openai.py

prepare_batch_jsonl.py 가 생성한 JSONL 파일을
OpenAI Batch API 에 제출하고 batch_id.txt 에 ID를 저장합니다.

Usage:
    python submit_batch_openai.py
    python submit_batch_openai.py --jsonl_path batch/batch_input.jsonl --out_dir batch
    python submit_batch_openai.py --no_watch        # 제출만 하고 모니터링 안 함
    python submit_batch_openai.py --sleep 30        # 30초 간격으로 폴링
"""

import argparse
from dotenv import load_dotenv
from openai import OpenAI
from os.path import join

from watch_batch import watch_batch


def main(args):
    client = OpenAI()

    print(f"📤 Uploading {args.jsonl_path} ...")
    with open(args.jsonl_path, "rb") as f:
        file = client.files.create(file=f, purpose="batch")
    print(f"✅ Uploaded file ID : {file.id}")

    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"✅ Batch created ID : {batch.id}")
    print(f"🌐 콘솔에서 확인   : https://platform.openai.com/batches/{batch.id}")

    id_path = join(args.out_dir, "batch_id.txt")
    with open(id_path, "w") as fout:
        fout.write(batch.id)
    print(f"💾 Batch ID saved   : {id_path}")

    if args.no_watch:
        print("\n👋 --no_watch 플래그가 설정되어 모니터링을 건너뜁니다.")
        print(f"   나중에 확인하려면: python watch_batch.py --batch_id {batch.id}")
        return

    print(f"\n🔍 상태 모니터링 시작 (폴링 간격: {args.sleep}s, Ctrl+C로 중단)\n")
    watch_batch(batch_id=batch.id, repeat=-1, sleep_time=args.sleep)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Submit contact-annotation JSONL to OpenAI Batch API"
    )
    parser.add_argument(
        "--out_dir", type=str, default="batch",
        help="batch_id.txt 저장 디렉토리 (default: batch)"
    )
    parser.add_argument(
        "--jsonl_path", type=str, default="batch/batch_input.jsonl",
        help="prepare_batch_jsonl.py 가 생성한 JSONL 경로 (default: batch/batch_input.jsonl)"
    )
    parser.add_argument(
        "--no_watch", action="store_true",
        help="제출 후 상태 모니터링 없이 즉시 종료"
    )
    parser.add_argument(
        "--sleep", type=int, default=10,
        metavar="SEC",
        help="상태 폴링 간격(초) (default: 10)"
    )
    args = parser.parse_args()
    main(args)