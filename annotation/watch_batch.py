import argparse
import time
from openai import OpenAI
from dotenv import load_dotenv
from os.path import join

# convert unix timestamp to human-readable format
def format_timestamp(timestamp):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)) if timestamp else "N/A"


def watch_batch(batch_id: str, repeat: int = -1, sleep_time: int = 60):
    client = OpenAI()

    for i in range(repeat if repeat > 0 else int(2147483647)):  # repeat indefinitely if repeat is -1
        batch = client.batches.retrieve(batch_id)

        print(f"\n🧾 Batch ID: {batch.id}")
        print(f"🌐 콘솔 URL: https://platform.openai.com/batches/{batch.id}")
        print(f"📦 Status: {batch.status}")
        print(f"🕒 Created: {format_timestamp(batch.created_at)}")
        print(f"⏳ Expires: {format_timestamp(batch.expires_at)}")
        print(f"🟩 Completed: {format_timestamp(batch.completed_at)}")
        print(f"🟨 In Progress: {format_timestamp(batch.in_progress_at)}")
        print(f"🟥 Failed: {format_timestamp(batch.failed_at)}")

        rc = batch.request_counts
        if rc:
            total = rc.total or 0
            done  = rc.completed or 0
            failed = rc.failed or 0
            pct = done / total * 100 if total else 0
            bar_len = 30
            filled = int(bar_len * done / total) if total else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\n📊 Progress : [{bar}] {done}/{total} ({pct:.1f}%)  ❌ failed: {failed}")

        if batch.output_file_id:
            print(f"\n📁 Output File ID: {batch.output_file_id}")
            print("👉 Download it from: https://platform.openai.com/files/" + batch.output_file_id)
        else:
            print("\n📁 Output File is not available yet.")

        if batch.status in "completed failed cancelled expired".split():
            print(f"\n✅ Batch finished with status: {batch.status}. Exiting watch.")
            break

        print(f"\n⏱  Next check in {sleep_time}s  (Ctrl+C to stop watching)")
        time.sleep(sleep_time)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Watch OpenAI Batch status")
    parser.add_argument(
        "--batch_id", type=str, default=None,
        help="OpenAI Batch ID (미지정 시 batch/batch_id.txt 에서 읽음)"
    )
    parser.add_argument(
        "--sleep", type=int, default=60,
        metavar="SEC",
        help="폴링 간격(초) (default: 60)"
    )
    parser.add_argument(
        "--repeat", type=int, default=-1,
        help="최대 폴링 횟수. -1이면 완료/실패까지 무한 반복 (default: -1)"
    )
    args = parser.parse_args()

    batch_id = args.batch_id
    if batch_id is None:
        try:
            with open(join("batch", "batch_id.txt"), "r") as f:
                batch_id = f.read().strip()
            print(f"📄 Loaded batch ID from batch/batch_id.txt: {batch_id}")
        except FileNotFoundError:
            print("❌ batch_id.txt not found. --batch_id 를 직접 지정하거나 submit_batch_openai.py 를 먼저 실행하세요.")
            exit(1)

    watch_batch(batch_id=batch_id, repeat=args.repeat, sleep_time=args.sleep)

