import os
import time
import argparse

import runpod
import requests
from dotenv import load_dotenv

def main(args):
    try:
        load_dotenv()
        API_KEY = os.getenv('RUNPOD_API_KEY')
        base_url = 'https://rest.runpod.io/v1'
        if not API_KEY:
            raise FileNotFoundError('Cannot access to API_KEY file. Please check .env file exists in currnet folder.')
        
        runpod.api_key = API_KEY
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }

        cmd = ''
        if not args.start_command == 'None':
            with open(f'./runpod/model/{args.start_command}') as f:
                cmd = f.read()

            if not cmd:
                raise FileNotFoundError(f'Cannot read {args.start_command}.')
            
        pod_id = ''
        data = {
            'name': args.name,
            'templateId': args.template_id,
            'gpuTypeIds': [args.gpu],
            'gpuCount': args.gpu_count,
        }
        if args.spot:
            data['interruptible'] = True
        if cmd:
            data['dockerStartCmd'] = ['bash', '-lc', cmd]

        response = requests.post(f'{base_url}/pods', headers=headers, json=data)
        if response.status_code == 500:
            print(response.json()['error'])
        response.raise_for_status()

        pod_id = response.json()['id']

        if not pod_id:
            raise ValueError('Cannot get pod_id.')

        start_time = time.time()
        while True:
            response = runpod.get_pod(pod_id=pod_id)
            if not response:
                print(f'Cannot access {pod_id} pod. It may be terminated already.')
                break

            if response['desiredStatus'] == 'EXITED':
                break

            if time.time() - start_time > args.time_limit:
                print('Time limit exceeded. Try to terminate pod.')
                break
            time.sleep(1)
    finally:
        response = runpod.get_pod(pod_id=pod_id)
        if response and not args.not_terminate:
            runpod.terminate_pod(pod_id=pod_id)
            time.sleep(1)
            response = runpod.get_pod(pod_id=pod_id)
            if not response:
                print('Pod is terminated.')
            else:
                print('Pod is not terminated yet. Check dashboard.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=f'python-{" ".join(time.ctime().split()[1:-1])}')
    parser.add_argument('--template-id', type=str, default='o6u732ibrq') # docker
    parser.add_argument('--gpu', type=str, default='NVIDIA RTX 2000 Ada Generation')
    parser.add_argument('--gpu-count', type=int, default=1)
    parser.add_argument('--spot', action='store_true')
    parser.add_argument('--time-limit', type=float, default=60 * 60)
    parser.add_argument('--not-terminate', action='store_true')
    parser.add_argument('--start-command', type=str, default='None')

    args = parser.parse_args()

    main(args)

"""
Usage:
python runpod/train_in_runpod.py \
    --name eunjuri/soft_manip_data \
    --template-id o6u732ibrq \
    --gpu "NVIDIA RTX 2000 Ada Generation" \
    --spot \
    --start-command act.sh
"""