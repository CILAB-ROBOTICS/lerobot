import os
import time
import yaml
import argparse
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any, Dict
from itertools import islice

import runpod
import requests
from dotenv import load_dotenv

# ---
command_folder_path = './model'

class CmdType(Enum):
    HF_LOGIN = "HF_LOGIN"
    WANDB_LOGIN = "WANDB_LOGIN"
    TRAIN = "TRAIN"
    STOP = "STOP"

class ModelType(Enum):
    ACT = "ACT"
    SmolVLA = "SmolVLA"
    XVLA = "XVLA"
    pi0 = "pi0"
    pi05 = "pi05"
    pi0_fast = "pi0_fast"
    GROOT = "GROOT"

def cmd_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> CmdType:
    value = loader.construct_scalar(node)
    return CmdType(value)

def model_type_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> ModelType:
    value = loader.construct_scalar(node)
    return ModelType(value)

yaml.SafeLoader.add_constructor('!CMD', cmd_constructor)
yaml.SafeLoader.add_constructor('!MODEL_TYPE', model_type_constructor)

def parse_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("-c", "--config", type=str, help="YAML config file path", default='config.yaml')
    temp_args, remaining_argv = conf_parser.parse_known_args()

    defaults = {
        'name': f'python-{" ".join(time.ctime().split()[1:-1])}',
        'template_id': 'o6u732ibrq',
        'gpu': 'NVIDIA RTX 2000 Ada Generation',
        'gpu_count': 1,
        'spot': False,
        'timeout': False,
        'time_limit': 60.0 * 60,
        'terminate': True,
        'commands': []
    }

    if temp_args.config and Path(temp_args.config).exists():
        with open(temp_args.config, "r") as f:
            data = yaml.safe_load(f)

            pod_cfg = data.get('pod', {})
            opt_cfg = data.get('options', {})
            
            yaml_updates = {
                'template_id': pod_cfg.get('template_id'),
                'gpu': pod_cfg.get('gpu'),
                'gpu_count': pod_cfg.get('gpu_count'),
                'spot': pod_cfg.get('spot'),
                'timeout': opt_cfg.get('timeout'),
                'time_limit': opt_cfg.get('time_limit'),
                'terminate': opt_cfg.get('terminate')
            }
            
            yaml_updates = {k: v for k, v in yaml_updates.items() if v is not None}
            commands = []
            for entry in data.get('runtime', {}).get('cmds', []):
                cmd_type = next(iter(entry))
                if cmd_type == CmdType.TRAIN:
                    commands.append((cmd_type, dict(islice(entry.items(), 1, None))))
                else:
                    commands.append((cmd_type, None))

            yaml_updates['commands'] = commands
            print(commands)
            defaults.update(yaml_updates)

    parser = argparse.ArgumentParser(parents=[conf_parser])

    parser.add_argument('--name', type=str)

    parser.add_argument('--template-id', type=str)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--gpu-count', type=int)
    parser.add_argument('--spot', action=argparse.BooleanOptionalAction)
    
    parser.add_argument('--timeout', action=argparse.BooleanOptionalAction)
    parser.add_argument('--time-limit', type=float)
    parser.add_argument('--terminate', action=argparse.BooleanOptionalAction)

    parser.set_defaults(**defaults)
    
    return parser.parse_args()

def get_train_command(train_content):
    model_type = train_content['model_type']
    args = train_content['args']

    match model_type:
        case ModelType.ACT:
            path = f'{command_folder_path}/train-act.sh'
        case ModelType.SmolVLA:
            path = f'{command_folder_path}/train-smolvla.sh'
        case ModelType.XVLA:
            path = f'{command_folder_path}/train-xvla.sh'
        case ModelType.pi0:
            path = f'{command_folder_path}/train-pi0.sh'
        case ModelType.pi05:
            path = f'{command_folder_path}/train-pi05.sh'
        case ModelType.pi0_fast:
            path = f'{command_folder_path}/train-pi0-fash.sh'
        case ModelType.GROOT:
            path = f'{command_folder_path}/train-groot.sh'
            
        case _:
            raise ValueError(f'{model_type} is invalid model type.')
        
    with open(path) as f:
        command_baseline = f.read()
    result = Template(command_baseline).substitute(args)
        
    return result

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

        cmd = []
        if args.commands:
            for cmd_typ, content in args.commands:
                match cmd_typ:
                    case CmdType.HF_LOGIN:
                        with open('command_set/huggingface-login.sh') as f:
                            current_cmd = f.read()
                    case CmdType.WANDB_LOGIN:
                        with open('command_set/wandb-login.sh') as f:
                            current_cmd = f.read()
                    case CmdType.TRAIN:
                        current_cmd = get_train_command(content)
                    case CmdType.STOP:
                        with open('command_set/pod-stop.sh') as f:
                            current_cmd = f.read()
                    case _:
                        raise ValueError(f'{cmd_typ} is invalid.')
                cmd.append(current_cmd)
        
        cmd = " && \\\n".join(cmd)
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

            if args.timeout and (time.time()-start_time > args.time_limit):
                print('Time limit exceeded. Try to terminate pod if not stop only.')
                break
            time.sleep(1)

    finally:
        response = runpod.get_pod(pod_id=pod_id)
        if response and args.terminate:
            runpod.terminate_pod(pod_id=pod_id)
            time.sleep(1)
            response = runpod.get_pod(pod_id=pod_id)
            if not response:
                print('Pod is terminated.')
            else:
                print('Pod is not terminated yet. Check dashboard.')

if __name__ == '__main__':
    args = parse_args()
    main(args)

"""
Usage:
python runpod/train_in_runpod.py \
    --config config.yaml
"""