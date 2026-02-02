# train_in_runpod.py

python을 이용한 runpod의 pod제어를 위한 스크립트 입니다.
특정 template, gpu id와 count를 지정하여 실행 가능합니다.
만약 실행하던 pod가 stop되거나, timeout에 도달하면 해당 pod를 terminate합니다.
<br>
**실행 스크립트와 같은 파일에 `.env`로 정의된 파일과 `RUNPOD_API_KEY`로 정의된 환경 변수가 필요합니다.**


## enviorments
```
pip install -r runpod/requirements-runpod.txt
```
or
```
pip install runpod dotenv
```

## 실행 인자
- --name: 해당 pod의 이름을 지정합니다. 기본값: 'python-{현재 시간}'
- --template-id: pod를 만들 때 사용할 template의 id입니다.
- --gpu: pod가 사용할 gpu의 id값 입니다.
- --gpu-count: pod가 사용할 gpu의 수량 입니다.
- --time-limit: python 스크립트로 제어할 pod의 수명입니다. 초 단위로 입력하며, 기본 값은 1시간(3600초) 입니다.
- --start-command: start command로 사용할 파일의 이름입니다. `train_in_runpod.py`를 기준으로 `./start_commands/` 폴더 내 저장된 파일을 인식합니다. 지정하지 않을 경우 template에 작성된 start command를 사용합니다. 자세한 사항은 **참고사항**을 확인해 주십시오.
- --spot: 해당 인자를 포함할 경우 spot pod로 생성합니다. 만약 포함되지 않을 경우 on demand로 생성합니다.
- --not-terminate: 해당 인자를 포함할 경우, 해당 pod는 terminate되지 않습니다. 수동으로 해당 pod를 terminate해야 합니다.

## start-command 관련 참고사항
1. 실행 스크립트를 기준으로 `./start_commands/` 폴더 내부의 파일을 탐색합니다. <br> 예시: `./start_commands/act-command-v1.sh`에 파일이 있을 경우, 실행 인자로 `--start-command act-command-v1.sh`를 추가.
2. 작성된 항목은 pod의 start command에 `bash -lc {command}`로 입력됩니다. 따라서 command가 한 줄로 실행되므로, `&&` 등을 이용하여 작성해야 합니다. <br> ```huggingface-cli login --token "$HUGGINGFACE_TOKEN" && \ wandb login "$WANDB_API_KEY"```
3. train_in_runpod.py는 pod를 생성하고, 만들어진 pod가 stop되면 terminate합니다. 따라서 실행 스크립트 마지막 command로 스스로를 종료해야 합니다. 예시 명령어는 다음과 같습니다. <br> `runpodctl stop pod "$RUNPOD_POD_ID"` <br> 이때, `$RUNPOD_POD_ID`는 기본적으로 설정된 환경 변수로, 추가적인 필요가 없는 이상 추가적으로 설정하지 않아도 괜찮습니다.


## GPU ID 목록
### 주의! 아래 항목은 2026-01-29 기준 `runpod.get_gpus()`의 결과 입니다. 아래 결과에 gpu가 있더라도 해당 gpu를 이용해 pod를 만들 수 없을 가능성도 있습니다. 
```
[{'id': 'AMD Instinct MI300X OAM', 'displayName': 'MI300X', 'memoryInGb': 192},
 {'id': 'NVIDIA A100 80GB PCIe', 'displayName': 'A100 PCIe', 'memoryInGb': 80},
 {'id': 'NVIDIA A100-SXM4-80GB', 'displayName': 'A100 SXM', 'memoryInGb': 80},
 {'id': 'NVIDIA A30', 'displayName': 'A30', 'memoryInGb': 24},
 {'id': 'NVIDIA A40', 'displayName': 'A40', 'memoryInGb': 48},
 {'id': 'NVIDIA B200', 'displayName': 'B200', 'memoryInGb': 180},
 {'id': 'NVIDIA B300 SXM6 AC', 'displayName': 'B300', 'memoryInGb': 288},
 {'id': 'NVIDIA GeForce RTX 3070', 'displayName': 'RTX 3070', 'memoryInGb': 8},
 {'id': 'NVIDIA GeForce RTX 3080',
  'displayName': 'RTX 3080',
  'memoryInGb': 10},
 {'id': 'NVIDIA GeForce RTX 3080 Ti',
  'displayName': 'RTX 3080 Ti',
  'memoryInGb': 12},
 {'id': 'NVIDIA GeForce RTX 3090',
  'displayName': 'RTX 3090',
  'memoryInGb': 24},
 {'id': 'NVIDIA GeForce RTX 3090 Ti',
  'displayName': 'RTX 3090 Ti',
  'memoryInGb': 24},
 {'id': 'NVIDIA GeForce RTX 4070 Ti',
  'displayName': 'RTX 4070 Ti',
  'memoryInGb': 12},
 {'id': 'NVIDIA GeForce RTX 4080',
  'displayName': 'RTX 4080',
  'memoryInGb': 16},
 {'id': 'NVIDIA GeForce RTX 4080 SUPER',
  'displayName': 'RTX 4080 SUPER',
  'memoryInGb': 16},
 {'id': 'NVIDIA GeForce RTX 4090',
  'displayName': 'RTX 4090',
  'memoryInGb': 24},
 {'id': 'NVIDIA GeForce RTX 5080',
  'displayName': 'RTX 5080',
  'memoryInGb': 16},
 {'id': 'NVIDIA GeForce RTX 5090',
  'displayName': 'RTX 5090',
  'memoryInGb': 32},
 {'id': 'NVIDIA H100 80GB HBM3', 'displayName': 'H100 SXM', 'memoryInGb': 80},
 {'id': 'NVIDIA H100 NVL', 'displayName': 'H100 NVL', 'memoryInGb': 94},
 {'id': 'NVIDIA H100 PCIe', 'displayName': 'H100 PCIe', 'memoryInGb': 80},
 {'id': 'NVIDIA H200', 'displayName': 'H200 SXM', 'memoryInGb': 141},
 {'id': 'NVIDIA H200 NVL',
  'displayName': 'NVIDIA H200 NVL',
  'memoryInGb': 143},
 {'id': 'NVIDIA L4', 'displayName': 'L4', 'memoryInGb': 24},
 {'id': 'NVIDIA L40', 'displayName': 'L40', 'memoryInGb': 48},
 {'id': 'NVIDIA L40S', 'displayName': 'L40S', 'memoryInGb': 48},
 {'id': 'NVIDIA RTX 2000 Ada Generation',
  'displayName': 'RTX 2000 Ada',
  'memoryInGb': 16},
 {'id': 'NVIDIA RTX 4000 Ada Generation',
  'displayName': 'RTX 4000 Ada',
  'memoryInGb': 20},
 {'id': 'NVIDIA RTX 4000 SFF Ada Generation',
  'displayName': 'RTX 4000 Ada SFF',
  'memoryInGb': 20},
 {'id': 'NVIDIA RTX 5000 Ada Generation',
  'displayName': 'RTX 5000 Ada',
  'memoryInGb': 32},
 {'id': 'NVIDIA RTX 6000 Ada Generation',
  'displayName': 'RTX 6000 Ada',
  'memoryInGb': 48},
 {'id': 'NVIDIA RTX A2000', 'displayName': 'RTX A2000', 'memoryInGb': 6},
 {'id': 'NVIDIA RTX A4000', 'displayName': 'RTX A4000', 'memoryInGb': 16},
 {'id': 'NVIDIA RTX A4500', 'displayName': 'RTX A4500', 'memoryInGb': 20},
 {'id': 'NVIDIA RTX A5000', 'displayName': 'RTX A5000', 'memoryInGb': 24},
 {'id': 'NVIDIA RTX A6000', 'displayName': 'RTX A6000', 'memoryInGb': 48},
 {'id': 'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition',
  'displayName': 'RTX PRO 6000 MaxQ',
  'memoryInGb': 96},
 {'id': 'NVIDIA RTX PRO 6000 Blackwell Server Edition',
  'displayName': 'RTX PRO 6000',
  'memoryInGb': 96},
 {'id': 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition',
  'displayName': 'RTX PRO 6000 WK',
  'memoryInGb': 96},
 {'id': 'Tesla V100-PCIE-16GB', 'displayName': 'Tesla V100', 'memoryInGb': 16},
 {'id': 'Tesla V100-SXM2-16GB', 'displayName': 'V100 SXM2', 'memoryInGb': 16},
 {'id': 'Tesla V100-SXM2-32GB',
  'displayName': 'V100 SXM2 32GB',
  'memoryInGb': 32},
 {'id': 'unknown', 'displayName': 'unknown', 'memoryInGb': 0}]
```