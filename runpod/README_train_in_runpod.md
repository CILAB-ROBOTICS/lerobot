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
실행 인자는 기본적으로 `config.yaml`의 값을 덮어씁니다.
#### 입력값이 있는 인자
- `-c` 또는 `--config`: 설정으로 사용할 YAML 파일의 경로입니다. 기본값: `config.yaml`
- `--name`: 해당 pod의 이름을 지정합니다. 기본값: `'python-{현재 시간}'`
- `--template-id`: pod를 만들 때 사용할 template의 id입니다.
- `--gpu`: pod가 사용할 gpu의 id값 입니다. id에 관한 정보는 하단의 gpu id 목록을 참고 부탁드립니다.
- `--gpu-count`: pod가 사용할 gpu의 수량 입니다.
- `--time-limit`: python 스크립트로 제어할 pod의 수명입니다. 초 단위로 입력하며, 기본 값은 1시간(3600초) 입니다. <br> `--timeout`이 불활성화 되어있으면 사용되지 않습니다.

#### 입력값이 없는 인자 / 플래그 / 토글 인자
같은 종류의 인자를 중복해서 사용할 경우, 가장 마지막에 사용한 인자만 적용됩니다.
- `--spot` 또는 `--no-spot`: 인자에 `--spot`를 포함하면 pod를 spot으로 생성합니다. `--no-spot`를 포함하는 경우, pod를 on demand로 생성합니다. 어떤 인자도 사용하지 않을 경우 기본적으로 on demand pod로 생성합니다.
- `--terminate` 또는 `--no-terminate`: 인자에 `--terminate`를 포함하면 이 스크립트를 통해 만든 pod가 자신을 stop한 경우 자동으로 해당 pod를 terminate 합니다. `--no-terminate`가 포함된 경우, pod가 stop되어도 terminate하지 않습니다. 어떤 인자도 사용하지 않을 경우 자동으로 종료합니다.
- `--timeout` 또는 `--no-timeout`: 인자에 `--timeout`을 포함했으며, `--terminate`가 활성화된 경우 작동합니다. pod를 생성한 후 `--time-limit`를 초과하는 경우 해당 pod의 즉시 terminate를 시도합니다. `--no-timeout`이 포함된 경우, 시도하지 않습니다. 아무런 인자도 사용하지 않을 경우, 자동 terminate를 시도하지 않습니다.

## 실행 인자 및 `config.yaml` 관련 참고사항
0. 실행 인자와 `config.yaml`가 중복되는 경우, 실행 인자가 우선합니다.
1. `config.yaml` 파일 내부에서 pod를 만들 때 생성할 인자를 관리할 수 있습니다.
2. 만약 실행 인자와 config.yaml에서 아무 인자도 지정하지 않은 경우, 아래의 기본값이 사용됩니다.
    ```
    'name': f'python-{" ".join(time.ctime().split()[1:-1])}',
    'template_id': 'o6u732ibrq',
    'gpu': 'NVIDIA RTX 2000 Ada Generation',
    'gpu_count': 1,
    'spot': False,
    'timeout': False,
    'time_limit': 60.0 * 60, # 3600초, 1시간
    'terminate': True,
    'commands': []
    ```
3. start command를 runtime, cmds항목에 작성해서 pod를 실행할 수 있습니다. 이때 사용되는 스크립트 파일은 `train_in_runpod.py` 스크립트의 실행 위치 기준으로 `./model`에 저장된 스크립트를 start command로 사용합니다. 허용되는 명령어 및 설명은 다음과 같습니다.
- `!CMD HF_LOGIN`: `huggingface-login.sh`를 실행합니다. 환경변수로 주입된 Hugging Face의 API를 이용하여 로그인을 시도합니다.
- `!CMD WANDB_LOGIN`: `wandb-login.sh`를 실행합니다. 환경변수로 주입된 wandb의 key를 이용하여 로그인을 시도합니다.
- `!CMD TRAIN`: 이후 주어지는 매개변수들을 기준으로, 각각의 스크립트를 실행합니다. (예시) `!MODEL_TYPE ACT`로 model_type을 입력한 경우, `train-act.sh`를 실행합니다. 주요 항목은 아래 TRAIN 및 스크립트 관련 참고사항을 확인 부탁드립니다.
- `!CMD STOP`: 현재 pod를 stop 상태로 만듭니다. `train_in_runpod.py`로 pod를 terminate를 하기 위한 필수적인 조치입니다. 학습 코드 맨 아래에 포함을 권장합니다. 해당 명령어가 포함되지 않을 경우 **수동으로 pod를 종료**해야 합니다.
4. start commnd를 입력하는 경우 `!CMD`나 `!MODEL_TYPE`를 붙여야 스크립트에서 해석 가능합니다.

작성된 `config.yaml`의 예시는 아래와 같습니다. <br>
```
pod:
  template_id: "o6u732ibrq"
  gpu: "NVIDIA RTX 2000 Ada Generation"
  gpu_count: 1
  spot: true

options:
  timeout: false
  time_limit: 3600
  terminate: true

runtime:
  cmds:
    - !CMD HF_LOGIN: ~
    - !CMD WANDB_LOGIN: ~
    - !CMD TRAIN:
      model_type: !MODEL_TYPE ACT
      args:
        repo_id: "soft_manip_data"
        step: 1000
    - !CMD STOP: ~
```


## TRAIN 및 스크립트 관련 참고사항
- `!CMD TRAIN` 이후, model type과 매개변수를 지정할 수 있습니다.
- 사용할 수 있는 매개변수는 현재 repo_id와 step이 있습니다.
- 현재 작성된 스크립트는 ACT, SmolVLA, XVLA, pi0, pi0.5, pi0_fast, GROOT가 존재하나, ACT의 format에 맞춰 스크립트를 작성했기 때문에 실제 policy.repo_id등과 일치하지 않을 수 있습니다. 사용한 format은 아래와 같습니다.<br> 매개변수로 주어지는 것은 repo_id, step을 지정할 수 있으며, 모든 스크립트 파일에서 동일한 format을 사용했습니다.
    ```
    # train-{model}.sh 파일에서,
    dataset.repo_id = eunjuri/${repo_id}
    output_dir = outputs/train/{model}_${repo_id}
    job_name = {model}_{repo_id}
    policy.repo_id = eunjuri/{model}_policy_{repo_id}
    step = ${step}
    ```
- 정의된 model type은 다음과 같습니다. model_type 항목에 `!MODEL_TYPE`을 포함하여 대소문자를 구분해 입력해야 합니다.
    - ACT
    - SmolVLA
    - XVLA
    - pi0
    - pi05
    - pi0_fast
    - GROOT
- 실행 인자 중 모든 항목을 작성해야 합니다.
- 여러 개의 학습을 하나의 pod에서 다른 조건으로 하기 위해서는 이어서 작성하면 됩니다. 예시는 아래와 같습니다.
    ```
    runtime:
      cmds:
        - !CMD HF_LOGIN: ~
        - !CMD WANDB_LOGIN: ~
        - !CMD TRAIN:
          model_type: !MODEL_TYPE ACT
          args:
            repo_id: "soft_manip_data"
            step: 1000
        - !CMD TRAIN:
          model_type: !MODEL_TYPE ACT
          args:
            repo_id: "soft_manip_data"
            step: 3000
        - !CMD STOP
    ```

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
 {'id': 'NVIDIA GeForce RTX 3080', 'displayName': 'RTX 3080', 'memoryInGb': 10},
 {'id': 'NVIDIA GeForce RTX 3080 Ti', 'displayName': 'RTX 3080 Ti', 'memoryInGb': 12},
 {'id': 'NVIDIA GeForce RTX 3090', 'displayName': 'RTX 3090', 'memoryInGb': 24},
 {'id': 'NVIDIA GeForce RTX 3090 Ti', 'displayName': 'RTX 3090 Ti', 'memoryInGb': 24},
 {'id': 'NVIDIA GeForce RTX 4070 Ti', 'displayName': 'RTX 4070 Ti', 'memoryInGb': 12},
 {'id': 'NVIDIA GeForce RTX 4080', 'displayName': 'RTX 4080', 'memoryInGb': 16},
 {'id': 'NVIDIA GeForce RTX 4080 SUPER', 'displayName': 'RTX 4080 SUPER', 'memoryInGb': 16},
 {'id': 'NVIDIA GeForce RTX 4090', 'displayName': 'RTX 4090', 'memoryInGb': 24},
 {'id': 'NVIDIA GeForce RTX 5080', 'displayName': 'RTX 5080', 'memoryInGb': 16}, 
 {'id': 'NVIDIA GeForce RTX 5090', 'displayName': 'RTX 5090', 'memoryInGb': 32},
 {'id': 'NVIDIA H100 80GB HBM3', 'displayName': 'H100 SXM', 'memoryInGb': 80},
 {'id': 'NVIDIA H100 NVL', 'displayName': 'H100 NVL', 'memoryInGb': 94},
 {'id': 'NVIDIA H100 PCIe', 'displayName': 'H100 PCIe', 'memoryInGb': 80},
 {'id': 'NVIDIA H200', 'displayName': 'H200 SXM', 'memoryInGb': 141},
 {'id': 'NVIDIA H200 NVL', 'displayName': 'NVIDIA H200 NVL', 'memoryInGb': 143},
 {'id': 'NVIDIA L4', 'displayName': 'L4', 'memoryInGb': 24},
 {'id': 'NVIDIA L40', 'displayName': 'L40', 'memoryInGb': 48},
 {'id': 'NVIDIA L40S', 'displayName': 'L40S', 'memoryInGb': 48},
 {'id': 'NVIDIA RTX 2000 Ada Generation', 'displayName': 'RTX 2000 Ada', 'memoryInGb': 16},
 {'id': 'NVIDIA RTX 4000 Ada Generation', 'displayName': 'RTX 4000 Ada', 'memoryInGb': 20},
 {'id': 'NVIDIA RTX 4000 SFF Ada Generation', 'displayName': 'RTX 4000 Ada SFF', 'memoryInGb': 20},
 {'id': 'NVIDIA RTX 5000 Ada Generation', 'displayName': 'RTX 5000 Ada', 'memoryInGb': 32},
 {'id': 'NVIDIA RTX 6000 Ada Generation', 'displayName': 'RTX 6000 Ada', 'memoryInGb': 48},
 {'id': 'NVIDIA RTX A2000', 'displayName': 'RTX A2000', 'memoryInGb': 6},
 {'id': 'NVIDIA RTX A4000', 'displayName': 'RTX A4000', 'memoryInGb': 16},
 {'id': 'NVIDIA RTX A4500', 'displayName': 'RTX A4500', 'memoryInGb': 20},
 {'id': 'NVIDIA RTX A5000', 'displayName': 'RTX A5000', 'memoryInGb': 24},
 {'id': 'NVIDIA RTX A6000', 'displayName': 'RTX A6000', 'memoryInGb': 48},
 {'id': 'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition', 'displayName': 'RTX PRO 6000 MaxQ',  'memoryInGb': 96},
 {'id': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'displayName': 'RTX PRO 6000', 'memoryInGb': 96},
 {'id': 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition', 'displayName': 'RTX PRO 6000 WK', 'memoryInGb': 96},
 {'id': 'Tesla V100-PCIE-16GB', 'displayName': 'Tesla V100', 'memoryInGb': 16},
 {'id': 'Tesla V100-SXM2-16GB', 'displayName': 'V100 SXM2', 'memoryInGb': 16},
 {'id': 'Tesla V100-SXM2-32GB', 'displayName': 'V100 SXM2 32GB', 'memoryInGb': 32},
 {'id': 'unknown', 'displayName': 'unknown', 'memoryInGb': 0}]
```