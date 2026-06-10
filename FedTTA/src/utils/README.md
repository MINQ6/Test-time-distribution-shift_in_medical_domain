# `src/utils` Guide

이 디렉토리는 프로젝트의 보조 로직을 모아둔 곳이며, 현재는 크게 네 영역으로 나뉜다.

- `training/`: federated training과 local round 실행
- `analysis/`: final Global LoRA 분석
- `inference/`: 학습된 체크포인트 평가 및 결과 요약
- 루트의 얇은 래퍼 파일들: 예전 import 경로 호환 또는 독립 실행 스크립트

현재 교수님 요청 범위인 "FedDPA-F Stage 1 Global LoRA 학습 후 분석" 기준으로는
`training/`과 `analysis/`가 핵심이고, `inference/`와 local baseline 쪽은 당장 필수는 아니다.

## Current Priority

지금 실험에서 직접 중요한 흐름은 아래와 같다.

1. `src/main.py`
2. `utils.train`
3. `utils.training.federated`
4. `utils.training.local_round`
5. `utils.analysis.global_lora`

즉, 실제 실행 관점에서 핵심은 다음 파일들이다.

- `train.py`
- `training/common.py`
- `training/local_round.py`
- `training/federated.py`
- `training/wandb.py`:
  wandb를 켠 경우에만 사용
- `analysis/global_lora.py`

## File Map

### `train.py`

호환용 facade 파일이다. 실제 구현은 `utils/training/` 아래에 있고,
기존 코드가 `from utils.train import ...` 형태로 import할 수 있게 다시 export만 한다.

- 현재 필요 여부: 필요
- 이유: `src/main.py`가 이 경로를 직접 사용한다

### `run_local_baseline.py`

모든 client에 대해 local-finetuned dual-LoRA baseline 체크포인트를 따로 만드는 독립 실행 스크립트다.

- 현재 필요 여부: 지금 실험에는 불필요
- 이유: Stage 1 global LoRA 분석 파이프라인에는 연결되어 있지 않다
- 언제 쓰는가: "ours" 말고 local-finetuned baseline까지 따로 비교하고 싶을 때

## `training/`

### `training/common.py`

학습 공통 유틸 모음.

- seed 설정
- HF token 로드
- state dict를 CPU로 안전하게 복사
- checkpoint suffix 처리

- 현재 필요 여부: 필요

### `training/local_round.py`

한 client의 한 round 학습을 실제로 수행하는 핵심 파일이다.

- Global adapter 학습
- `stage1_only=True`일 때 Local adapter 학습 생략
- upload payload 생성
- local/global state 반영

- 현재 필요 여부: 매우 중요

### `training/federated.py`

전체 federated training orchestration을 담당한다.

- 초기 dual LoRA state broadcast
- 각 round의 client 학습 실행
- client upload 수집
- personalized global LoRA 생성
- 최종 checkpoint 저장
- final Global LoRA similarity 분석 자동 실행
- wandb summary/artifact 로깅

- 현재 필요 여부: 매우 중요

### `training/wandb.py`

wandb 의존성을 직접 코드 전체에 퍼뜨리지 않기 위한 얇은 logger wrapper다.

- 현재 필요 여부: 선택적
- 이유: wandb를 안 쓰면 없어도 되지만, 현재 실험에서는 쓰는 편이 유용하다

### `training/local_finetuned.py`

federated training이 아니라, 각 client별 local-only baseline을 학습해 저장하는 코드다.

- 현재 필요 여부: 지금 실험에는 불필요
- 이유: Stage 1 Global LoRA 분석에는 쓰이지 않는다
- 언제 쓰는가: local fine-tuning baseline 비교 실험

## `analysis/`

### `analysis/global_lora.py`

현재 교수님 요청사항을 직접 담당하는 핵심 분석 파일이다.

이 파일에서 수행하는 일:

- 각 client의 final Global LoRA 추출
- `Delta W = B @ A` 계산
- SVD 기반 top-1 left singular vector 추출
- QR 기반 top-1 column-space vector 추출
- 8x8 pairwise cosine similarity matrix 계산
- heatmap 저장
- dendrogram 저장
- summary json/csv/pt 저장

- 현재 필요 여부: 매우 중요

### `analysis/__init__.py`

분석 함수 export용 파일이다.

- 현재 필요 여부: 필요

## `inference/`

이 폴더는 학습 완료 후 체크포인트를 사용해 평가할 때 쓰는 코드다.
지금 단계의 "Stage 1 Global LoRA 구조 분석"만 한다면 직접 필요하지 않다.

### `inference/common.py`

추론용 공통 상수와 metric helper가 들어 있다.

- 현재 필요 여부: 지금은 불필요
- 주의: 경로 상수가 아직 `/home/0630rb/research/...`를 가리키는 흔적이 있어,
  실제 추론에 쓰려면 먼저 경로 정리가 필요하다

### `inference/data.py`

평가 샘플을 불러오는 wrapper.

- 현재 필요 여부: 지금은 불필요

### `inference/model_loader.py`

base model / adapter model을 추론용으로 로드한다.

- 현재 필요 여부: 지금은 불필요

### `inference/evaluator.py`

생성 결과를 F1, BLEU, ROUGE, METEOR로 평가한다.

- 현재 필요 여부: 지금은 불필요

### `inference/pipeline.py`

실제 추론 평가 전체 파이프라인.

- 현재 필요 여부: 지금은 불필요

### `inference/summary.py`

여러 client의 inference 결과 JSON을 모아 summary markdown/json으로 합친다.

- 현재 필요 여부: 지금은 불필요

## Keep vs Remove

현재 기준으로 "안 쓰이는 것처럼 보여도 바로 삭제하면 안 되는 파일"이 있다.

- `train.py`:
  중복처럼 보이지만 엔트리포인트 호환용이라 유지하는 게 맞다
- `run_local_baseline.py`, `training/local_finetuned.py`:
  지금은 안 써도 baseline 실험에 다시 쓸 수 있다
- `inference/*`:
  지금 단계에는 안 쓰지만, 이후 성능 비교 평가 단계에서 필요할 수 있다

즉, 지금 상태는 "불필요한 코드가 섞여 있다"기보다,
"현재 실험 범위를 넘어서는 코드도 함께 들어 있다"에 가깝다.

## Recommendation

현재 실험에 집중할 때는 아래 파일들만 우선 보면 된다.

- `utils/training/federated.py`
- `utils/training/local_round.py`
- `utils/training/common.py`
- `utils/analysis/global_lora.py`

추후 정리가 필요하면 다음 순서가 좋다.

1. `inference/common.py` 경로 상수 수정
2. baseline 코드와 main training 코드의 역할 구분을 더 명확히 문서화
3. 정말 사용하지 않는 legacy import가 생기면 그때 제거
