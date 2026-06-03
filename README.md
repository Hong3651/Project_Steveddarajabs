# Project_Steveddarajabs

발표 영상을 입력받아 제스처(영상), 음성, 텍스트 feature를 함께 분석하고, 발표 구간별 강조 점수를 예측하는 멀티모달 발표 분석 프로젝트입니다.

## 1. 프로젝트 개요

### 목표

- 발표 영상을 일정 길이의 세그먼트로 분할
- 각 세그먼트에서 영상, 음성, 텍스트 feature 추출
- 모달리티별 feature를 결합해 구간별 강조 여부와 강조 score 예측
- Flask API를 통해 영상 업로드부터 JSON 결과 반환까지 연결

### 핵심 접근 방식

- 영상, 음성, 텍스트를 각각 독립적인 feature로 처리
- 각 모달리티의 feature를 Feature-level Late Fusion 방식으로 결합
- Gated Fusion 모델을 사용해 최종 강조 score 산출
- 모델 가중치와 원본 영상 데이터는 GitHub에 포함하지 않고 별도 관리

## 2. 전체 아키텍처

1. 입력: 발표 영상 파일 (`mp4`, `avi`, `mov`, `mkv`)
2. 전처리
   - Video: 1초 단위 세그먼트 생성, 16프레임 clip 추출
   - Audio: `librosa` 기반 Log-Mel Spectrogram 추출
   - Text: 사전에 생성된 1024차원 텍스트 feature 사용
3. 모달리티별 모델 추론
   - Gesture Model: R3D-18 기반 3D CNN
   - Audio Model: BiLSTM
   - Text Model: 1024차원 feature classifier
4. Fusion
   - Gesture 512차원 + Audio 128차원 + Text 1024차원 feature 결합
   - Gated Fusion으로 최종 강조 score 예측
5. 출력
   - 세그먼트별 `start`, `end`, `score` JSON 반환
   - 결과 JSON 파일 저장

## 3. 모델 구성

### 3.1 Gesture / Visual Model

- 구현 파일: `Code/utils/models.py`
- 모델: `torchvision.models.video.r3d_18`
- 입력 shape: `(B, 3, 16, 112, 112)`
- 출력:
  - 제스처 분류 logits
  - Fusion 입력용 512차원 feature

### 3.2 Audio Model

- 구현 파일: `Code/utils/models.py`
- 모델: BiLSTM
- 입력:
  - sample rate: 16000
  - n_mels: 80
  - window size: 50
- 출력:
  - 오디오 클래스 logits
  - Fusion 입력용 128차원 feature
- 클래스:
  - `Normal`
  - `Pause_Talk`
  - `High_Tone`
  - `Loud`

### 3.3 Text Feature

- 텍스트 feature 생성 실험: `Code/04_make_text.ipynb`
- 코드 설정상 기본 모델명: `klue/roberta-large`
- Fusion 입력 차원: 1024
- 현재 Flask API에서는 업로드 영상마다 STT를 새로 수행하지 않습니다.
- `Code/total_tensor.pt`가 있으면 해당 tensor를 사용하고, 없으면 zero vector로 대체합니다.

### 3.4 Fusion Model

- 구현 파일: `Code/utils/models.py`
- 사용 모델: `GatedFusion`
- 입력 feature:
  - Gesture: 512
  - Audio: 128
  - Text: 1024
  - Total: 1664
- 출력:
  - 세그먼트별 강조 score

`FusionMLP`, `GatedFusion`, `TransformerFusion` 클래스가 정의되어 있으며, 현재 Flask 추론 서버에서는 `GatedFusion`을 사용합니다.

## 4. 학습 및 실험 파일

```text
Code/
├── 01_gesture_labeling_tool.ipynb          # 제스처 라벨링 도구
├── 02_train_gesture.ipynb                  # Gesture 모델 학습
├── 03_train_audio_real.ipynb               # Audio 모델 학습
├── 04_make_text.ipynb                      # 텍스트 feature 생성 실험
├── 05_train_Gated.ipynb                    # Gated Fusion 학습
├── 06_Main_inference_model_binary.ipynb    # 통합 추론 실험
└── experiments/                            # 모델 비교 및 추가 실험
```

실험 노트북에는 R3D-18, MC3-18, R(2+1)D, MLP Fusion, Gated Fusion, Transformer Fusion 비교 흔적이 포함되어 있습니다.

## 5. Flask API

### 실행

```bash
pip install -r requirements.txt
cd Code
python app.py
```

서버 실행 후 접속:

```text
http://localhost:5000
```

### `GET /`

영상 업로드 테스트 페이지를 반환합니다.

### `POST /analyze_presentation`

발표 영상을 업로드받아 멀티모달 추론을 수행합니다.

응답 예시:

```json
{
  "success": true,
  "video_id": "abcd1234",
  "total_segments": 30,
  "emphasized_segments": 8,
  "results": [
    {
      "start": 0.0,
      "end": 1.0,
      "score": 0.73
    }
  ]
}
```

## 6. 모델 파일 배치

추론에는 학습된 모델 가중치가 필요합니다. 모델 파일은 용량과 관리 문제로 GitHub에 포함하지 않았습니다.

프로젝트 루트에 `모델집합/` 폴더를 만들고 아래 파일을 배치합니다.

```text
모델집합/
├── gesture_model.pt
├── best_bi_lstm.pth
└── fusion_model_gate.pt
```

모델 파일이 없으면 `/analyze_presentation` API는 임의 가중치로 추론하지 않고, 누락된 모델 목록을 포함한 503 응답을 반환합니다.

## 7. 폴더 구조

```text
Project_Steveddarajabs/
├── Code/
│   ├── app.py
│   ├── config.py
│   ├── utils/
│   │   ├── datasets.py
│   │   ├── helpers.py
│   │   └── models.py
│   ├── experiments/
│   └── outputs/
├── stt_results/
├── A3_최종보고서.pdf
├── project_steve_발표자료.pdf
├── PROJECT_SUMMARY_KO.md
├── requirements.txt
└── README.md
```

## 8. 포트폴리오 설명 포인트

- 발표 영상을 영상, 음성, 텍스트 관점에서 나누어 분석하는 멀티모달 구조를 설계했습니다.
- R3D-18 기반 제스처 모델과 BiLSTM 기반 오디오 모델에서 feature를 추출했습니다.
- 텍스트 feature를 포함해 총 1664차원 feature를 구성하고 Gated Fusion으로 최종 강조 score를 예측했습니다.
- 학습/실험 노트북과 Flask 추론 서버를 분리해 실험 코드와 서비스 코드를 함께 관리했습니다.
- 대용량 모델 파일, 원본 영상, 업로드 결과물은 `.gitignore`로 제외했습니다.

## 9. 현재 한계

- GitHub 저장소만으로는 모델 가중치가 없어 실제 추론까지 재현할 수 없습니다.
- 업로드 영상별 STT와 텍스트 feature 생성 과정은 Flask API에 완전히 통합되어 있지 않습니다.
- 현재 API 응답은 세그먼트별 score 중심이며, 자연어 코칭 리포트 생성은 별도 실험 코드 영역입니다.
- 테스트 코드와 샘플 입력 영상은 포함되어 있지 않습니다.

## 10. 참고 자료

- `A3_최종보고서.pdf`: 프로젝트 최종 보고서
- `project_steve_발표자료.pdf`: 발표 자료
- `PROJECT_SUMMARY_KO.md`: 포트폴리오/면접용 요약
