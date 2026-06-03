# Project_Steveddarajabs

## 한 줄 소개

발표 영상을 입력하면 제스처, 음성, 텍스트 정보를 함께 분석해 발표 구간별 강조 점수를 예측하는 멀티모달 발표 코칭 AI 프로젝트입니다.

## 프로젝트 개요

발표에서 중요한 내용은 말의 내용만으로 전달되지 않습니다. 손동작이나 몸의 움직임, 목소리의 크기와 높낮이, 문장 자체의 중요도가 함께 작용합니다.

이 프로젝트는 발표 영상을 하나의 입력으로 받아 시각, 청각, 언어 정보를 함께 분석하고, 발표자가 어느 구간에서 강조를 했는지 판단할 수 있도록 설계했습니다. 최종 결과는 세그먼트별 강조 점수 JSON으로 반환됩니다.

- 입력: 발표 영상 파일 (`mp4`, `avi`, `mov`, `mkv`)
- 출력: 구간별 강조 score, 강조 구간 수, 세그먼트별 JSON 결과
- 핵심 구조: 제스처 모델 + 오디오 모델 + 텍스트 feature + Gated Fusion
- 활용 목적: 발표 연습, 발표 피드백, 커뮤니케이션 분석

## 기술 스택

- Language: Python
- Backend: Flask, Flask-CORS
- Deep Learning: PyTorch, TorchVision, TorchAudio
- Video Processing: OpenCV
- Audio Processing: Librosa
- Text Feature: KLUE/RoBERTa 기반 1024차원 feature 실험
- ML Utilities: NumPy, Scikit-learn

## 주요 기능

### 1. 발표 영상 업로드 및 분석

Flask 서버에서 발표 영상을 업로드받고, 영상/음성/텍스트 feature를 이용한 추론 파이프라인을 실행합니다.

### 2. 제스처 강조 탐지

R3D-18 기반 3D CNN을 사용해 발표자의 손동작, 몸짓, 상반신 움직임을 분석합니다.

- 16프레임 단위 clip 사용
- 입력 크기: `(B, 3, 16, 112, 112)`
- Fusion 입력용 512차원 gesture feature 추출

### 3. 음성 강조 패턴 탐지

발표 영상에서 오디오를 추출한 뒤 Log-Mel Spectrogram으로 변환하고, BiLSTM 모델로 음성 패턴을 분석합니다.

- sample rate: 16000
- n_mels: 80
- window size: 50
- Fusion 입력용 128차원 audio feature 추출
- 클래스: `Normal`, `Pause_Talk`, `High_Tone`, `Loud`

### 4. 텍스트 feature 활용

텍스트 파트는 `klue/roberta-large` 기반 1024차원 feature를 Fusion 입력으로 사용하는 구조입니다.

현재 Flask API에서는 업로드 영상마다 STT를 새로 수행하지 않고, `Code/total_tensor.pt`가 있으면 해당 tensor를 사용합니다. 파일이 없을 경우에는 zero vector로 대체해 추론 흐름이 끊기지 않도록 처리했습니다.

### 5. 멀티모달 Fusion

각 모달리티에서 나온 feature를 결합해 최종 강조 score를 예측합니다.

- Gesture feature: 512차원
- Audio feature: 128차원
- Text feature: 1024차원
- Total feature: 1664차원
- Fusion model: Gated Fusion

`Code/utils/models.py`에는 `FusionMLP`, `GatedFusion`, `TransformerFusion`이 정의되어 있고, 현재 Flask 추론 서버에서는 `GatedFusion`을 사용합니다.

## 폴더 구조

```text
Project_Steveddarajabs
├── Code
│   ├── app.py
│   ├── config.py
│   ├── utils
│   │   ├── datasets.py
│   │   ├── helpers.py
│   │   └── models.py
│   ├── outputs
│   └── experiments
├── stt_results
├── README.md
├── PROJECT_SUMMARY_KO.md
├── requirements.txt
├── A3_최종보고서.pdf
└── project_steve_발표자료.pdf
```

## 주요 파일 설명

- `Code/app.py`: Flask 기반 영상 업로드 및 멀티모달 추론 API
- `Code/config.py`: 데이터 경로, 모델 경로, 학습/추론 설정값 관리
- `Code/utils/models.py`: Gesture, Audio, Text, Fusion 모델 정의
- `Code/utils/datasets.py`: 영상, 오디오, 텍스트, Fusion Dataset 정의
- `Code/utils/helpers.py`: 데이터 준비, 학습/평가, 추론, 후처리 유틸리티
- `Code/01_gesture_labeling_tool.ipynb`: 제스처 라벨링 도구
- `Code/02_train_gesture.ipynb`: 제스처 모델 학습
- `Code/03_train_audio_real.ipynb`: 오디오 모델 학습
- `Code/04_make_text.ipynb`: 텍스트 feature 생성 실험
- `Code/05_train_Gated.ipynb`: Gated Fusion 모델 학습
- `Code/06_Main_inference_model_binary.ipynb`: 통합 추론 실험

## 실행 방법

```bash
pip install -r requirements.txt
cd Code
python app.py
```

서버 실행 후 브라우저에서 아래 주소로 접속합니다.

```text
http://localhost:5000
```

## 모델 파일 배치

추론에는 학습된 모델 가중치가 필요합니다. 모델 파일은 용량 문제로 GitHub 저장소에 포함하지 않았습니다.

프로젝트 루트에 `모델집합/` 폴더를 만들고 아래 파일을 배치합니다.

```text
모델집합/
├── gesture_model.pt
├── best_bi_lstm.pth
└── fusion_model_gate.pt
```

모델 파일이 없으면 `/analyze_presentation` API는 임의 가중치로 추론하지 않고, 누락된 모델 목록을 포함한 503 응답을 반환합니다.

## API 개요

### `GET /`

영상 업로드 테스트 페이지를 반환합니다.

### `POST /analyze_presentation`

발표 영상을 업로드받아 멀티모달 분석을 수행합니다.

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

## 포트폴리오 설명

이 프로젝트는 발표 영상을 단일 입력으로 받아 시각, 음성, 텍스트 정보를 함께 분석하는 멀티모달 발표 분석 시스템입니다.

제스처 분석에는 R3D-18 기반 3D CNN을 사용했고, 음성 분석에는 Log-Mel Spectrogram 기반 BiLSTM을 사용했습니다. 텍스트는 1024차원 feature로 변환해 Fusion 입력에 포함했습니다. 세 모달리티에서 추출한 feature를 Gated Fusion 구조로 결합해 최종 강조 점수를 예측하고, 결과를 JSON 형태로 반환하도록 Flask API까지 연결했습니다.

## 면접 설명 포인트

- 단순 영상 분류가 아니라 시각, 음성, 언어 정보를 결합한 멀티모달 구조를 설계했습니다.
- 각 모달리티를 독립적으로 처리한 뒤 feature-level late fusion으로 통합했습니다.
- Gated Fusion을 사용해 모달리티별 feature를 하나의 강조 score로 결합했습니다.
- 영상/오디오 세그먼트 분할, feature 추출, 모델 추론, JSON 반환까지 이어지는 추론 파이프라인을 구성했습니다.
- 학습 노트북과 Flask 추론 서버를 분리해 실험 코드와 실행 코드를 함께 관리했습니다.

## 현재 상태 및 보완 예정

- 학습된 모델 파일(`.pt`, `.pth`)은 GitHub에 포함하지 않았기 때문에 별도로 배치해야 실제 추론이 가능합니다.
- 업로드 영상별 STT와 텍스트 feature 생성 과정은 현재 API에 완전히 통합되어 있지 않습니다.
- 현재 API는 세그먼트별 강조 score 중심으로 반환하며, 자연어 코칭 리포트 생성은 실험 코드 영역에 남아 있습니다.
- 샘플 입력 영상과 테스트 코드를 추가하면 재현성과 검증 편의성이 더 좋아질 수 있습니다.
