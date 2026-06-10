# Project_Steveddarajabs

## 한 줄 소개

발표 영상을 입력하면 제스처, 음성, 텍스트 정보를 함께 분석해 발표 구간별 강조 점수를 예측하는 멀티모달 발표 코칭 AI 프로젝트입니다.

이 프로젝트에서 가장 강조하고 싶은 부분은 **여러 모델 구조를 비교 실험한 뒤, 최종 추론 파이프라인에 가장 적합한 모델 조합을 선택했다는 점**입니다.

## 프로젝트 자료

- [최종 보고서](./A3_최종보고서.pdf)
- [발표 자료](./project_steve_발표자료.pdf)

## 프로젝트 개요

발표에서 중요한 내용은 말의 내용만으로 전달되지 않습니다. 손동작이나 몸의 움직임, 목소리의 크기와 높낮이, 문장 자체의 중요도가 함께 작용합니다.

이 프로젝트는 발표 영상을 하나의 입력으로 받아 시각, 청각, 언어 정보를 함께 분석하고, 발표자가 어느 구간에서 강조를 했는지 판단할 수 있도록 설계했습니다. 최종 결과는 세그먼트별 강조 점수 JSON으로 반환됩니다.

- 입력: 발표 영상 파일 (`mp4`, `avi`, `mov`, `mkv`)
- 출력: 구간별 강조 score, 강조 구간 수, 세그먼트별 JSON 결과
- 핵심 구조: Gesture Model + Audio Model + Text Feature + Gated Fusion
- 활용 목적: 발표 연습, 발표 피드백, 커뮤니케이션 분석

## 모델 비교 실험

단일 모델을 바로 정해 구현한 것이 아니라, 모달리티별 모델과 Fusion 구조를 나누어 비교했습니다. 실험 노트북은 [`experiments/`](./experiments) 폴더에 정리되어 있습니다.

| 실험 영역 | 비교 대상 | 최종 적용 |
| --- | --- | --- |
| Gesture / Visual | R3D-18, MC3-18, R(2+1)D | R3D-18 |
| Audio | Log-Mel Spectrogram + BiLSTM 기반 음성 패턴 분류 | BiLSTM |
| Text | KLUE/RoBERTa 기반 1024차원 text feature | Text feature 입력 |
| Fusion | MLP Fusion, Gated Fusion, Transformer Fusion | Gated Fusion |

최종 Flask 추론 서버에서는 R3D-18 기반 제스처 feature, BiLSTM 기반 오디오 feature, 1024차원 텍스트 feature를 결합하고, Gated Fusion 모델로 구간별 강조 score를 예측합니다.

## 기술 스택

- Language: Python
- Backend: Flask, Flask-CORS
- Deep Learning: PyTorch, TorchVision, TorchAudio
- Video Processing: OpenCV
- Audio Processing: Librosa
- Text Feature: KLUE/RoBERTa 기반 feature 실험
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

현재 Flask API에서는 `Code/total_tensor.pt`가 있으면 해당 tensor를 사용합니다. 파일이 없을 경우에는 zero vector로 대체해 추론 흐름이 끊기지 않도록 처리했습니다.

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
│   └── outputs
├── experiments
│   ├── model_comparison
│   └── training_pipeline
├── stt_results
├── A3_최종보고서.pdf
├── project_steve_발표자료.pdf
├── PROJECT_SUMMARY_KO.md
├── README.md
├── LICENSE
└── requirements.txt
```

## 주요 파일 설명

- `Code/app.py`: Flask 기반 영상 업로드 및 멀티모달 추론 API
- `Code/config.py`: 데이터 경로, 모델 경로, 학습/추론 설정값 관리
- `Code/utils/models.py`: Gesture, Audio, Text, Fusion 모델 정의
- `Code/utils/datasets.py`: 영상, 오디오, 텍스트, Fusion Dataset 정의
- `Code/utils/helpers.py`: 데이터 준비, 학습/평가, 추론, 후처리 유틸리티
- `experiments/training_pipeline/`: 라벨링, 학습, feature 생성, 통합 추론 실험
- `experiments/model_comparison/`: Gesture/Fusion 모델 비교 실험

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

제스처 분석에는 여러 3D CNN 계열 모델을 비교한 뒤 R3D-18을 적용했고, 음성 분석에는 Log-Mel Spectrogram 기반 BiLSTM을 사용했습니다. Fusion 단계에서는 MLP, Gated Fusion, Transformer Fusion을 비교하고, 최종 추론 서버에는 Gated Fusion을 적용했습니다.

즉, 단순히 하나의 모델을 구현한 프로젝트가 아니라 **여러 후보 모델을 실험하고 비교하면서 최종 구조를 선택한 프로젝트**입니다.

## 면접 설명 포인트

- 단순 영상 분류가 아니라 시각, 음성, 언어 정보를 결합한 멀티모달 구조를 설계했습니다.
- R3D-18, MC3-18, R(2+1)D 등 제스처 모델 후보를 비교했습니다.
- MLP Fusion, Gated Fusion, Transformer Fusion을 비교하고 최종적으로 Gated Fusion을 선택했습니다.
- 각 모달리티를 독립적으로 처리한 뒤 feature-level late fusion으로 통합했습니다.
- 영상/오디오 세그먼트 분할, feature 추출, 모델 추론, JSON 반환까지 이어지는 추론 파이프라인을 구성했습니다.
- 학습/실험 노트북과 Flask 추론 서버를 분리해 실험 과정과 실행 코드를 함께 관리했습니다.
