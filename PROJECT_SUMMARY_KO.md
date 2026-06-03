# Project_Steveddarajabs 프로젝트 정리

## 한 줄 소개

발표 영상을 입력하면 제스처, 음성, 텍스트를 함께 분석해 발표자가 강조한 구간과 강조가 필요한 구간을 찾아주는 멀티모달 발표 코칭 AI 프로젝트입니다.

## 프로젝트 개요

- 목표: 발표 영상의 시각, 청각, 언어 정보를 통합 분석해 발표 개선 피드백 제공
- 입력: 발표 영상 파일(`mp4`, `avi`, `mov`, `mkv`)
- 출력: 구간별 강조 점수 JSON, 강조 구간, 발표 코칭 피드백
- 핵심 구조: 모달리티별 전문가 모델 학습 후 Gated Fusion으로 통합
- 주요 활용: 발표 연습, 발표 코칭, 커뮤니케이션 역량 분석

## 기술 스택

- Language: Python
- Backend: Flask, Flask-CORS
- Deep Learning: PyTorch, TorchVision, TorchAudio
- Video Processing: OpenCV
- Audio Processing: Librosa
- NLP/STT: Whisper, Transformers, KLUE/RoBERTa
- ML Utilities: NumPy, Pandas, Scikit-learn

## 주요 기능

1. 발표 영상 업로드 및 분석
   - Flask 서버에서 영상 파일을 업로드받아 추론 파이프라인 실행

2. 제스처 강조 탐지
   - R3D-18 기반 3D CNN으로 손동작, 몸짓, 상반신 움직임을 분석
   - 16프레임 단위 클립을 입력으로 사용
   - 512차원 제스처 feature를 추출

3. 음성 강조 패턴 탐지
   - Log-Mel Spectrogram을 추출한 뒤 Bi-LSTM으로 음성 패턴 분류
   - 침묵 후 발화, 높은 톤, 큰 발성 등 반언어적 강조 요소를 분석
   - 128차원 오디오 feature를 추출

4. 텍스트 중요도 분석
   - STT 결과와 KLUE/RoBERTa 기반 텍스트 임베딩을 활용
   - 발표 문장별 중요도와 강조 필요성을 반영
   - 1024차원 텍스트 feature를 사용

5. 멀티모달 Fusion
   - 제스처 512차원, 오디오 128차원, 텍스트 1024차원을 결합
   - 총 1664차원 feature를 Gated Fusion 모델에 입력
   - 모달리티별 신뢰도를 학습해 최종 강조 점수 산출

6. 결과 후처리
   - Thresholding으로 강조 여부 판단
   - 인접 구간 병합
   - 너무 짧은 노이즈 구간 제거
   - JSON 형태 결과 저장

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
- `Code/04_make_text.ipynb`: 텍스트 feature 생성
- `Code/05_train_Gated.ipynb`: Gated Fusion 모델 학습
- `Code/06_Main_inference_model_binary.ipynb`: 통합 추론 실험

## 실행 방법

```bash
pip install -r requirements.txt
cd Code
python app.py
```

서버 실행 후 브라우저에서 다음 주소로 접속합니다.

```text
http://localhost:5000
```

영상 파일을 업로드하면 `/analyze_presentation` API가 분석 결과를 JSON으로 반환합니다.

## API 개요

### `GET /`

간단한 업로드 테스트 페이지를 반환합니다.

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

## 포트폴리오용 설명

이 프로젝트는 발표 영상을 단일 입력으로 받아 시각, 음성, 텍스트 정보를 함께 분석하는 멀티모달 AI 발표 코칭 시스템입니다. 제스처 분석에는 R3D-18 기반 3D CNN, 음성 분석에는 Log-Mel Spectrogram 기반 Bi-LSTM, 텍스트 분석에는 KLUE/RoBERTa 임베딩을 활용했습니다. 각 모달리티에서 추출한 feature를 Gated Fusion 구조로 결합해 최종 강조 점수를 예측하고, 결과를 JSON 형태로 정리해 발표자가 어느 구간에서 강조를 잘했는지 또는 놓쳤는지를 확인할 수 있도록 설계했습니다.

## 면접 설명 포인트

- 단순 영상 분류가 아니라 시각, 음성, 언어 정보를 결합한 멀티모달 구조를 설계했습니다.
- 각 모달리티를 독립적인 전문가 모델로 학습한 뒤 feature-level late fusion으로 통합했습니다.
- Gated Fusion을 사용해 노이즈가 있는 모달리티의 영향은 줄이고 신뢰도 높은 모달리티를 더 반영하도록 구성했습니다.
- 영상/오디오 슬라이딩 윈도우, negative sampling, segment merging 등 실제 추론 파이프라인에 필요한 후처리를 구현했습니다.
- Flask API로 모델 추론을 서비스 형태로 연결해 업로드부터 JSON 결과 반환까지 동작하는 구조를 만들었습니다.

## 개선하면 좋은 점

- `README.md`의 실행 환경과 모델 파일 배치 방법을 더 구체화하면 재현성이 좋아집니다.
- 학습된 모델 파일(`.pt`, `.pth`)은 Git에 포함되지 않으므로 별도 다운로드 경로나 배치 가이드가 필요합니다.
- `config.py`가 `os.getcwd()` 기준 경로를 사용하므로 실행 위치에 따라 모델 경로가 달라질 수 있습니다.
- `app.py`에서 업로드 폴더 생성 전에 파일 저장이 발생할 수 있어 서버 시작 시 디렉토리 생성 보장이 필요합니다.
- 텍스트 추론은 현재 `total_tensor.pt` 기반으로 동작하므로 업로드 영상별 STT/텍스트 feature 생성 흐름을 API에 통합하면 완성도가 높아집니다.
- 테스트 코드와 샘플 입력/출력 JSON을 추가하면 프로젝트 검증과 설명이 쉬워집니다.
