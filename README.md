# Project_Steveddarajabs
발표 영상 하나만 넣으면, 제스처(시각), 음성(청각), 텍스트(언어) 세 가지 채널을 동시에 분석해  
“어디서 어떻게 강조했는지 / 어떻게 개선할지”를 피드백해주는 멀티모달 발표 코칭 AI 프로젝트입니다.

## 1. 프로젝트 개요

### 목표

- 발표 영상(Video)을 입력받아 다음 세 가지 요소를 자동 분석:
  - Visual: 손동작, 몸짓, 시선 등 **제스처 기반 강조 표현**
  - Audio: 목소리 크기, 속도, 높낮이, 침묵 등 **반언어적(Paralinguistic) 강조 표현**
  - Text: 발화 내용/대본의 **문맥적 중요도와 강조 필요 지점**
- 각 모달리티의 분석 결과를 통합하여:
  - 발표자가 실제로 **강조한 구간**과
  - **강조했어야 하는데 놓친 구간**을 찾아내고
  - LLM을 통해 **행동 지침형 코칭 리포트**를 생성

### 핵심 접근 방식

- 모달리티별 **전문가 모델(Expert Models)**을 각각 학습
- 각 전문가 모델에서 **Feature Vector**를 추출
- 이들을 결합하는 **Feature-level Late Fusion + Gated Fusion** 전략 채택
- 마지막에 **Generative LLM**을 통해 텍스트 기반 피드백 생성


## 2. 전체 아키텍처 개요

1. 입력: 발표 영상 (MP4)
2. 전처리 파이프라인
   - Video → 클립 단위 분할 (3D CNN 입력용)
   - Audio → Mel-Spectrogram 추출 (LSTM 입력용)
   - STT(Whisper) → 문장 단위 텍스트 + 타임스탬프
3. 전문가 모델별 추론
   - Gesture Model (3D CNN)
   - Audio Model (Bi-LSTM)
   - Text Model (Solar-pro2 + klue/roberta-large)
4. 멀티모달 Fusion
   - 각 모달 Feature를 하나의 벡터로 결합
   - Gated Fusion MLP를 통해 최종 강조 점수 산출
5. 후처리 & 리포트 생성
   - Thresholding, Segment Merging, Noise Filtering
   - JSON 결과 정리
   - LLM 기반 자연어 코칭 리포트 및 시각화 영상 생성

---

## 3. 전문가 모델 상세 (Expert Models)

### 3.1 Gesture / Visual Model (제스처 강조 탐지)

- 역할  
  - 손동작, 몸통 움직임, 상반신 포즈 등을 통해 **강조 제스처 여부** 판단
- 모델 아키텍처  
  - 3D CNN (ResNet-3D 계열)
    - 후보 모델 비교:
      - `r3d_18` (ResNet-3D-18, 기본 3D CNN)
      - `mc3_18` (Mixed Convolution: 일부 2D + 3D 구조)
      - `r2plus1d_18` ((2+1)D 분해 컨볼루션)
    - 실험 결과: 성능/안정성/실장 용이성 관점에서 **`r3d_18` 채택**
  - Pretrained Weights: **Kinetics-400** 사전학습 가중치 활용
- 입력 데이터
  - Shape: `(B, 3, 16, 112, 112)`  
    - B: Batch Size  
    - 3: RGB 채널  
    - 16: 연속 프레임 수 (약 0.5~1초)  
    - 112×112: 공간 해상도
- 출력
  - Binary Classification: 제스처 강조 여부 (0 = Non-Emphasis, 1 = Emphasis)
  - 최종 FC 직전 Feature: **512-dim Feature Vector**

---

### 3.2 Audio / Paralinguistic Model (음성 강조 패턴 탐지)

- 역할  
  - 목소리 크기, 높낮이(Tone), 속도, 침묵(Pause) 등을 분석해 **강조 패턴** 분류
- 입력 데이터
  - 오디오 신호 → `librosa` 기반 **Log-Mel Spectrogram**
  - 예시 세팅:
    - `n_mels = 80`
    - 윈도우 단위: `window_size = 50` (프레임)
    - StandardScaler로 정규화
- 모델 아키텍처
  - **Bi-LSTM (Bidirectional LSTM)**
    - 시계열 특성을 살리기 위해 전/후 문맥을 모두 반영
- 출력
  - 다중 클래스 분류 (예시 5-class)
    - 0: Normal (비강조)
    - 1: Silence + Emphasis (침묵 후 강조)
    - 2: Pitch Change + Emphasis (높낮이 변화)
    - 3: Loud + Emphasis (발성 강도)
    - 4: Slow + Emphasis (느린 속도)
  - Fusion용 Feature: **128-dim Feature Vector**

---

### 3.3 Text / Linguistic Model (텍스트 중요도 및 문맥 강조)

- 전체 역할  
  - STT로 얻은 대본을 기반으로,  
    - 어떤 문장이 **내용적으로 중요한지**,  
    - 어디를 더 강조해야 하는지 평가

#### (1) Generative LLM: Solar-pro2

- 역할
  - 문장별/구간별 코칭 포인트를 JSON 형태로 생성
  - “어디가 핵심 메시지인지 / 어떤 표현을 바꾸면 좋은지” 분석
- 선정 근거
  - 한국어 중심 환경에서, Qwen/Llama3 대비 한국어 문맥 이해 성능 우수
- 출력
  - 문장별:
    - 중요도 스코어
    - 강조 필요 여부
    - 간단 코멘트
  - 전체:
    - 요약, 개선 방향, 발표 구조 피드백 등 JSON 형태 메타데이터

#### (2) Sentence Encoder: klue/roberta-large

- 역할
  - 각 문장을 고차원 임베딩 벡터로 표현 → Fusion 모델 입력용
- 선정 근거
  - 한국어에 특화된 KLUE 기반
  - Base(768-dim) 대비 Large(1024-dim) 모델의 표현력 강화
- 입력
  - Whisper v3-large로부터 얻은 문장 단위 텍스트 (KSS로 문장 분리)
- 출력
  - 문장 단위 **1024-dim Embedding Vector**

---

### 3.4 Multimodal Fusion Model

- 역할
  - Gesture, Audio, Text 세 가지 전문가 모델의 Feature를 종합해
    - 최종적으로 **“이 구간은 발표자가 강조했는가 / 해야 하는가”** 판단
- 입력 Feature 차원
  - Gesture: 512
  - Audio: 128
  - Text: 1024  
  → 총 **1664-dim (512 + 128 + 1024)**

#### Fusion 방식 비교

1. Simple MLP (Baseline)
   - `[g_feat, a_feat, t_feat]`를 단순 Concatenation
   - 2~3층 Fully Connected MLP로 Binary Classification

2. Gated Fusion (채택)
   - 각 모달리티에 대해 Gate(0~1 스칼라/벡터)를 학습
     - 예: `g' = gate_g * g_feat` 등
   - 신뢰도/중요도가 높은 모달에 더 큰 가중치가 실리도록 학습
   - 노이즈가 많은 모달(예: 음질 불량 오디오, 떨리는 카메라 등)을 자동으로 덜 반영

3. Transformer Fusion (실험 대상)
   - 각 모달을 하나의 토큰으로 보고,  
     - `[Gesture Token, Audio Token, Text Token]`  
     - Multi-Head Attention으로 상호작용 모델링
   - 설명력은 좋지만, 복잡도와 데이터 요구량 측면에서 실 서비스 1차 버전에는 미적용

---

### 3.5 Generative Feedback LLM

- 역할
  - 상기 모든 분석 결과(JSON)를 받아
    - “어디가 부족한지”
    - “어떻게 개선하면 좋은지”
  - 실제 발표자가 이해하기 쉬운 자연어 코칭 리포트로 변환
- 입력
  - Gesture / Audio / Text / Fusion JSON 결과
  - 시간 구간별 강조 여부, 문제점, 좋은 사례
- 출력
  - 전체 요약 피드백
  - 구간별 상세 코칭
  - 문장/제스처/발성 관점에서의 Actionable Feedback

---

## 4. 학습 파이프라인 및 최적화

### 4.1 데이터 처리

- Negative Sampling
  - 라벨링된 강조 구간(Positive) 외, 나머지 시간 구간에서 안전하게 Non-Emphasis 샘플 생성
  - 클래스 불균형을 줄이기 위한 규칙 기반 샘플링 전략 적용
- Sliding Window
  - 긴 발표 영상을 일정 길이의 클립 단위로 분할
  - 예: 16프레임 클립, stride 8프레임 등
  - 연속된 클립 단위로 강조 확률 추론 후, 후처리로 구간 병합

### 4.2 학습 환경 및 최적화

- 인프라
  - Linux + NVIDIA Tesla P40 (24GB) × 2대
- 모델 학습
  - `torch.nn.DataParallel`로 멀티 GPU 사용
  - Batch Size: 실험적으로 2 → 64 → 92 → 128 → 190까지 점진 확대
- 데이터 로딩
  - `num_workers=8`로 설정하여 I/O 병목 최소화

### 4.3 추론 후처리 (Post-processing)

- Thresholding
  - 각 모델에서 나온 Probability를 기준으로 강조 여부 결정  
    - 예: 0.5 ~ 0.6 이상 강조로 간주
- Segment Merging
  - 슬라이딩 윈도우 특성상 끊겨 나오는 클립들을
    - `merge_gap` 기준으로 인접 구간 병합
- Noise Filtering
  - `min_duration` (예: 0.1~0.3초) 미만의 짧은 구간은 노이즈로 처리

---

## 5. 결과물 (Outputs)

### 5.1 JSON 기반 정형 데이터

- Gesture 결과: `<video_id>_gesture.json`
  - 클립별 probability, 강조 구간 리스트
- Audio 결과: `<video_id>_audio.json`
  - 클래스별 확률, class_name, 강조 여부
- Text 결과: `<video_id>_text.json`
  - 문장별 중요도/강조 점수
- Fusion 결과: `<video_id>_fusion.json`
  - 최종 강조 구간, 각 모달 스코어, 설정값(FUSION_CONFIG)

이 JSON들은 이후 LLM 프롬프트에 그대로 투입 가능한 형태로 설계.


### 5.2 코칭 리포트 (Text Feedback)

- LLM이 생성한 자연어 리포트
  - **어디서 잘했는지 / 어디가 아쉬웠는지**
  - **문장별·구간별로 어떻게 개선할지**
  - 제스처, 발성, 내용 측면의 행동 지침(Actionable Advice)

---

## 6. 주요 시도 및 실험

1. 멀티모달 분리 설계 및 통합 구조 설계
   - Vision / Audio / Text를 완전히 독립된 파이프라인으로 구축 후 Fusion 단계에서 통합
2. Gesture 모델 비교
   - `r3d_18`, `mc3_18`, `r2plus1d_18` 비교 실험
   - Optimizer: AdamW vs SGD 비교
   - 학습률, 배치 크기, 입력 클립 길이 등 하이퍼파라미터 튜닝
3. 데이터 파이프라인 구축
   - 영상 경로, 세그먼트 정보, 분석 결과를 **CSV + JSON** 조합으로 관리
   - 실제 서비스/연구 환경에서도 재사용 가능한 형태로 설계
4. LLM 기반 피드백 구조 설계
   - 단순 지적이 아니라,  
     - “왜 문제인지” + “어떻게 고치면 되는지”까지 포함하는  
       Actionable Coaching Prompt를 설계

---

## 7. 리소스 및 역할

- 인프라 지원
  - POSTECH: 라벨링용 발표 영상 제작 및 GPU 서버 지원
  - POSCO 인재창조원: 워크스테이션 및 연구 환경 지원
- 팀 구성
  - 프로젝트 팀원 6명 (모달리티별 담당 및 통합 담당 역할 분담)

---

## 8. 프로젝트 성과 및 배운 점

1. **멀티모달 설계 경험**
   - Vision / Audio / Text 각각의 특성을 살린 뒤 Late Fusion으로 통합하는 구조를 end-to-end로 구현
2. **데이터/실험 관리 역량**
   - 대용량 영상·오디오 데이터를 효율적으로 관리하고,  
     다양한 모델/파라미터 실험을 반복할 수 있는 파이프라인 구축
3. **LLM 활용 능력**
   - 단순 분류 결과를 넘어서,  
     LLM을 통해 **실제 발표자에게 도움이 되는 행동 지침**으로 변환하는 전체 흐름 설계
4. **실제 운영 환경 경험**
   - GPU 서버(Tesla P40)에서의 분산 학습, 메모리 이슈, I/O 병목 등  
     실전 트러블슈팅 경험 축적


# 프로젝트 상세 개요

 - 기간 : 2025년 11월 17일 ~ 2025년 12월 17일
 - Languages : Python(ver.3.13.7)
 - Cloud : Google Cloud Platform
 - 주요 라이브러리 버전 : [required.txt](required.txt) 참조