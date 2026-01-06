import os

# ===== 기본 경로 =====
BASE_DIR = os.getcwd()

# 데이터 경로
VIDEO_RAW_DIR = os.path.join(BASE_DIR, "영상raw")
CSV_DIR = os.path.join(BASE_DIR, "영상csv")
MODEL_DIR = os.path.join(BASE_DIR, "모델집합")
RESULT_DIR = os.path.join(BASE_DIR, "모델집합")
STT_DIR = os.path.join(BASE_DIR, "stt_results")
Voice_DATA_DIR = os.path.join(BASE_DIR, "voice_data")
TEXT_TENSORS_PATH =  os.path.join(MODEL_DIR, "total_tensor.pt")
TEXT_SCORES_PATH = os.path.join(MODEL_DIR, "total_text.json" )
# 모델 저장 경로
GESTURE_MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pt")
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "best_bi_lstm.pth")
AUDIO_SCALER_PATH = os.path.join(MODEL_DIR, "audio_scaler.pkl")
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "text_model.pt")
FUSION_MODEL_PATH = os.path.join(MODEL_DIR, "fusion_model_gate.pt")

# ===== 제스처 모델 설정 =====
GESTURE_CONFIG = {
    "clip_len": 16,
    "resize_hw": (112, 112),
    "stride": 8,
    "batch_size": 180,
    "num_epochs": 10,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "threshold": 0.6,
    "min_duration": 0.3,
}

# ===== 오디오 모델 설정 =====
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 512,
    "window_size": 50,
    "input_size": 80,
    "hidden_size": 64,
    "batch_size": 256,
    "num_epochs": 50,
    "lr": 1e-3,
    "num_layers":2,
    "num_classes": 4,
    "patience": 10,
    "stride_sec": 0.5,
    "dropout" : 0.2,
    "n_mels" : 80
}



# ===== 텍스트 모델 설정 =====
TEXT_CONFIG = {
    "model_name": "klue/roberta-large",
    "max_length": 128,
    "batch_size": 64,
    "num_epochs": 5,
    "lr": 2e-5,
    "threshold": 0.8,
}

# ===== 퓨전 설정 =====
# FUSION_CONFIG = {
#     "weights": {'gesture': 0.4, 'audio': 0.35, 'text': 0.25},
#     "threshold": 0.6,
#     "min_duration": 0.3,
#     "batch_size": 180,
#     "num_epochs": 300,
#     "lr": 1e-4,
#     "gesture_dim": 512,
#     "audio_dim": 128,
#     "text_dim": 1024,
#     "hidden_dim": 128,
# }

# =============================================================================
# 추론 설정
# =============================================================================
INFERENCE_CONFIG = {
    'emphasis_threshold': 0.5,  # Fusion 출력 임계값
    'gesture_threshold': 0.5,
    'audio_threshold': 0.5,
    'min_segment_duration': 0.3,  # 최소 세그먼트 길이 (초)
}
# =============================================================================
# 클래스 이름
# =============================================================================
GESTURE_CLASS_NAMES = ['No_Gesture', 'Gesture']
AUDIO_CLASS_NAMES = ['Normal', 'Pause_Talk', 'High_Tone', 'Loud']
FUSION_CLASS_NAMES = ['No_Emphasis', 'Emphasis']

# =============================================================================
# 유틸리티 함수
# =============================================================================

# =============================================================================
# 클래스 이름
# =============================================================================


def ensure_dirs():

    dirs = [
        VIDEO_RAW_DIR,
        CSV_DIR,
        MODEL_DIR,
        RESULT_DIR,
        STT_DIR,
        Voice_DATA_DIR,  
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)