# app.py - 06_inference_model_binary.ipynb 연동 버전
import os
import sys
import json
import uuid
import torch
import cv2
import librosa
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tqdm import tqdm

# ★ AI 프로젝트 경로 추가
AI_PROJECT_PATH = '/home/stu/ai_project'
sys.path.insert(0, AI_PROJECT_PATH)

# utils.py에서 필요한 함수 import
from utils import load_gesture_model, load_audio_model, create_text_model

app = Flask(__name__)
CORS(app)

# ===== 설정 =====
UPLOAD_FOLDER = os.path.join(AI_PROJECT_PATH, 'new_video')
RESULT_FOLDER = os.path.join(AI_PROJECT_PATH, '결과')
MODEL_FOLDER = os.path.join(AI_PROJECT_PATH, '모델집합')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 제한

# ===== 06번 파일 설정 복사 =====
SEGMENT_DURATION = 1.0
OVERLAP = 0.5
RESIZE_HW = (112, 112)
CLIP_LEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 오디오 설정
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'n_mels': 80,
    'n_fft': 1024,
    'hop_length': 512,
    'window_size': 50
}

# 퓨전 모델 설정 (학습 시 사용한 FUSION_CONFIG)
FUSION_CONFIG = {
    "gesture_dim": 512,
    "audio_dim": 128,
    "text_dim": 1024,
    "hidden_dim": 256,
}

# 경로 설정
GESTURE_MODEL_PATH = os.path.join(MODEL_FOLDER, "gesture_model.pt")
AUDIO_MODEL_PATH = os.path.join(MODEL_FOLDER, "best_bi_lstm.pth")
FUSION_MODEL_PATH = os.path.join(MODEL_FOLDER, "fusion_model_gate.pt")

print(f"[Server] Device: {DEVICE}")

# ===== 06번 파일 클래스 복사 =====
class AudioFeatureExtractorInferenceStyle:
    """학습 코드와 100% 동일한 오디오 전처리 클래스"""
    def __init__(self, config=None):
        if config is None: config = AUDIO_CONFIG
        self.sr = config.get('sample_rate', 16000)
        self.n_mels = config.get('n_mels', 80)
        self.n_fft = config.get('n_fft', 1024)
        self.hop_length = config.get('hop_length', 512)
        self.window_size = config.get('window_size', 50)
    
    def process_full_audio(self, y_audio):
        # 1. Waveform Normalization
        y_audio = librosa.util.normalize(y_audio)
        
        # 2. Mel-Spectrogram
        mel = librosa.feature.melspectrogram(
            y=y_audio, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        features = librosa.power_to_db(mel, ref=np.max).T 
        
        # 3. Global Scaling (파일 전체 기준)
        scaler = StandardScaler()
        try:
            features_norm = scaler.fit_transform(features)
        except ValueError:
            return np.zeros((0, self.n_mels), dtype=np.float32)

        return features_norm.astype(np.float32)


class GatedFusion(nn.Module):
    """학습된 Fusion 모델 구조"""
    def __init__(self, gesture_dim, audio_dim, text_dim, hidden_dim, num_classes=1, use_text=True, dropout=0.5):
        super(GatedFusion, self).__init__()
        self.use_text = use_text
        
        self.g_net = nn.Sequential(
            nn.Linear(gesture_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Dropout(p=dropout)
        )
        self.a_net = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Dropout(p=dropout)
        )
        
        if use_text:
            self.t_net = nn.Sequential(
                nn.Linear(text_dim, hidden_dim), nn.ReLU(),
                nn.BatchNorm1d(hidden_dim), nn.Dropout(p=dropout)
            )
            gate_input_dim = hidden_dim * 3
        else:
            gate_input_dim = hidden_dim * 2

        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 3 if use_text else 2),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, gesture, audio, text=None):
        h_g = self.g_net(gesture)
        h_a = self.a_net(audio)
        features = [h_g, h_a]
        
        if self.use_text and text is not None:
            h_t = self.t_net(text)
            features.append(h_t)
            
        concat_feat = torch.cat(features, dim=1)
        gates = self.gate_net(concat_feat)
        
        h_fused = gates[:, 0:1] * h_g + gates[:, 1:2] * h_a
        if self.use_text and text is not None:
            h_fused += gates[:, 2:3] * h_t
            
        return self.classifier(h_fused)


# ===== 06번 파일 함수 복사 =====
def load_all_models():
    """모든 서브 모델과 퓨전 모델을 로드합니다."""
    print("[Server] 🔧 모델 로드 중...")
    
    # Gesture & Audio Model Load
    g_model = load_gesture_model(GESTURE_MODEL_PATH, DEVICE)
    a_model = load_audio_model(AUDIO_MODEL_PATH, DEVICE)
    t_model = create_text_model().to(DEVICE)  # Text Model (RoBERTa 등)

    # Fusion Model Load
    f_model = GatedFusion(
        gesture_dim=FUSION_CONFIG['gesture_dim'],
        audio_dim=FUSION_CONFIG['audio_dim'],
        text_dim=FUSION_CONFIG['text_dim'],
        hidden_dim=FUSION_CONFIG['hidden_dim'],
        use_text=True
    ).to(DEVICE)
    
    # 가중치 로드
    if os.path.exists(FUSION_MODEL_PATH):
        f_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
        print("[Server] ✅ Fusion 모델 가중치 로드 완료")
    else:
        print(f"[Server] ❌ Fusion 모델 파일 없음: {FUSION_MODEL_PATH}")

    g_model.eval()
    a_model.eval()
    t_model.eval()
    f_model.eval()
    
    return g_model, a_model, t_model, f_model


def preprocess_new_video(video_path, text_tensor_path=None):
    """
    새로운 비디오에 대해 학습 코드와 동일한 전처리를 수행합니다.
    Video: Resize -> Normalize
    Audio: Scaler -> Slice
    Text: (Optional) Placeholder or Embedding extraction
    """
    # 1. 비디오 세그먼트 생성
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    step = SEGMENT_DURATION * (1 - OVERLAP)
    segments = []
    current = 0
    while current + SEGMENT_DURATION <= duration:
        segments.append((current, current + SEGMENT_DURATION))
        current += step

    if not segments:
        return None, None, None, []

    # 2. 비디오 프레임 추출
    video_tensors = []
    
    print("[API] Processing video segments...")
    for start_sec, end_sec in tqdm(segments, desc="Processing Segments"):
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        # 프레임 인덱스 계산 (16개 균등 추출)
        seg_len = end_frame - start_frame
        indices = np.linspace(start_frame, end_frame-1, num=CLIP_LEN, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, RESIZE_HW)
                frames.append(frame)
            else:
                # 실패시 검은 화면
                frames.append(np.zeros((RESIZE_HW[0], RESIZE_HW[1], 3), dtype=np.uint8))
        
        # 정규화 (0~1 -> -1~1) 및 차원 변경 (C, T, H, W)
        frames = np.stack(frames).astype(np.float32) / 255.0
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
        frames = (frames - 0.5) / 0.5
        video_tensors.append(torch.from_numpy(frames))
        
    cap.release()

    # 3. 오디오 전처리 (Global Scaling 적용)
    audio_extractor = AudioFeatureExtractorInferenceStyle(AUDIO_CONFIG)
    audio_tensors = []
    
    try:
        y_full, sr = librosa.load(video_path, sr=AUDIO_CONFIG['sample_rate'])
        full_feat = audio_extractor.process_full_audio(y_full)
        
        frames_per_sec = sr / AUDIO_CONFIG['hop_length']
        
        for start_sec, end_sec in segments:
            start_idx = int(start_sec * frames_per_sec)
            end_idx = start_idx + AUDIO_CONFIG['window_size']
            
            feat_seg = full_feat[start_idx:end_idx]
            
            # 패딩
            if feat_seg.shape[0] < AUDIO_CONFIG['window_size']:
                pad = AUDIO_CONFIG['window_size'] - feat_seg.shape[0]
                feat_seg = np.pad(feat_seg, ((0, pad), (0, 0)), mode='constant')
            
            feat_seg = feat_seg[:AUDIO_CONFIG['window_size']]
            audio_tensors.append(torch.from_numpy(feat_seg).float())
            
    except Exception as e:
        print(f"[API] Audio Error: {e}")
        audio_tensors = [torch.zeros(AUDIO_CONFIG['window_size'], AUDIO_CONFIG['n_mels'])] * len(segments)

    # 4. 텍스트 처리
    if text_tensor_path is None:
        text_tensor_path = os.path.join(AI_PROJECT_PATH, '시연', 'total_tensor.pt')
    
    text_tensors = []
    
    if os.path.exists(text_tensor_path):
        print(f"[API] ℹ️ 텍스트 텐서 로드 중: {text_tensor_path}")
        try:
            loaded_data = torch.load(text_tensor_path, map_location='cpu')
            
            # 리스트인 경우 텐서로 변환 (Stack)
            if isinstance(loaded_data, list):
                if len(loaded_data) > 0:
                    loaded_text = torch.stack(loaded_data)
                else:
                    loaded_text = torch.zeros(1024)
            else:
                loaded_text = loaded_data

            # 차원 축소 (평균)
            loaded_text = loaded_text.float()
            
            if loaded_text.dim() > 1:
                loaded_text = loaded_text.mean(dim=0)
            
            if loaded_text.dim() == 2 and loaded_text.shape[0] == 1:
                 loaded_text = loaded_text.squeeze(0)

            # 최종 형태 확인 및 적용
            if loaded_text.shape[0] != 1024:
                print(f"[API] ⚠️ 텍스트 차원 이상 ({loaded_text.shape}). 0으로 초기화합니다.")
                loaded_text = torch.zeros(1024)

            # 모든 세그먼트에 동일한 텍스트 피처 적용
            for _ in segments:
                text_tensors.append(loaded_text)
                
        except Exception as e:
            print(f"[API] ❌ 텍스트 텐서 로드 실패: {e}")
            print("[API] ⚠️ Zero Tensor로 대체합니다.")
            text_tensors = [torch.zeros(1024) for _ in segments]
            
    else:
        print(f"[API] ⚠️ 텍스트 파일이 없습니다: {text_tensor_path}")
        print("[API] ⚠️ Zero Tensor로 대체합니다.")
        text_tensors = [torch.zeros(1024) for _ in segments]

    return torch.stack(video_tensors), torch.stack(audio_tensors), torch.stack(text_tensors), segments


@torch.no_grad()
def inference_fusion(video_path):
    """
    06번 파일의 메인 추론 함수
    """
    print(f"[API] 🚀 추론 시작: {os.path.basename(video_path)}")
    
    # 1. 데이터 전처리
    v_data, a_data, t_data, segments = preprocess_new_video(video_path)
    
    if v_data is None:
        print("[API] ❌ 데이터 처리 실패")
        return None

    # 2. 모델 로드
    g_model, a_model, t_model, f_model = load_all_models()
    
    # 3. 배치 단위 추론
    batch_size = 32
    num_samples = len(v_data)
    results = []
    
    print(f"[API] 📊 총 세그먼트 수: {num_samples}")
    
    # 배치 처리
    for i in range(0, num_samples, batch_size):
        v_batch = v_data[i:i+batch_size].to(DEVICE)
        a_batch = a_data[i:i+batch_size].to(DEVICE)
        t_batch = t_data[i:i+batch_size].to(DEVICE)
        
        _, g_feats = g_model(v_batch, return_feature=True)
        _, a_feats = a_model(a_batch, return_feature=True)
        
        if hasattr(t_model, 'forward'):
             _, t_feats = t_model(t_batch, return_feature=True)
        else:
             t_feats = t_batch 

        logits = f_model(g_feats, a_feats, t_feats)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        # 결과 리스트에 담기
        for j, prob in enumerate(probs):
            idx = i + j
            start, end = segments[idx]
            
            results.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "score": float(prob)
            })
            
    # 4. 결과 저장 (JSON)
    save_path = video_path.replace(".mp4", "_results.json")
    if save_path == video_path: 
        save_path = video_path + "_results.json"
        
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"[API] 💾 예측 결과(점수)가 저장되었습니다: {save_path}")

    # 5. 강조 구간 필터링
    emphasized_segments = [r for r in results if r['score'] >= 0.5]
    print(f"[API] 총 {len(emphasized_segments)}/{len(results)} 구간 강조됨.")
    
    return results, save_path


# ===== Flask 헬퍼 함수 =====
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ===== Flask 라우트 =====
@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <head>
        <title>AI Presentation Coach</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
            button { background-color: #007bff; color: white; border: none; padding: 10px 20px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 AI 프레젠테이션 코치 서버</h1>
            <p>서버가 정상적으로 실행 중입니다.</p>
            
            <h3>영상 분석 테스트</h3>
            <form action="/analyze_presentation" method="post" enctype="multipart/form-data">
                <input type="file" name="video" accept=".mp4,.avi,.mov,.mkv">
                <br><br>
                <button type="submit">분석 시작 (JSON 결과 반환)</button>
            </form>
        </div>
    </body>
    </html>
    '''


@app.route('/analyze_presentation', methods=['POST'])
def analyze_presentation():
    """
    HTML에서 영상을 업로드 받아 06번 파일의 inference_fusion 함수를 실행하고
    JSON 결과를 반환합니다.
    """
    try:
        # 1. 파일 확인
        if 'video' not in request.files:
            return jsonify({"success": False, "message": "영상 파일이 없습니다."}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"success": False, "message": "파일이 선택되지 않았습니다."}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "지원하지 않는 파일 형식입니다."}), 400
        
        # 2. 파일 저장
        video_id = str(uuid.uuid4())[:8]
        filename = f"{video_id}_{secure_filename(file.filename)}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        print(f"[API] Video saved: {video_path}")
        
        # 3. 06번 파일의 inference_fusion 함수 실행
        results, result_json_path = inference_fusion(video_path)
        
        if results is None:
            return jsonify({
                "success": False, 
                "message": "추론 실패"
            }), 500
        
        # 4. 응답 반환
        return jsonify({
            "success": True,
            "video_id": video_id,
            "video_path": video_path,
            "result_json_path": result_json_path,
            "total_segments": len(results),
            "emphasized_segments": len([r for r in results if r['score'] >= 0.5]),
            "results": results  # 전체 결과 포함
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/videos/<path:filename>')
def serve_video(filename):
    """분석된 영상 파일 제공"""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/results/<path:filename>')
def serve_result(filename):
    """분석 결과 JSON 제공"""
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    
    print("[Server] 서버 시작 중...")
    print(f"[Server] 업로드 폴더: {UPLOAD_FOLDER}")
    print(f"[Server] 결과 폴더: {RESULT_FOLDER}")
    print(f"[Server] 모델 폴더: {MODEL_FOLDER}")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
