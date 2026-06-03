"""
app.py - AI 프레젠테이션 코치 Flask 서버
발표 영상을 업로드하면 멀티모달 분석 결과를 JSON으로 반환
"""
import os
import json
import uuid
import torch
import cv2
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tqdm import tqdm

from config import (
    AUDIO_CONFIG, FUSION_CONFIG, INFERENCE_CONFIG,
    GESTURE_MODEL_PATH, AUDIO_MODEL_PATH, FUSION_MODEL_PATH,
    missing_required_models,
)
from utils import load_gesture_model, load_audio_model, create_text_model
from utils.models import GatedFusion

app = Flask(__name__)
CORS(app)

# =============================================================================
# 설정
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# 비디오 전처리 설정
SEGMENT_DURATION = 1.0
OVERLAP = 0.5
RESIZE_HW = (112, 112)
CLIP_LEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Server] Device: {DEVICE}")


def ensure_runtime_dirs():
    """업로드와 결과 저장 폴더를 생성한다."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)


# =============================================================================
# 오디오 전처리
# =============================================================================
class AudioFeatureExtractorInference:
    """학습 코드와 동일한 오디오 전처리"""
    def __init__(self, config=None):
        if config is None:
            config = AUDIO_CONFIG
        self.sr = config.get('sample_rate', 16000)
        self.n_mels = config.get('n_mels', 80)
        self.n_fft = config.get('n_fft', 1024)
        self.hop_length = config.get('hop_length', 512)
        self.window_size = config.get('window_size', 50)

    def process_full_audio(self, y_audio):
        y_audio = librosa.util.normalize(y_audio)

        mel = librosa.feature.melspectrogram(
            y=y_audio, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        features = librosa.power_to_db(mel, ref=np.max).T

        scaler = StandardScaler()
        try:
            features_norm = scaler.fit_transform(features)
        except ValueError:
            return np.zeros((0, self.n_mels), dtype=np.float32)

        return features_norm.astype(np.float32)


# =============================================================================
# 모델 로드
# =============================================================================
def load_all_models():
    """모든 서브 모델과 퓨전 모델 로드"""
    missing = missing_required_models()
    if missing:
        missing_lines = [f"{name}: {path}" for name, path in missing.items()]
        raise FileNotFoundError("필수 모델 파일이 없습니다.\n" + "\n".join(missing_lines))

    print("[Server] 모델 로드 중...")

    g_model = load_gesture_model(GESTURE_MODEL_PATH, DEVICE)
    a_model = load_audio_model(AUDIO_MODEL_PATH, DEVICE)
    t_model = create_text_model().to(DEVICE)

    f_model = GatedFusion(
        gesture_dim=FUSION_CONFIG['gesture_dim'],
        audio_dim=FUSION_CONFIG['audio_dim'],
        text_dim=FUSION_CONFIG['text_dim'],
        hidden_dim=FUSION_CONFIG['hidden_dim'],
        use_text=True
    ).to(DEVICE)

    if os.path.exists(FUSION_MODEL_PATH):
        f_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
        print("[Server] Fusion 모델 가중치 로드 완료")
    else:
        print(f"[Server] Fusion 모델 파일 없음: {FUSION_MODEL_PATH}")

    g_model.eval()
    a_model.eval()
    t_model.eval()
    f_model.eval()

    return g_model, a_model, t_model, f_model


# =============================================================================
# 전처리
# =============================================================================
def preprocess_new_video(video_path, text_tensor_path=None):
    """비디오에 대해 학습 코드와 동일한 전처리 수행"""
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

    for start_sec, end_sec in tqdm(segments, desc="Processing Segments"):
        indices = np.linspace(int(start_sec * fps), int(end_sec * fps) - 1,
                              num=CLIP_LEN, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, RESIZE_HW)
                frames.append(frame)
            else:
                frames.append(np.zeros((RESIZE_HW[0], RESIZE_HW[1], 3), dtype=np.uint8))

        frames = np.stack(frames).astype(np.float32) / 255.0
        frames = np.transpose(frames, (3, 0, 1, 2))
        frames = (frames - 0.5) / 0.5
        video_tensors.append(torch.from_numpy(frames))

    cap.release()

    # 3. 오디오 전처리
    audio_extractor = AudioFeatureExtractorInference(AUDIO_CONFIG)
    audio_tensors = []

    try:
        y_full, sr = librosa.load(video_path, sr=AUDIO_CONFIG['sample_rate'])
        full_feat = audio_extractor.process_full_audio(y_full)
        frames_per_sec = sr / AUDIO_CONFIG['hop_length']

        for start_sec, end_sec in segments:
            start_idx = int(start_sec * frames_per_sec)
            end_idx = start_idx + AUDIO_CONFIG['window_size']

            feat_seg = full_feat[start_idx:end_idx]

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
        text_tensor_path = os.path.join(BASE_DIR, 'total_tensor.pt')

    text_tensors = []

    if os.path.exists(text_tensor_path):
        try:
            loaded_data = torch.load(text_tensor_path, map_location='cpu')
            if isinstance(loaded_data, list):
                loaded_text = torch.stack(loaded_data) if len(loaded_data) > 0 else torch.zeros(1024)
            else:
                loaded_text = loaded_data

            loaded_text = loaded_text.float()
            if loaded_text.dim() > 1:
                loaded_text = loaded_text.mean(dim=0)
            if loaded_text.dim() == 2 and loaded_text.shape[0] == 1:
                loaded_text = loaded_text.squeeze(0)
            if loaded_text.shape[0] != 1024:
                loaded_text = torch.zeros(1024)

            for _ in segments:
                text_tensors.append(loaded_text)

        except Exception as e:
            print(f"[API] 텍스트 텐서 로드 실패: {e}")
            text_tensors = [torch.zeros(1024) for _ in segments]
    else:
        text_tensors = [torch.zeros(1024) for _ in segments]

    return torch.stack(video_tensors), torch.stack(audio_tensors), torch.stack(text_tensors), segments


# =============================================================================
# 추론
# =============================================================================
@torch.no_grad()
def inference_fusion(video_path):
    """멀티모달 퓨전 추론"""
    print(f"[API] 추론 시작: {os.path.basename(video_path)}")

    v_data, a_data, t_data, segments = preprocess_new_video(video_path)

    if v_data is None:
        print("[API] 데이터 처리 실패")
        return None

    g_model, a_model, t_model, f_model = load_all_models()

    batch_size = 32
    num_samples = len(v_data)
    results = []

    print(f"[API] 총 세그먼트 수: {num_samples}")

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

        for j, prob in enumerate(probs):
            idx = i + j
            start, end = segments[idx]
            results.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "score": float(prob)
            })

    # 결과 저장
    ensure_runtime_dirs()
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(RESULT_FOLDER, f"{video_stem}_results.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"[API] 예측 결과 저장: {save_path}")

    threshold = INFERENCE_CONFIG['emphasis_threshold']
    emphasized = [r for r in results if r['score'] >= threshold]
    print(f"[API] 총 {len(emphasized)}/{len(results)} 구간 강조됨.")

    return results, save_path


# =============================================================================
# Flask 라우트
# =============================================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
            <h1>AI Presentation Coach</h1>
            <p>서버가 정상적으로 실행 중입니다.</p>
            <h3>영상 분석 테스트</h3>
            <form action="/analyze_presentation" method="post" enctype="multipart/form-data">
                <input type="file" name="video" accept=".mp4,.avi,.mov,.mkv">
                <br><br>
                <button type="submit">분석 시작</button>
            </form>
        </div>
    </body>
    </html>
    '''


@app.route('/analyze_presentation', methods=['POST'])
def analyze_presentation():
    """영상 업로드 -> 멀티모달 추론 -> JSON 결과 반환"""
    try:
        ensure_runtime_dirs()

        missing = missing_required_models()
        if missing:
            return jsonify({
                "success": False,
                "message": "필수 모델 파일이 없습니다.",
                "missing_models": missing,
            }), 503

        if 'video' not in request.files:
            return jsonify({"success": False, "message": "영상 파일이 없습니다."}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({"success": False, "message": "파일이 선택되지 않았습니다."}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "지원하지 않는 파일 형식입니다."}), 400

        video_id = str(uuid.uuid4())[:8]
        filename = f"{video_id}_{secure_filename(file.filename)}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        inference_result = inference_fusion(video_path)

        if inference_result is None:
            return jsonify({"success": False, "message": "추론 실패"}), 500

        results, result_json_path = inference_result

        threshold = INFERENCE_CONFIG['emphasis_threshold']
        return jsonify({
            "success": True,
            "video_id": video_id,
            "video_path": video_path,
            "result_json_path": result_json_path,
            "total_segments": len(results),
            "emphasized_segments": len([r for r in results if r['score'] >= threshold]),
            "results": results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == '__main__':
    ensure_runtime_dirs()

    print("[Server] 서버 시작 중...")
    print(f"[Server] 업로드 폴더: {UPLOAD_FOLDER}")
    print(f"[Server] 결과 폴더: {RESULT_FOLDER}")

    app.run(debug=True, port=5000, host='0.0.0.0')
