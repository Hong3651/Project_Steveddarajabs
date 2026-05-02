"""
helpers.py - 데이터 준비, 학습/평가, 추론 유틸리티
"""
import os
import sys
import subprocess
import csv
import cv2
import json
import random
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import librosa
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from .datasets import AudioFeatureExtractor


# =============================================================================
# MelSpectrogramExtractor (AudioFeatureExtractor 확장 - 구간 추출 지원)
# =============================================================================
class MelSpectrogramExtractor:
    """Mel-Spectrogram 추출기 (구간 추출 + Global Scaler 지원)"""

    def __init__(self, config=None):
        if config is None:
            config = {
                'sample_rate': 16000,
                'n_mels': 80,
                'n_fft': 1024,
                'hop_length': 512,
                'window_size': 50,
            }

        self.sr = config.get('sample_rate', 16000)
        self.n_mels = config.get('n_mels', 80)
        self.n_fft = config.get('n_fft', 1024)
        self.hop_length = config.get('hop_length', 512)
        self.window_size = config.get('window_size', 50)

        self.scaler = None
        self.is_fitted = False

    def extract_features_raw(self, y_audio, start_sec=None, end_sec=None):
        """Mel-Spectrogram 추출 (구간 지정 가능)"""
        if isinstance(y_audio, str):
            y_audio, _ = librosa.load(y_audio, sr=self.sr, offset=start_sec,
                                      duration=(end_sec - start_sec) if end_sec else None)
        elif start_sec is not None and end_sec is not None:
            start_sample = int(start_sec * self.sr)
            end_sample = int(end_sec * self.sr)
            y_audio = y_audio[start_sample:end_sample]

        if len(y_audio) < self.n_fft:
            return None

        y_audio = librosa.util.normalize(y_audio)
        mel = librosa.feature.melspectrogram(
            y=y_audio, sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        feat = librosa.power_to_db(mel, ref=np.max).T

        return feat

    def pad_or_truncate(self, features, target_len=None):
        """윈도우 크기에 맞게 패딩/자르기"""
        if target_len is None:
            target_len = self.window_size

        if features is None:
            return np.zeros((target_len, self.n_mels), dtype=np.float32)

        if len(features) < target_len:
            pad_len = target_len - len(features)
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
        else:
            features = features[:target_len]

        return features.astype(np.float32)

    def fit_scaler(self, all_features):
        """전체 데이터로 Scaler fit"""
        if isinstance(all_features, list):
            stacked = np.vstack(all_features)
        else:
            stacked = all_features

        self.scaler = StandardScaler()
        self.scaler.fit(stacked)
        self.is_fitted = True
        print(f"[INFO] Scaler fitted on {stacked.shape[0]} frames")

    def transform(self, features):
        """스케일링 적용"""
        if not self.is_fitted:
            scaler = StandardScaler()
            return scaler.fit_transform(features)
        return self.scaler.transform(features)

    def save_scaler(self, path):
        """Scaler 저장"""
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'config': {
                    'sr': self.sr, 'n_mels': self.n_mels,
                    'n_fft': self.n_fft, 'hop_length': self.hop_length,
                    'window_size': self.window_size,
                }
            }, f)
        print(f"[INFO] Scaler saved: {path}")

    def load_scaler(self, path):
        """Scaler 로드"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.scaler = data['scaler']
        self.is_fitted = True
        print(f"[INFO] Scaler loaded: {path}")


# =============================================================================
# 데이터 준비
# =============================================================================
def load_label_entries(csv_path):
    """CSV에서 라벨링된 entries 로드"""
    entries = []
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV file not found: {csv_path}")
        return []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "video_id": row.get("video_id", "unknown"),
                "video_path": row["video_path"],
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "label": int(row["label"]),
            })
    print(f"[INFO] Loaded {len(entries)} entries from {csv_path}")
    return entries


def get_video_duration(video_path):
    """비디오의 fps, 총 프레임 수, 길이(초) 반환"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frames, frames / fps if fps > 0 else 0


def generate_negative_entries(positive_entries, num_neg_per_pos=1,
                              clip_duration=1.0, min_gap=0.1):
    """Positive 구간 사이에서 Negative 샘플 생성"""
    if not positive_entries:
        return []

    by_video = defaultdict(list)
    for e in positive_entries:
        by_video[e["video_path"]].append(e)

    all_negative = []

    for video_path, pos_list in by_video.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARNING] Cannot open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        cap.release()

        if video_duration < clip_duration:
            continue

        sorted_pos = sorted(pos_list, key=lambda x: x['start_sec'])
        video_id = sorted_pos[0].get('video_id', 'unknown')
        negative_candidates = []

        # 첫 번째 positive 이전 구간
        first_start = sorted_pos[0]['start_sec']
        if first_start > clip_duration + min_gap:
            t = 0.0
            while t + clip_duration <= first_start - min_gap:
                negative_candidates.append({
                    'video_id': video_id,
                    'video_path': video_path,
                    'start_sec': round(t, 3),
                    'end_sec': round(t + clip_duration, 3),
                    'label': 0
                })
                t += clip_duration

        # Positive 구간 사이사이
        for i in range(len(sorted_pos) - 1):
            gap_start = sorted_pos[i]['end_sec'] + min_gap
            gap_end = sorted_pos[i + 1]['start_sec'] - min_gap

            if gap_end - gap_start >= clip_duration:
                t = gap_start
                while t + clip_duration <= gap_end:
                    negative_candidates.append({
                        'video_id': video_id,
                        'video_path': video_path,
                        'start_sec': round(t, 3),
                        'end_sec': round(t + clip_duration, 3),
                        'label': 0
                    })
                    t += clip_duration

        # 마지막 positive 이후 구간
        last_end = sorted_pos[-1]['end_sec']
        if video_duration - last_end > clip_duration + min_gap:
            t = last_end + min_gap
            while t + clip_duration <= video_duration - min_gap:
                negative_candidates.append({
                    'video_id': video_id,
                    'video_path': video_path,
                    'start_sec': round(t, 3),
                    'end_sec': round(t + clip_duration, 3),
                    'label': 0
                })
                t += clip_duration

        target_num = len(pos_list) * num_neg_per_pos
        if len(negative_candidates) > target_num:
            sampled = random.sample(negative_candidates, target_num)
        else:
            sampled = negative_candidates

        all_negative.extend(sampled)

    print(f"[INFO] Generated {len(all_negative)} negative entries")
    return all_negative


def prepare_all_entries(csv_path, num_neg_per_pos=1, shuffle=True):
    """전체 entries 준비 (positive + negative)"""
    pos = load_label_entries(csv_path)
    neg = generate_negative_entries(pos, num_neg_per_pos)
    all_e = pos + neg
    if shuffle:
        random.shuffle(all_e)
    print(f"[INFO] Total entries: {len(all_e)} (pos: {len(pos)}, neg: {len(neg)})")
    return all_e


# =============================================================================
# Train/Val 분할
# =============================================================================
def train_val_split(entries, split_ratio=0.8):
    """단순 랜덤 분할"""
    if len(entries) < 2:
        return entries, []
    random.shuffle(entries)
    idx = int(len(entries) * split_ratio)
    if idx == 0:
        idx = 1
    return entries[:idx], entries[idx:]


def train_val_split_by_video(entries, split_ratio=0.8, seed=42):
    """비디오 단위로 분할 (데이터 누수 방지)"""
    if len(entries) < 2:
        return entries, []

    random.seed(seed)

    by_video = defaultdict(list)
    for e in entries:
        vid = e.get('video_id', e.get('video_path', 'unknown'))
        by_video[vid].append(e)

    video_ids = list(by_video.keys())
    random.shuffle(video_ids)

    split_idx = max(1, int(len(video_ids) * split_ratio))
    train_vids = set(video_ids[:split_idx])

    train_entries = []
    val_entries = []

    for vid, ents in by_video.items():
        if vid in train_vids:
            train_entries.extend(ents)
        else:
            val_entries.extend(ents)

    random.shuffle(train_entries)
    random.shuffle(val_entries)

    print(f"[INFO] Split by video: {len(train_vids)} train, {len(video_ids) - len(train_vids)} val")
    print(f"[INFO] Entries: {len(train_entries)} train, {len(val_entries)} val")

    return train_entries, val_entries


# =============================================================================
# Auto Labeling
# =============================================================================
def auto_label_frames(y_audio, sr=16000, n_fft=1024, hop_length=512, num_frames=None):
    """RMS/Spectral Centroid 기반 자동 라벨링"""
    rms = librosa.feature.rms(y=y_audio, frame_length=n_fft, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y_audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

    if num_frames is None:
        num_frames = len(rms)

    min_len = min(num_frames, len(rms), len(cent))
    rms = rms[:min_len]
    cent = cent[:min_len]

    rms_z = (rms - np.mean(rms)) / (np.std(rms) + 1e-6)
    cent_z = (cent - np.mean(cent)) / (np.std(cent) + 1e-6)

    labels = np.zeros(min_len, dtype=int)

    for i in range(len(labels)):
        if i > 5 and np.mean(rms_z[i-5:i]) < -0.5 and rms_z[i] > 0:
            labels[i] = 1  # Pause_Talk
        elif rms_z[i] > 1.5:
            labels[i] = 3  # Loud
        elif cent_z[i] > 1.5:
            labels[i] = 2  # High_Tone

    return labels


# =============================================================================
# 보컬 분리 (Demucs)
# =============================================================================
def extract_vocal_from_video(video_path, output_dir):
    """비디오에서 보컬 분리 (Demucs 사용)"""
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        print("[WARN] moviepy not installed. pip install moviepy")
        return None

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.splitext(os.path.basename(video_path))[0]
    temp_wav = os.path.join(output_dir, f"{filename}_temp.wav")
    final_vocal = os.path.join(output_dir, f"{filename}_vocals.wav")

    if os.path.exists(final_vocal):
        print(f"[INFO] 보컬 파일이 이미 존재합니다: {final_vocal}")
        return final_vocal

    # 비디오에서 오디오 추출
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(temp_wav, codec='pcm_s16le', verbose=False, logger=None)
        video.close()
    except Exception as e:
        print(f"[ERROR] 비디오 읽기 실패: {e}")
        return None

    # Demucs로 보컬 분리
    cmd = [sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems=vocals",
           temp_wav, "-o", output_dir]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if os.path.exists(temp_wav):
            os.rename(temp_wav, final_vocal)
            print(f"[WARN] Demucs 실패, 원본 오디오 사용: {final_vocal}")
            return final_vocal
        return None

    # 결과 파일 이동
    demucs_out = os.path.join(output_dir, "htdemucs", f"{filename}_temp", "vocals.wav")
    if os.path.exists(demucs_out):
        if os.path.exists(final_vocal):
            os.remove(final_vocal)
        os.replace(demucs_out, final_vocal)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        print(f"[INFO] 보컬 추출 완료: {final_vocal}")
        return final_vocal

    return None


# =============================================================================
# 오디오 분석
# =============================================================================
def analyze_audio_emphasis(audio_path, model, device, config=None):
    """오디오 파일에서 강조 패턴 분석"""
    if config is None:
        config = {
            'sample_rate': 16000, 'n_mels': 80, 'n_fft': 1024,
            'hop_length': 512, 'window_size': 50,
        }

    SR = config.get('sample_rate', 16000)
    N_MELS = config.get('n_mels', 80)
    N_FFT = config.get('n_fft', 1024)
    HOP_LENGTH = config.get('hop_length', 512)
    WINDOW_SIZE = config.get('window_size', 50)

    CLASS_NAMES = ['Normal', 'Pause_Talk', 'High_Tone', 'Loud']

    y, _ = librosa.load(audio_path, sr=SR)
    y = librosa.util.normalize(y)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    features = log_mel.T

    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    input_seqs = []
    for i in range(len(features_norm) - WINDOW_SIZE):
        input_seqs.append(features_norm[i: i + WINDOW_SIZE])

    if len(input_seqs) == 0:
        print("[WARN] 오디오 길이가 너무 짧습니다.")
        return []

    model.eval()
    predictions = []
    probabilities = []

    input_tensor = torch.FloatTensor(np.array(input_seqs)).to(device)
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i: i + batch_size]
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    results = postprocess_predictions(
        predictions, probabilities, HOP_LENGTH, SR, CLASS_NAMES
    )

    return results


def postprocess_predictions(predictions, probabilities, hop_length, sr, class_names,
                            min_duration=0.3, gap_limit_sec=0.3):
    """예측 결과 후처리 (빈틈 메우기 + 세그먼트 생성)"""
    GAP_LIMIT_FRAMES = int(gap_limit_sec / (hop_length / sr))
    time_per_frame = hop_length / sr

    smoothed_preds = np.array(predictions)
    i = 0
    while i < len(smoothed_preds):
        current = smoothed_preds[i]

        if current != 0:
            j = i + 1
            while j < len(smoothed_preds) and smoothed_preds[j] == current:
                j += 1

            next_same_start = -1
            for k in range(j, min(j + GAP_LIMIT_FRAMES, len(smoothed_preds))):
                if smoothed_preds[k] == current:
                    next_same_start = k
                    break

            if next_same_start != -1:
                smoothed_preds[j:next_same_start] = current
                i = j
                continue
        i += 1

    results = []
    probabilities = np.array(probabilities)

    current_label = smoothed_preds[0]
    start_frame = 0

    for i, pred in enumerate(smoothed_preds):
        if pred != current_label:
            if current_label != 0:
                start_time = start_frame * time_per_frame
                end_time = i * time_per_frame
                duration = end_time - start_time

                if duration >= min_duration:
                    segment_probs = np.mean(probabilities[start_frame:i], axis=0)
                    results.append({
                        "start_time": round(start_time, 2),
                        "end_time": round(end_time, 2),
                        "type": class_names[current_label],
                        "class_id": int(current_label),
                        "scores": {
                            class_names[j]: round(float(segment_probs[j]), 4)
                            for j in range(len(class_names))
                        }
                    })

            current_label = pred
            start_frame = i

    # 마지막 구간 처리
    if current_label != 0:
        start_time = start_frame * time_per_frame
        end_time = len(smoothed_preds) * time_per_frame
        duration = end_time - start_time

        if duration >= min_duration:
            segment_probs = np.mean(probabilities[start_frame:], axis=0)
            results.append({
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "type": class_names[current_label],
                "class_id": int(current_label),
                "scores": {
                    class_names[j]: round(float(segment_probs[j]), 4)
                    for j in range(len(class_names))
                }
            })

    return results


# =============================================================================
# Audio 데이터 로드 (학습용)
# =============================================================================
def load_all_data_for_audio(data_folder, config=None, window_size=50, stride=10):
    """폴더 내 모든 wav 파일 -> Mel-Spectrogram + Auto Labeling -> 윈도우 데이터 생성"""
    if config is None:
        config = {
            'sample_rate': 16000, 'n_mels': 80,
            'n_fft': 1024, 'hop_length': 512, 'window_size': window_size,
        }

    extractor = MelSpectrogramExtractor(config)

    all_files = [f for f in os.listdir(data_folder) if f.endswith('.wav')]

    if len(all_files) == 0:
        print(f"[ERROR] No wav files found in {data_folder}")
        return None, None, None, extractor

    print(f"[INFO] Found {len(all_files)} wav files")

    all_raw_features = []
    file_data = []

    for fname in all_files:
        try:
            path = os.path.join(data_folder, fname)
            y, _ = librosa.load(path, sr=config['sample_rate'])

            if len(y) < config['n_fft']:
                continue

            features = extractor.extract_features_raw(y)
            labels = auto_label_frames(
                y, config['sample_rate'], config['n_fft'],
                config['hop_length'], len(features)
            )

            min_len = min(len(features), len(labels))
            features = features[:min_len]
            labels = labels[:min_len]

            all_raw_features.append(features)
            file_data.append((features, labels))

        except Exception as e:
            print(f"[WARN] Error processing {fname}: {e}")
            continue

    if len(all_raw_features) == 0:
        print("[ERROR] No valid data!")
        return None, None, None, extractor

    extractor.fit_scaler(all_raw_features)

    X_list = []
    y_list = []

    for features, labels in file_data:
        features_scaled = extractor.transform(features)

        for i in range(0, len(features_scaled) - window_size, stride):
            window = features_scaled[i: i + window_size]
            label = labels[i + window_size - 1]

            X_list.append(window)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)

    print(f"[INFO] Dataset: X={X.shape}, y={y.shape}")
    print(f"[INFO] Class distribution: {np.bincount(y, minlength=4)}")

    return X, y, weights, extractor


# =============================================================================
# 텍스트 분석 데이터 로드
# =============================================================================
def load_text_analysis_data(tensors_path, scores_path):
    """미리 분석된 텍스트 특징 텐서와 LLM 점수 JSON 파일 로드"""
    text_tensors_list = []
    text_scores_json = {}

    if os.path.exists(tensors_path) and os.path.exists(scores_path):
        try:
            text_tensors_list = torch.load(tensors_path, map_location='cpu')
            with open(scores_path, 'r', encoding='utf-8') as f:
                text_scores_json = json.load(f)
            print(f"[INFO] Text analysis loaded: {len(text_tensors_list)} tensors")
        except Exception as e:
            print(f"[WARN] Text load error: {e}")

    return text_tensors_list, text_scores_json


# =============================================================================
# 학습/평가 루프
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """1 에폭 학습"""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """평가"""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


# =============================================================================
# 추론
# =============================================================================
def infer_video_emphasis(video_path, model, device, clip_len=16, stride=8,
                         resize_hw=(112, 112)):
    """Gesture 모델로 비디오 추론"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clip_infos = []
    start_frame = 0
    model.eval()

    while start_frame + clip_len <= total_frames:
        frames = []
        for f in range(start_frame, start_frame + clip_len):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), resize_hw)
            frames.append(frame)

        if len(frames) == clip_len:
            x = np.array(frames, dtype=np.float32) / 255.0
            x = np.transpose(x, (3, 0, 1, 2))
            x = (x - 0.5) / 0.5
            x = torch.from_numpy(x).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = F.softmax(model(x), dim=1)[0, 1].item()

            clip_infos.append({
                "start_sec": round(start_frame / fps, 2),
                "end_sec": round((start_frame + clip_len - 1) / fps, 2),
                "prob": round(prob, 4)
            })
        start_frame += stride

    cap.release()
    print(f"[INFO] Gesture inference: {len(clip_infos)} clips")
    return clip_infos


def infer_audio_emphasis(audio_path, model, device, extractor=None, config=None):
    """오디오 추론"""
    if config is None:
        config = {
            'sample_rate': 16000, 'n_mels': 80, 'n_fft': 1024,
            'hop_length': 512, 'window_size': 50,
        }
    return analyze_audio_emphasis(audio_path, model, device, config)


def infer_feature_fusion(video_path, gesture_model, audio_model, text_model, fusion_model,
                         device, text_tensors_list=None, text_scores_list=None,
                         clip_len=16, stride=8, audio_extractor=None, audio_scaler_path=None,
                         audio_config=None):
    """3-Modal Feature Fusion 추론"""
    if text_tensors_list is None:
        text_tensors_list = []
    if text_scores_list is None:
        text_scores_list = []

    if audio_config is None:
        audio_config = {
            'sample_rate': 16000, 'n_mels': 80, 'n_fft': 1024,
            'hop_length': 512, 'window_size': 50,
        }

    WINDOW_SIZE = audio_config.get('window_size', 50)

    if audio_extractor is None:
        audio_extractor = MelSpectrogramExtractor(audio_config)
        if audio_scaler_path and os.path.exists(audio_scaler_path):
            audio_extractor.load_scaler(audio_scaler_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    SR = audio_config.get('sample_rate', 16000)
    try:
        y_full, sr = librosa.load(video_path, sr=SR)
    except:
        y_full = np.zeros(int(total_frames / fps * SR))

    results = []

    gesture_model.eval()
    audio_model.eval()
    text_model.eval()
    fusion_model.eval()

    current_text_idx = 0
    start_frame = 0

    print(f"[INFO] Feature fusion inference started: {os.path.basename(video_path)}")

    with torch.no_grad():
        while start_frame + clip_len <= total_frames:
            start_sec = start_frame / fps
            end_sec = (start_frame + clip_len) / fps

            # 1. Gesture
            frames = []
            for i in range(start_frame, start_frame + clip_len):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (112, 112))
                frames.append(frame)

            if len(frames) < clip_len:
                break

            vid_tensor = np.array(frames, dtype=np.float32) / 255.0
            vid_tensor = np.transpose(vid_tensor, (3, 0, 1, 2))
            vid_tensor = (vid_tensor - 0.5) / 0.5
            vid_tensor = torch.from_numpy(vid_tensor).unsqueeze(0).to(device)

            # 2. Audio
            raw = audio_extractor.extract_features_raw(y_full, start_sec, end_sec)
            if raw is not None:
                features = audio_extractor.pad_or_truncate(raw)
                if audio_extractor.is_fitted:
                    features = audio_extractor.transform(features)
                aud_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
            else:
                aud_tensor = torch.zeros(1, WINDOW_SIZE, 80).to(device)

            # 3. Text
            t_feat = torch.zeros(1, 768).to(device)
            t_prob = 0.0

            if current_text_idx < len(text_scores_list):
                item = text_scores_list[current_text_idx]
                try:
                    time_range = item.get('time', '0.0~0.0').split('~')
                    sent_start = float(time_range[0])
                    sent_end = float(time_range[1])
                except:
                    sent_start, sent_end = 0.0, 0.0

                if start_sec >= sent_end and current_text_idx < len(text_scores_list) - 1:
                    current_text_idx += 1
                elif sent_start <= start_sec < sent_end:
                    t_prob = float(item.get('emphasis_score', 0.0))
                    if current_text_idx < len(text_tensors_list) and text_tensors_list[current_text_idx] is not None:
                        t_tensor_full = text_tensors_list[current_text_idx].to(device)
                        t_feat = torch.mean(t_tensor_full, dim=1) if t_tensor_full.dim() == 3 else t_tensor_full

            # 4. Feature Fusion
            _, g_feat = gesture_model(vid_tensor, return_feature=True)
            _, a_feat = audio_model(aud_tensor, return_feature=True)

            f_logits = fusion_model(g_feat, a_feat, t_feat)

            g_prob = F.softmax(gesture_model(vid_tensor), dim=1)[0, 1].item()

            a_logits_full = audio_model(aud_tensor)
            a_probs = F.softmax(a_logits_full, dim=1)[0]
            a_class = torch.argmax(a_probs).item()
            a_prob = 1.0 - a_probs[0].item()

            f_prob = torch.sigmoid(f_logits)[0, 0].item()

            results.append({
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "scores": {
                    "gesture": round(g_prob, 4),
                    "audio": round(a_prob, 4),
                    "audio_class": a_class,
                    "text": round(t_prob, 4),
                    "fusion": round(f_prob, 4)
                }
            })

            start_frame += stride

    cap.release()
    print(f"[INFO] Feature fusion done: {len(results)} clips")
    return results


def merge_clips_to_segments(results, threshold=0.5, merge_gap=0.5, min_duration=0.0):
    """연속 클립을 세그먼트로 병합"""
    if not results:
        return []

    segments = []
    in_segment = False
    seg_start = 0

    for r in results:
        prob = r.get("emphasis_prob", r.get("fusion", 0))

        if prob >= threshold and not in_segment:
            seg_start = r["start_sec"]
            in_segment = True
        elif prob < threshold and in_segment:
            segments.append((seg_start, r["start_sec"]))
            in_segment = False

    if in_segment:
        segments.append((seg_start, results[-1]["end_sec"]))

    # 병합
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    return [s for s in merged if (s[1] - s[0]) >= min_duration]


def visualize_emphasis_on_video(video_path, segments, out_path=None, show_window=False):
    """강조 구간 시각화 영상 생성"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    def get_segment_info(seg):
        if isinstance(seg, tuple):
            return seg[0], seg[1], "EMPHASIS"
        elif isinstance(seg, dict):
            return seg.get("start_sec", seg.get("start_time", 0)), \
                   seg.get("end_sec", seg.get("end_time", 0)), \
                   seg.get("type", seg.get("class_name", "EMPHASIS"))
        return 0, 0, "UNKNOWN"

    def is_emph(t):
        for seg in segments:
            s, e, name = get_segment_info(seg)
            if s <= t <= e:
                return True, name
        return False, None

    if show_window:
        cv2.namedWindow("Emphasis", cv2.WINDOW_NORMAL)

    frame_idx = 0
    print(f"[INFO] Processing video... (output: {out_path})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        emph, class_name = is_emph(t)

        color = (0, 0, 255) if emph else (0, 128, 0)
        text = class_name if emph else "NORMAL"

        cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        if show_window:
            cv2.imshow("Emphasis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if writer:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print("[INFO] Video saved!")
    if show_window:
        cv2.destroyAllWindows()
