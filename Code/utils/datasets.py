"""
datasets.py - 멀티모달 데이터셋 클래스
기존 파일에 다음 클래스들을 추가하세요.
"""
import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from sklearn.preprocessing import StandardScaler
import pickle


# =============================================================================
# ⭐ Audio Feature Extractor (Scaler 포함) - 새로 추가
# =============================================================================
class AudioFeatureExtractor:
    """
    Mel-Spectrogram 특징 추출기 + StandardScaler
    학습 시 fit(), 추론 시 transform() 사용
    """
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
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_features_raw(self, y_audio):
        """원본 Mel-Spectrogram 추출 (스케일링 전)"""
        if len(y_audio) == 0:
            return np.zeros((self.window_size, self.n_mels))
        
        y_audio = librosa.util.normalize(y_audio)
        
        mel = librosa.feature.melspectrogram(
            y=y_audio, 
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        features = librosa.power_to_db(mel, ref=np.max).T
        
        return features
    
    def pad_or_truncate(self, features):
        """고정 길이로 패딩/자르기"""
        if features.shape[0] < self.window_size:
            pad_len = self.window_size - features.shape[0]
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
        else:
            features = features[:self.window_size, :]
        return features
    
    def fit(self, all_features):
        """전체 학습 데이터로 Scaler fit"""
        concat = np.vstack(all_features)
        self.scaler.fit(concat)
        self.is_fitted = True
        print(f"✅ AudioFeatureExtractor fitted on {len(concat)} frames")
    
    def transform(self, features):
        """fit된 Scaler로 transform"""
        if not self.is_fitted:
            raise RuntimeError("Scaler가 fit되지 않았습니다.")
        return self.scaler.transform(features)
    
    def extract_and_process(self, y_audio):
        """전체 파이프라인: 추출 -> 패딩 -> 스케일링"""
        raw = self.extract_features_raw(y_audio)
        padded = self.pad_or_truncate(raw)
        
        if self.is_fitted:
            scaled = self.transform(padded)
        else:
            scaled = padded
        
        return scaled
    
    def save(self, path):
        """Scaler 저장"""
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'config': {
                    'sr': self.sr,
                    'n_mels': self.n_mels,
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'window_size': self.window_size,
                },
                'is_fitted': self.is_fitted
            }, f)
        print(f"✅ AudioFeatureExtractor 저장: {path}")
    
    def load(self, path):
        """Scaler 로드"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        config = data['config']
        self.sr = config['sr']
        self.n_mels = config['n_mels']
        self.n_fft = config['n_fft']
        self.hop_length = config['hop_length']
        self.window_size = config['window_size']
        print(f"✅ AudioFeatureExtractor 로드: {path}")


# =============================================================================
# 제스처(영상) Dataset (기존과 동일하게 유지)
# =============================================================================
class VideoEmphasisDataset(Dataset):
    def __init__(self, entries, clip_len=16, resize_hw=(112, 112)):
        self.entries = entries
        self.clip_len = clip_len
        self.resize_hw = resize_hw

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        video_path = entry["video_path"]
        start_sec = entry["start_sec"]
        end_sec = entry["end_sec"]
        label = entry["label"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            frames = np.zeros((3, self.clip_len, self.resize_hw[0], self.resize_hw[1]), dtype=np.float32)
            return torch.from_numpy(frames), torch.tensor(label, dtype=torch.long)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))

        if end_frame <= start_frame:
            end_frame = min(start_frame + self.clip_len, total_frames - 1)

        seg_len = end_frame - start_frame + 1

        if seg_len >= self.clip_len:
            indices = np.linspace(start_frame, end_frame, num=self.clip_len, dtype=int)
        else:
            base_indices = np.linspace(start_frame, end_frame, num=seg_len, dtype=int)
            if seg_len > 0:
                repeat = int(np.ceil(self.clip_len / seg_len))
                indices = np.tile(base_indices, repeat)[:self.clip_len]
            else:
                indices = np.zeros(self.clip_len, dtype=int)

        frames = []
        for f_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                if len(frames) > 0:
                    frame = frames[-1]
                else:
                    frame = np.zeros((self.resize_hw[0], self.resize_hw[1], 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.resize_hw)
            frames.append(frame)

        cap.release()

        frames = np.stack(frames, axis=0)
        frames = frames.astype(np.float32) / 255.0
        frames = np.transpose(frames, (3, 0, 1, 2))
        frames = (frames - 0.5) / 0.5

        x = torch.from_numpy(frames)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# =============================================================================
# 오디오 Dataset (기존과 동일하게 유지)
# =============================================================================
class AudioEmphasisDataset(Dataset):
    def __init__(self, entries, audio_dir=None, extractor=None, config=None):
        self.entries = entries
        self.audio_dir = audio_dir
        self.extractor = extractor

        if config is None:
            config = {
                'sample_rate': 16000,
                'n_mels': 80,
                'window_size': 50,
                'n_fft': 1024,
                'hop_length': 512,
            }

        self.sr = config.get('sample_rate', 16000)
        self.n_mels = config.get('n_mels', 80)
        self.window_size = config.get('window_size', 50)
        self.n_fft = config.get('n_fft', 1024)
        self.hop_length = config.get('hop_length', 512)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        video_path = entry["video_path"]
        start_sec = entry["start_sec"]
        end_sec = entry["end_sec"]
        label = entry["label"]

        if self.audio_dir:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(self.audio_dir, f"{video_name}.wav")
        else:
            audio_path = video_path

        try:
            duration = max(end_sec - start_sec, 0.1)
            y_audio, sr = librosa.load(audio_path, sr=self.sr, offset=start_sec, duration=duration)

            if len(y_audio) > 0 and self.extractor is not None:
                raw = self.extractor.extract_features_raw(y_audio)
                features = self.extractor.pad_or_truncate(raw)

                if self.extractor.is_fitted:
                    features = self.extractor.transform(features)

                x = torch.from_numpy(features).float()

            elif len(y_audio) > 0:
                y_audio = librosa.util.normalize(y_audio)
                
                mel = librosa.feature.melspectrogram(
                    y=y_audio, sr=sr, 
                    n_mels=self.n_mels,
                    n_fft=self.n_fft, 
                    hop_length=self.hop_length
                )
                features = librosa.power_to_db(mel, ref=np.max).T
                
                if features.shape[0] < self.window_size:
                    pad_len = self.window_size - features.shape[0]
                    features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
                else:
                    features = features[:self.window_size, :]

                x = torch.from_numpy(features).float()
            else:
                x = torch.zeros(self.window_size, self.n_mels)

        except Exception as e:
            x = torch.zeros(self.window_size, self.n_mels)

        y = torch.tensor(label, dtype=torch.long)
        return x, y


# =============================================================================
# 텍스트 Dataset (기존과 동일하게 유지)
# =============================================================================
class TextEmphasisDataset(Dataset):
    def __init__(self, entries, stt_dir=None, tokenizer=None, max_length=128):
        self.entries = entries
        self.stt_dir = stt_dir
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.text_dim = 1024

        self.stt_data = self._load_stt_data()

    def _load_stt_data(self):
        stt_data = {}

        if self.stt_dir is None or not os.path.exists(self.stt_dir):
            return stt_data

        for fname in os.listdir(self.stt_dir):
            if fname.endswith('.json'):
                fpath = os.path.join(self.stt_dir, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                video_id = os.path.splitext(fname)[0]
                stt_data[video_id] = data

        print(f"Loaded STT data for {len(stt_data)} videos")
        return stt_data

    def _get_text_for_segment(self, video_path, start_sec, end_sec):
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        if video_id not in self.stt_data:
            return ""

        stt_info = self.stt_data[video_id]

        texts = []
        for seg in stt_info.get("segments", []):
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)

            if not (seg_end <= start_sec or seg_start >= end_sec):
                texts.append(seg.get("text", ""))

        return " ".join(texts)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        video_path = entry["video_path"]
        start_sec = entry["start_sec"]
        end_sec = entry["end_sec"]
        label = entry["label"]

        x = torch.zeros(self.text_dim)

        if self.stt_dir:
            text = self._get_text_for_segment(video_path, start_sec, end_sec)

        y = torch.tensor(label, dtype=torch.long)
        return x, y


# =============================================================================
# ⭐ Fusion Dataset (새로 추가) - 멀티모달 통합
# =============================================================================
class FusionDataset(Dataset):
    """
    Video + Audio + Text를 한 번에 반환하는 Dataset
    """
    def __init__(self, entries, audio_extractor=None, 
                 gesture_config=None, audio_config=None, stt_dir=None):
        self.entries = entries
        
        if gesture_config is None:
            gesture_config = {'clip_len': 16, 'resize_hw': (112, 112)}
        if audio_config is None:
            audio_config = {'sample_rate': 16000, 'n_mels': 80, 'window_size': 50}
        
        self.video_dataset = VideoEmphasisDataset(
            entries,
            clip_len=gesture_config.get('clip_len', 16),
            resize_hw=gesture_config.get('resize_hw', (112, 112))
        )
        self.audio_dataset = AudioEmphasisDataset(
            entries,
            extractor=audio_extractor,
            config=audio_config
        )
        self.text_dataset = TextEmphasisDataset(entries, stt_dir=stt_dir)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        X_vid, y_vid = self.video_dataset[idx]
        X_aud, y_aud = self.audio_dataset[idx]
        X_txt, y_txt = self.text_dataset[idx]

        label = y_vid.float() if isinstance(y_vid, torch.Tensor) else torch.tensor(y_vid, dtype=torch.float)

        return X_vid, X_aud, X_txt, label


# =============================================================================
# TensorDataset 래퍼 (기존과 동일)
# =============================================================================
class MelSpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# import os
# import cv2
# import json
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import librosa
# from sklearn.preprocessing import StandardScaler

# # =============================================================================
# # Audio Feature Extractor (Scaler 포함)
# # =============================================================================
# class AudioFeatureExtractor:
#     """
#     Mel-Spectrogram 특징 추출기 + StandardScaler
#     학습 시 fit(), 추론 시 transform() 사용
#     """
#     def __init__(self, config=None):
#         if config is None:
#             config = {
#                 'sample_rate': 16000,
#                 'n_mels': 80,
#                 'n_fft': 1024,
#                 'hop_length': 512,
#                 'window_size': 50,
#             }
        
#         self.sr = config.get('sample_rate', 16000)
#         self.n_mels = config.get('n_mels', 80)
#         self.n_fft = config.get('n_fft', 1024)
#         self.hop_length = config.get('hop_length', 512)
#         self.window_size = config.get('window_size', 50)
        
#         self.scaler = StandardScaler()
#         self.is_fitted = False
    
#     def extract_features_raw(self, y_audio):
#         """
#         원본 Mel-Spectrogram 추출 (스케일링 전)
#         """
#         if len(y_audio) == 0:
#             return np.zeros((self.window_size, self.n_mels))
        
#         # 정규화
#         y_audio = librosa.util.normalize(y_audio)
        
#         # Mel-Spectrogram
#         mel = librosa.feature.melspectrogram(
#             y=y_audio, 
#             sr=self.sr,
#             n_mels=self.n_mels,
#             n_fft=self.n_fft,
#             hop_length=self.hop_length
#         )
#         features = librosa.power_to_db(mel, ref=np.max).T  # (Time, n_mels)
        
#         return features
    
#     def pad_or_truncate(self, features):
#         """
#         고정 길이로 패딩/자르기
#         """
#         if features.shape[0] < self.window_size:
#             pad_len = self.window_size - features.shape[0]
#             features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
#         else:
#             features = features[:self.window_size, :]
#         return features
    
#     def fit(self, all_features):
#         """
#         전체 학습 데이터로 Scaler fit
#         all_features: list of (Time, n_mels) arrays
#         """
#         # 모든 프레임을 하나로 합쳐서 fit
#         concat = np.vstack(all_features)
#         self.scaler.fit(concat)
#         self.is_fitted = True
#         print(f"✅ AudioFeatureExtractor fitted on {len(concat)} frames")
    
#     def transform(self, features):
#         """
#         fit된 Scaler로 transform
#         """
#         if not self.is_fitted:
#             raise RuntimeError("Scaler가 fit되지 않았습니다. fit()을 먼저 호출하세요.")
#         return self.scaler.transform(features)
    
#     def fit_transform(self, features):
#         """
#         fit + transform (단일 샘플용 - 비추천)
#         """
#         return self.scaler.fit_transform(features)
    
#     def extract_and_process(self, y_audio):
#         """
#         전체 파이프라인: 추출 -> 패딩 -> 스케일링
#         """
#         raw = self.extract_features_raw(y_audio)
#         padded = self.pad_or_truncate(raw)
        
#         if self.is_fitted:
#             scaled = self.transform(padded)
#         else:
#             # fit 안됐으면 그냥 반환 (추론 시에는 반드시 fit 필요)
#             scaled = padded
        
#         return scaled
    
#     def save(self, path):
#         """Scaler 저장"""
#         import pickle
#         with open(path, 'wb') as f:
#             pickle.dump({
#                 'scaler': self.scaler,
#                 'config': {
#                     'sr': self.sr,
#                     'n_mels': self.n_mels,
#                     'n_fft': self.n_fft,
#                     'hop_length': self.hop_length,
#                     'window_size': self.window_size,
#                 },
#                 'is_fitted': self.is_fitted
#             }, f)
#         print(f"✅ AudioFeatureExtractor 저장: {path}")
    
#     def load(self, path):
#         """Scaler 로드"""
#         import pickle
#         with open(path, 'rb') as f:
#             data = pickle.load(f)
#         self.scaler = data['scaler']
#         self.is_fitted = data['is_fitted']
#         config = data['config']
#         self.sr = config['sr']
#         self.n_mels = config['n_mels']
#         self.n_fft = config['n_fft']
#         self.hop_length = config['hop_length']
#         self.window_size = config['window_size']
#         print(f"✅ AudioFeatureExtractor 로드: {path}")
# # =============================================================================
# # 제스처(영상) Dataset
# # =============================================================================
# class VideoEmphasisDataset(Dataset):
#     def __init__(self, entries, clip_len=16, resize_hw=(112, 112)):
#         self.entries = entries
#         self.clip_len = clip_len
#         self.resize_hw = resize_hw

#     def __len__(self):
#         return len(self.entries)

#     def __getitem__(self, idx):
#         entry = self.entries[idx]
#         video_path = entry["video_path"]
#         start_sec = entry["start_sec"]
#         end_sec = entry["end_sec"]
#         label = entry["label"]

#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             frames = np.zeros((3, self.clip_len, self.resize_hw[0], self.resize_hw[1]), dtype=np.float32)
#             return torch.from_numpy(frames), torch.tensor(label, dtype=torch.long)

#         fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         start_frame = int(start_sec * fps)
#         end_frame = int(end_sec * fps)

#         start_frame = max(0, min(start_frame, total_frames - 1))
#         end_frame = max(0, min(end_frame, total_frames - 1))

#         if end_frame <= start_frame:
#             end_frame = min(start_frame + self.clip_len, total_frames - 1)

#         seg_len = end_frame - start_frame + 1

#         if seg_len >= self.clip_len:
#             indices = np.linspace(start_frame, end_frame, num=self.clip_len, dtype=int)
#         else:
#             base_indices = np.linspace(start_frame, end_frame, num=seg_len, dtype=int)
#             if seg_len > 0:
#                 repeat = int(np.ceil(self.clip_len / seg_len))
#                 indices = np.tile(base_indices, repeat)[:self.clip_len]
#             else:
#                 indices = np.zeros(self.clip_len, dtype=int)

#         frames = []
#         for f_idx in indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
#             ret, frame = cap.read()
#             if not ret:
#                 if len(frames) > 0:
#                     frame = frames[-1]
#                 else:
#                     frame = np.zeros((self.resize_hw[0], self.resize_hw[1], 3), dtype=np.uint8)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, self.resize_hw)
#             frames.append(frame)

#         cap.release()

#         # 텐서 변환 (한 번만!)
#         frames = np.stack(frames, axis=0)            # (T, H, W, C)
#         frames = frames.astype(np.float32) / 255.0   # [0, 1]
#         frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
#         frames = (frames - 0.5) / 0.5                # [-1, 1]

#         x = torch.from_numpy(frames)
#         y = torch.tensor(label, dtype=torch.long)
#         return x, y


# # =============================================================================
# # 오디오 Dataset
# # =============================================================================
# class AudioEmphasisDataset(Dataset):
#     def __init__(self, entries, audio_dir=None, extractor=None, config=None):
     
#         self.entries = entries
#         self.audio_dir = audio_dir
#         self.extractor = extractor  # ⭐ 외부에서 주입!

#         # config에서 설정 가져오기
#         if config is None:
#             config = {
#                 'sample_rate': 16000,
#                 'n_mels': 80,
#                 'window_size': 50,
#                 'n_fft': 1024,
#                 'hop_length': 512,
#             }

#         self.sr = config.get('sample_rate', 16000)
#         self.n_mels = config.get('n_mels', 80)
#         self.window_size = config.get('window_size', 50)
#         self.n_fft = config.get('n_fft', 1024)
#         self.hop_length = config.get('hop_length', 512)

#     def __len__(self):
#         return len(self.entries)

#     def __getitem__(self, idx):
#         entry = self.entries[idx]
#         video_path = entry["video_path"]
#         start_sec = entry["start_sec"]
#         end_sec = entry["end_sec"]
#         label = entry["label"]

#         # 오디오 경로 결정
#         if self.audio_dir:
#             video_name = os.path.splitext(os.path.basename(video_path))[0]
#             audio_path = os.path.join(self.audio_dir, f"{video_name}.wav")
#         else:
#             audio_path = video_path

#         try:
#             duration = max(end_sec - start_sec, 0.1)
#             y_audio, sr = librosa.load(audio_path, sr=self.sr, offset=start_sec, duration=duration)

#             if len(y_audio) > 0 and self.extractor is not None:
#                 # extractor 사용 Scaler 포함
#                 raw = self.extractor.extract_features_raw(y_audio)
#                 features = self.extractor.pad_or_truncate(raw)

#                 # fit된 Scaler로 transform
#                 if self.extractor.is_fitted:
#                     features = self.extractor.transform(features)

#                 x = torch.from_numpy(features).float()

#             elif len(y_audio) > 0:
#                 #  Mel-Spectrogram 직접 추출
#                 y_audio = librosa.util.normalize(y_audio)
                
#                 mel = librosa.feature.melspectrogram(
#                     y=y_audio, sr=sr, 
#                     n_mels=self.n_mels,      
#                     n_fft=self.n_fft, 
#                     hop_length=self.hop_length
#                 )
#                 features = librosa.power_to_db(mel, ref=np.max).T  # (Time, 80)
                
#                 scaler = StandardScaler()
#                 features = scaler.fit_transform(features)
                
#                 # 패딩/자르기
#                 if features.shape[0] < self.window_size:
#                     pad_len = self.window_size - features.shape[0]
#                     features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
#                 else:
#                     features = features[:self.window_size, :]

#                 x = torch.from_numpy(features).float()
#             else:
#                 x = torch.zeros(self.window_size, self.n_mels)  

#         except Exception as e:
#             x = torch.zeros(self.window_size, self.n_mels)  

#         y = torch.tensor(label, dtype=torch.long)
#         return x, y


# # =============================================================================
# # 텍스트 Dataset
# # =============================================================================
# class TextEmphasisDataset(Dataset):
#     def __init__(self, entries, stt_dir=None, tokenizer=None, max_length=128):
#         self.entries = entries
#         self.stt_dir = stt_dir
#         self.max_length = max_length
#         self.tokenizer = tokenizer

#         # STT 결과 로드
#         self.stt_data = self._load_stt_data()

#     def _load_stt_data(self):
#         """STT JSON 파일들 로드"""
#         stt_data = {}

#         if self.stt_dir is None or not os.path.exists(self.stt_dir):
#             return stt_data

#         for fname in os.listdir(self.stt_dir):
#             if fname.endswith('.json'):
#                 fpath = os.path.join(self.stt_dir, fname)
#                 with open(fpath, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
#                 video_id = os.path.splitext(fname)[0]
#                 stt_data[video_id] = data

#         print(f"Loaded STT data for {len(stt_data)} videos")
#         return stt_data

#     def _get_text_for_segment(self, video_path, start_sec, end_sec):
#         """해당 시간 구간의 텍스트 추출"""
#         video_id = os.path.splitext(os.path.basename(video_path))[0]

#         if video_id not in self.stt_data:
#             return ""

#         stt_info = self.stt_data[video_id]

#         texts = []
#         for seg in stt_info.get("segments", []):
#             seg_start = seg.get("start", 0)
#             seg_end = seg.get("end", 0)

#             if not (seg_end <= start_sec or seg_start >= end_sec):
#                 texts.append(seg.get("text", ""))

#         return " ".join(texts)

#     def __len__(self):
#         return len(self.entries)

#     def __getitem__(self, idx):
#         entry = self.entries[idx]
#         video_path = entry["video_path"]
#         start_sec = entry["start_sec"]
#         end_sec = entry["end_sec"]
#         label = entry["label"]

#         # 텍스트용 더미 특징 (1024차원) - 실제 BERT 사용 시 교체
#         x = torch.zeros(1024)

#         # STT 데이터가 있으면 텍스트 추출 (나중에 BERT 토큰화에 사용)
#         if self.stt_dir:
#             text = self._get_text_for_segment(video_path, start_sec, end_sec)
#             # TODO: tokenizer로 실제 토큰화 구현

#         y = torch.tensor(label, dtype=torch.long)
#         return x, y
# # =============================================================================
# #  TensorDataset 래퍼
# # =============================================================================
# class MelSpectrogramDataset(Dataset):
    
#     def __init__(self, X, y):
       
#         self.X = torch.FloatTensor(X)
#         self.y = torch.LongTensor(y)
    
#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]