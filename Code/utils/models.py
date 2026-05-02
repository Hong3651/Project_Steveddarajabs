"""
models.py - 멀티모달 모델 정의 및 유틸리티
"""
import os
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models.video import r3d_18, R3D_18_Weights


# =============================================================================
# 공통 유틸: 안전한 가중치 로드 함수
# =============================================================================
def load_state_dict_safe(model, path, device):
    """DataParallel로 저장된 모델(module. prefix)도 안전하게 로드"""
    if not os.path.exists(path):
        print(f"  ⚠️ 가중치 파일 없음: {path}")
        return model

    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}

    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict, strict=False)
        print(f"  ✅ 모델 로드 성공: {os.path.basename(path)}")
    except Exception as e:
        print(f"  ❌ 모델 로드 중 에러 발생: {e}")

    return model


# =============================================================================
# 1. 제스처 모델 (Video - R3D-18)
# =============================================================================
class GestureModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(GestureModel, self).__init__()
        try:
            weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
            self.backbone = r3d_18(weights=weights)
        except:
            self.backbone = r3d_18(pretrained=pretrained)

        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)
        self.feature_dim = in_features

    def forward(self, x, return_feature=False):
        features = self.backbone(x)  # (B, 512)
        logits = self.classifier(features)
        if return_feature:
            return logits, features
        return logits


def create_gesture_model(num_classes=2, pretrained=True):
    return GestureModel(num_classes=num_classes, pretrained=pretrained)


def load_gesture_model(path, device, num_classes=2):
    model = create_gesture_model(num_classes=num_classes)
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        new_state = {}
        for k, v in state.items():
            k_new = k[7:] if k.startswith("module.") else k
            if 'fc' in k_new and 'backbone' not in k_new:
                k_new = k_new.replace('fc', 'classifier')
            new_state[k_new] = v

        try:
            model.load_state_dict(new_state, strict=False)
            print(f"  ✅ Gesture Model 로드 완료: {path}")
        except Exception as e:
            print(f"  ❌ Gesture Model 로드 실패: {e}")
    else:
        print(f"  ⚠️ Gesture Model 파일 없음: {path}")

    model.to(device).eval()
    return model


# =============================================================================
# 2. 오디오 모델 (Audio BiLSTM)
# =============================================================================
class AudioLSTMModel(nn.Module):
    def __init__(self, input_size=80, hidden_size=64, num_layers=2,
                 num_classes=4, dropout=0.2):
        super(AudioLSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.feature_dim = hidden_size * 2  # 128 (BiLSTM)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.use_simple_fc = True

    def forward(self, x, return_feature=False):
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]

        if self.use_simple_fc:
            logits = self.fc(features)
        else:
            logits = self.classifier(features)

        if return_feature:
            return logits, features
        return logits


class BiLSTM(nn.Module):
    def __init__(self, input_dim=80, hidden_size=64, num_classes=4):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.feature_dim = hidden_size * 2

    def forward(self, x, return_feature=False):
        out, _ = self.lstm(x)
        features = out[:, -1, :]
        logits = self.fc(features)
        if return_feature:
            return logits, features
        return logits


def create_audio_model(input_size=80, num_classes=4, hidden_size=64,
                       num_layers=2, dropout=0.2):
    return AudioLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )


def load_audio_model(path, device, input_size=80, num_classes=4,
                     hidden_size=64, num_layers=2, dropout=0.3):
    model = create_audio_model(input_size=input_size, num_classes=num_classes)

    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                new_state[k[7:] if k.startswith("module.") else k] = v
            model.load_state_dict(new_state, strict=False)
            print(f"  ✅ Audio Model 로드 완료: {path}")
        else:
            print(f"  ⚠️ Invalid state format: {path}")
    else:
        print(f"  ⚠️ Audio Model 파일 없음: {path}")

    return model.to(device).eval()


AUDIO_CLASS_NAMES = ['Normal', 'Pause_Talk', 'High_Tone', 'Loud']


# =============================================================================
# 3. 텍스트 모델 (Text - BERT Embedding Classifier)
# =============================================================================
class TextBERTModel(nn.Module):
    def __init__(self, input_dim=1024, num_classes=2):
        super(TextBERTModel, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.feature_dim = input_dim

    def forward(self, x, return_feature=False):
        logits = self.fc(x)
        if return_feature:
            return logits, x
        return logits


def create_text_model(model_name="klue/roberta-large", num_classes=2):
    input_dim = 1024 if "large" in model_name else 768
    print(f"  ℹ️ Text Model: {model_name} (Dim: {input_dim})")
    return TextBERTModel(input_dim=input_dim, num_classes=num_classes)


def load_text_model(path, device, model_name="klue/roberta-large"):
    model = create_text_model(model_name)
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"  ✅ Text Model 로드 완료: {path}")
    else:
        print(f"  ⚠️ Text Model 파일 없음 (더미 사용): {path}")
    model.to(device).eval()
    return model


# =============================================================================
# 4. Fusion MLP (텍스트 포함)
# =============================================================================
class FusionMLP(nn.Module):
    def __init__(self, gesture_dim=512, audio_dim=128, text_dim=1024,
                 hidden_dim=256, num_classes=1, dropout=0.3, use_text=True):
        super(FusionMLP, self).__init__()

        self.use_text = use_text
        input_dim = gesture_dim + audio_dim + (text_dim if use_text else 0)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, gesture_feat, audio_feat, text_feat=None):
        if self.use_text and text_feat is not None:
            combined = torch.cat((gesture_feat, audio_feat, text_feat), dim=1)
        else:
            combined = torch.cat((gesture_feat, audio_feat), dim=1)
        return self.net(combined)


# =============================================================================
# 5. Gated Fusion
# =============================================================================
class GatedFusion(nn.Module):
    def __init__(self, gesture_dim=512, audio_dim=128, text_dim=1024,
                 hidden_dim=256, num_classes=1, use_text=True, dropout=0.5):
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


# =============================================================================
# 6. Transformer Fusion
# =============================================================================
class TransformerFusion(nn.Module):
    def __init__(self, gesture_dim=512, audio_dim=128, text_dim=1024,
                 hidden_dim=256, num_classes=1, use_text=True,
                 dropout=0.5, nhead=4, num_layers=2):
        super(TransformerFusion, self).__init__()
        self.use_text = use_text

        self.g_proj = nn.Linear(gesture_dim, hidden_dim)
        self.a_proj = nn.Linear(audio_dim, hidden_dim)
        if use_text:
            self.t_proj = nn.Linear(text_dim, hidden_dim)
            self.num_tokens = 3
        else:
            self.num_tokens = 2

        self.modality_embed = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_tokens, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, gesture, audio, text=None):
        g_emb = self.g_proj(gesture).unsqueeze(1)
        a_emb = self.a_proj(audio).unsqueeze(1)
        tokens = [g_emb, a_emb]

        if self.use_text and text is not None:
            t_emb = self.t_proj(text).unsqueeze(1)
            tokens.append(t_emb)

        x = torch.cat(tokens, dim=1)
        x = x + self.modality_embed
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)

        return self.classifier(x)


# =============================================================================
# 7. End-to-End Multimodal Fusion Model
# =============================================================================
class MultimodalFusionModel(nn.Module):
    def __init__(self, gesture_model, audio_model, fusion_mlp, text_model=None):
        super(MultimodalFusionModel, self).__init__()
        self.gesture_model = gesture_model
        self.audio_model = audio_model
        self.text_model = text_model
        self.fusion_mlp = fusion_mlp

    def forward(self, video_input, audio_input, text_input=None):
        _, gesture_feat = self.gesture_model(video_input, return_feature=True)
        _, audio_feat = self.audio_model(audio_input, return_feature=True)

        if self.text_model is not None and text_input is not None:
            _, text_feat = self.text_model(text_input, return_feature=True)
        else:
            text_feat = torch.zeros(gesture_feat.size(0), 1024).to(gesture_feat.device)

        return self.fusion_mlp(gesture_feat, audio_feat, text_feat)


# =============================================================================
# Factory Functions
# =============================================================================
def create_fusion_model(gesture_dim=512, audio_dim=128, text_dim=1024,
                        hidden_dim=256, num_classes=1, use_text=True,
                        dropout=0.5, model_type='gated'):
    """
    Fusion 모델 생성
    model_type: 'mlp', 'gated', 'transformer'
    """
    if model_type == 'transformer':
        return TransformerFusion(
            gesture_dim, audio_dim, text_dim, hidden_dim, num_classes,
            use_text=use_text, dropout=dropout
        )
    elif model_type == 'gated':
        return GatedFusion(
            gesture_dim, audio_dim, text_dim, hidden_dim, num_classes,
            use_text=use_text, dropout=dropout
        )
    else:
        return FusionMLP(
            gesture_dim, audio_dim, text_dim, hidden_dim, num_classes,
            dropout=dropout, use_text=use_text
        )


def load_fusion_model(path, device, model_type='gated', **kwargs):
    """저장된 Fusion 모델 로드"""
    model = create_fusion_model(model_type=model_type, **kwargs)
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        new_state = {}
        for k, v in state.items():
            new_state[k[7:] if k.startswith("module.") else k] = v
        model.load_state_dict(new_state, strict=False)
        print(f"  ✅ Fusion Model ({model_type}) 로드 완료: {path}")
    else:
        print(f"  ⚠️ Fusion Model 파일 없음: {path}")
    return model.to(device).eval()


# =============================================================================
# 세그먼트 생성 (추론용)
# =============================================================================
def generate_segments(video_path, segment_duration=1.0, overlap=0.5):
    """비디오를 고정 길이 세그먼트로 분할"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    cap.release()

    if video_duration <= 0:
        return []

    segments = []
    step = segment_duration * (1 - overlap)
    current = 0

    while current + segment_duration <= video_duration:
        segments.append({
            'video_path': video_path,
            'start_sec': current,
            'end_sec': current + segment_duration,
            'label': 0,
        })
        current += step

    if current < video_duration:
        segments.append({
            'video_path': video_path,
            'start_sec': max(0, video_duration - segment_duration),
            'end_sec': video_duration,
            'label': 0,
        })

    return segments


# =============================================================================
# 결과 후처리
# =============================================================================
def merge_overlapping_segments(segments, threshold=0.3):
    """겹치는 세그먼트 병합. segments: list of (start, end, score)"""
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda x: x[0])
    merged = [list(sorted_segs[0])]

    for start, end, score in sorted_segs[1:]:
        prev_start, prev_end, prev_score = merged[-1]
        if start <= prev_end + threshold:
            merged[-1] = [prev_start, max(prev_end, end), max(prev_score, score)]
        else:
            merged.append([start, end, score])

    return [(s, e, sc) for s, e, sc in merged]


def format_time(seconds):
    """초를 MM:SS 형식으로 변환"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp(seconds):
    """초를 HH:MM:SS.mmm 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# =============================================================================
# JSON 결과 저장/로드
# =============================================================================
def save_results_json(results, output_path):
    """분석 결과를 JSON으로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Results saved to: {output_path}")


def load_results_json(input_path):
    """JSON 결과 로드"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_text_analysis_data(tensors_path, scores_path):
    """텍스트 텐서(.pt)와 분석 결과(.json)를 로드"""
    text_tensors = []
    text_scores = {}

    if os.path.exists(tensors_path):
        try:
            text_tensors = torch.load(tensors_path, map_location='cpu')
            print(f"  ✅ 텍스트 텐서 로드 완료: {tensors_path}")
        except Exception as e:
            print(f"  ❌ 텍스트 텐서 로드 실패: {e}")

    if os.path.exists(scores_path):
        try:
            with open(scores_path, 'r', encoding='utf-8') as f:
                text_scores = json.load(f)
            print(f"  ✅ 텍스트 분석 결과 로드 완료: {scores_path}")
        except Exception as e:
            print(f"  ❌ 텍스트 분석 JSON 로드 실패: {e}")

    return text_tensors, text_scores


# =============================================================================
# 클래스 이름 상수
# =============================================================================
GESTURE_CLASS_NAMES = ['No_Gesture', 'Gesture']
FUSION_CLASS_NAMES = ['No_Emphasis', 'Emphasis']
