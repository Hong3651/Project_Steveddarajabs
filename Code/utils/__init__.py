"""
utils package - 멀티모달 발표 분석 유틸리티
"""

from .datasets import (
    AudioFeatureExtractor,
    VideoEmphasisDataset,
    AudioEmphasisDataset,
    TextEmphasisDataset,
    MelSpectrogramDataset,
    FusionDataset,
)

from .models import (
    # Gesture / Video
    GestureModel,
    create_gesture_model,
    load_gesture_model,

    # Audio
    AudioLSTMModel,
    BiLSTM,
    create_audio_model,
    load_audio_model,
    AUDIO_CLASS_NAMES,

    # Text
    TextBERTModel,
    create_text_model,
    load_text_model,

    # Fusion
    FusionMLP,
    GatedFusion,
    TransformerFusion,
    MultimodalFusionModel,
    create_fusion_model,
    load_fusion_model,
    FUSION_CLASS_NAMES,
    GESTURE_CLASS_NAMES,

    # Utilities
    load_state_dict_safe,
    generate_segments,
    merge_overlapping_segments,
    format_time,
    format_timestamp,
    save_results_json,
    load_results_json,
    load_text_analysis_data,
)

from .helpers import (
    # Data preparation
    MelSpectrogramExtractor,
    load_label_entries,
    get_video_duration,
    generate_negative_entries,
    prepare_all_entries,
    train_val_split,
    train_val_split_by_video,

    # Audio processing
    auto_label_frames,
    extract_vocal_from_video,
    analyze_audio_emphasis,
    postprocess_predictions,
    load_all_data_for_audio,
    load_text_analysis_data,

    # Training
    train_one_epoch,
    evaluate,

    # Inference
    infer_video_emphasis,
    infer_audio_emphasis,
    infer_feature_fusion,
    merge_clips_to_segments,
    visualize_emphasis_on_video,
)

__version__ = "2.0.0"
