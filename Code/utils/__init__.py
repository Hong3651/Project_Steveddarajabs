# utils/__init__.py

"""
utils package initialization
"""

from .datasets import (
    VideoEmphasisDataset, 
    AudioEmphasisDataset, 
    TextEmphasisDataset, 
    MelSpectrogramDataset, 
    FusionDataset, 
    AudioFeatureExtractor
)

from .models import (
    # Gesture / Video
    create_gesture_model, 
    load_gesture_model, 
    GestureModel,
    GESTURE_CLASS_NAMES,
    
    # Audio
    create_audio_model, 
    load_audio_model, 
    AudioLSTMModel, 
    BiLSTM, 
    AUDIO_CLASS_NAMES,
    
    # Text
    create_text_model, 
    load_text_model, 
    TextBERTModel,
    
    # Fusion
    create_fusion_model, 
    load_fusion_model, 
    FusionMLP, 
    FusionMLPNoText,
    MultimodalFusionModel, 
    FUSION_CLASS_NAMES,
    
    # Utils inside models
    generate_segments, 
    merge_overlapping_segments
)

from .helpers import (
    load_label_entries, 
    get_video_duration, 
    generate_negative_entries, 
    prepare_all_entries,
    train_val_split,
    train_val_split_by_video,
    AudioFeatureExtractor, # datasets에도 있지만 helpers에도 있다면 유지
    MelSpectrogramExtractor,
    analyze_audio_emphasis,
    postprocess_predictions,
    auto_label_frames,
    extract_vocal_from_video,
    train_one_epoch, 
    evaluate,
    load_all_data_for_audio,
    load_text_analysis_data,
    visualize_emphasis_on_video, 
    infer_feature_fusion, 
    infer_audio_emphasis,
    infer_video_emphasis, 
    merge_clips_to_segments,
    format_time,
    format_timestamp,
    save_results_json
)
__version__ = "2.0.0"  
__all__ = [
    # Datasets
    'VideoEmphasisDataset',
    'AudioEmphasisDataset',
    'TextEmphasisDataset',
    'MelSpectrogramDataset',
    
    # Models
    'GestureModel',
    'AudioLSTMModel',
    'BiLSTM',
    'TextBERTModel',
    'FusionMLP',
    
    # Model creators/loaders
    'create_gesture_model',
    'load_gesture_model',
    'create_audio_model',
    'load_audio_model',
    'create_text_model',
    'load_text_model',
    'create_fusion_model',
    'load_fusion_model',
    
    # Feature Extractors
    'MelSpectrogramExtractor',
    'AudioFeatureExtractor',
    
    # Data preparation
    'load_label_entries',
    'get_video_duration',
    'generate_negative_entries',
    'prepare_all_entries',
    'train_val_split',
    'train_val_split_by_video',
    
    # A.py functions
    'auto_label_frames',
    'extract_vocal_from_video',
    'analyze_audio_emphasis',
    'postprocess_predictions',
    
    # Training
    'train_one_epoch',
    'evaluate',
    'load_all_data_for_audio',
    
    # Inference
    'infer_video_emphasis',
    'infer_audio_emphasis',
    'infer_feature_fusion',
    'merge_clips_to_segments',
    'visualize_emphasis_on_video',
    'load_text_analysis_data',
    
    # Constants
    'AUDIO_CLASS_NAMES',
]