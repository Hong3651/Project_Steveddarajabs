# Experiments

이 폴더는 모델 비교와 학습 과정을 정리한 실험 노트북을 모아둔 공간입니다.

## model_comparison

여러 모델 후보를 비교하기 위한 실험 노트북입니다.

| 파일 | 내용 |
| --- | --- |
| `gesture_and_fusion_model_test.ipynb` | Gesture 모델 후보와 Fusion 구조 비교 실험 |
| `fusion_mlp_v1.ipynb` | MLP Fusion 1차 실험 |
| `fusion_mlp_v2.ipynb` | MLP Fusion 2차 실험 |
| `fusion_gated_v2.ipynb` | Gated Fusion 실험 |
| `fusion_transformer_v1.ipynb` | Transformer Fusion 1차 실험 |
| `fusion_transformer_v2.ipynb` | Transformer Fusion 2차 실험 |
| `json_output_experiment.ipynb` | JSON 출력 형태 실험 |

## training_pipeline

모달리티별 학습과 통합 추론 흐름을 정리한 노트북입니다.

| 파일 | 내용 |
| --- | --- |
| `01_gesture_labeling_tool.ipynb` | 제스처 라벨링 도구 |
| `02_train_gesture.ipynb` | Gesture 모델 학습 |
| `03_train_audio_model.ipynb` | Audio 모델 학습 |
| `04_make_text_features.ipynb` | Text feature 생성 실험 |
| `05_train_gated_fusion.ipynb` | Gated Fusion 학습 |
| `06_main_inference_binary.ipynb` | 통합 추론 실험 |
