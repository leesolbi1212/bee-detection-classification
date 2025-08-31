# 🐝 꿀벌 탐지 및 분류 프로젝트 

논문 기반 딥러닝을 활용한 꿀벌 객체 탐지 및 종 분류 토이 프로젝트
🔗 **[프로젝트 상세 실험 과정 및 결과 분석](https://cat-b0.tistory.com/132)**

## 📌 프로젝트 개요

7일간 진행된 딥러닝 토이 프로젝트로, 팀원들이 직접 논문을 읽고 구현한 프로젝트입니다.
- **팀명**: Team 채애리 (5명)
- **기간**: 7일 
- **목표**: 꿀벌 이미지에서 객체를 탐지하고 8가지 종으로 분류

### 주요 특징
- YOLOv3 논문을 기반으로 직접 구현한 객체 탐지 모델
- ResNet-18을 활용한 꿀벌 종 분류 모델  
- End-to-End 파이프라인 구축
- 8가지 꿀벌 종류 분류 (수일벌/여왕벌 × 4가지 품종)

## 📊 분류 대상 꿀벌 종류

| 코드 | 종류 | 영문명 |
|------|------|--------|
| AB_LI | 수일벌-이탈리안 | Worker Bee - Italian |
| QB_LI | 여왕벌-이탈리안 | Queen Bee - Italian |
| AB_CA | 수일벌-카니올란 | Worker Bee - Carniolan |
| QB_CA | 여왕벌-카니올란 | Queen Bee - Carniolan |
| AB_BI | 수일벌-호박벌 | Worker Bee - Bumblebee |
| QB_BI | 여왕벌-호박벌 | Queen Bee - Bumblebee |
| AB_AP | 수일벌-한봉 | Worker Bee - Korean Native |
| QB_AP | 여왕벌-한봉 | Queen Bee - Korean Native |

## 🔬 구현 모델

### 1. YOLOv3 (객체 탐지)
- **논문**: "YOLOv3: An Incremental Improvement" - Joseph Redmon, Ali Farhadi (2018)
- **백본**: Darknet-53
- **특징**: 
  - 3개 스케일에서의 다중 탐지 (13×13, 26×26, 52×52)
  - K-means 기반 앵커 박스 최적화
  - 논문 손실 함수 가중치 적용

### 2. ResNet-18 (종 분류)
- **논문**: "Deep Residual Learning for Image Recognition" - Kaiming He et al. (2015)
- **구조**: 18층 잔차 네트워크
- **특징**: Skip connection을 통한 그라디언트 소실 문제 해결

## 🛠️ 프로젝트 구조

```
bee_detection_pipeline/
├── data/
│   └── dataset.py              # 데이터 로더 및 전처리
├── models/
│   ├── yolov3.py              # YOLOv3 모델 구현
│   └── resnet18.py            # ResNet-18 모델 구현
├── train/
│   ├── train_yolo.py          # YOLOv3 학습 스크립트
│   ├── train_resnet.py        # ResNet-18 학습 스크립트
│   ├── yolo_loss.py           # YOLO 손실 함수
│   └── yolo_loss_improved.py  # 개선된 YOLO 손실 함수
├── utils/
│   ├── bee_species_info.py    # 꿀벌 종류 정보
│   ├── metrics.py             # 성능 평가 메트릭
│   └── yolo_utils.py          # YOLO 유틸리티 함수
├── pipeline/
│   └── bee_pipeline.py        # E2E 파이프라인
└── notebooks/
    ├── 7_yolov3탐지(최종).ipynb
    └── YOLOv3_ResNet18_꿀벌탐지분류_완성버전.ipynb
```

## 🚀 사용 방법

### 환경 설정
```bash
# 저장소 클론
git clone https://github.com/[your-username]/bee-detection-project.git
cd bee-detection-project

# 패키지 설치
pip install -r requirements.txt
```

### 학습 실행
```bash
# YOLOv3 학습
python train/train_yolo.py --data_dir ./data --epochs 100 --batch_size 8

# ResNet-18 학습  
python train/train_resnet.py --data_dir ./data --epochs 100 --batch_size 32
```

### 추론 실행
```python
from pipeline.bee_pipeline import BeeDetectionClassificationPipeline

# 파이프라인 초기화
pipeline = BeeDetectionClassificationPipeline(
    yolo_checkpoint='checkpoints/best_yolo.pth',
    resnet_checkpoint='checkpoints/best_resnet.pth'
)

# 이미지 처리
results = pipeline.process_image('test_image.jpg')
```

## 📈 주요 성과

- YOLOv3 논문을 직접 읽고 구현
- ResNet-18을 활용한 8가지 종 분류 달성
- K-means 앵커 최적화로 탐지 성능 향상
- End-to-End 파이프라인 구축

## 👥 팀원 역할

- **YOLOv2 구현팀**: YOLOv2 논문 분석 및 구현 시도
- **YOLOv3 구현팀**: YOLOv3 논문 분석 및 구현 (본인 모델 최종 채택)
- **ResNet-18 구현팀**: 꿀벌 종 분류 모델 구현

## 📚 참고 자료

### 논문
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### 블로그
- 🔗 **[프로젝트 상세 실험 과정 및 결과 분석](https://cat-b0.tistory.com/127)**
  - 데이터셋 구축 과정
  - 모델 구현 세부사항
  - 실험 결과 및 성능 분석
  - 문제 해결 과정

## 📋 요구사항

- Python 3.8+
- PyTorch 1.8.0+
- CUDA 11.0+ (GPU 학습 시)
- 자세한 패키지 목록은 `requirements.txt` 참조

## ⚡ Google Colab 실행

프로젝트는 Google Colab A100 GPU 환경에 최적화되어 있습니다.

```python
# Colab 환경 설정
!python colab_setup.py

# 노트북 파일 실행
# BeeDetection_Complete_Colab.ipynb
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- 7일간의 짧은 기간 동안 열정적으로 참여한 Team 채애리 Bee 팀원들
- 논문 저자들 (Joseph Redmon, Ali Farhadi, Kaiming He et al.)
- 데이터셋 제공 기관

---

**💡 Note**: 이 프로젝트는 학습 목적의 토이 프로젝트입니다. 실제 프로덕션 환경에서 사용하기 위해서는 추가적인 최적화와 테스트가 필요합니다.
