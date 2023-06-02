# level2_objectdetection-cv-13
# Project 연구일지


### 아쉬웠던 점

- k-fold 를 제대로 못해본점
- mislabeled 된 데이터를 제거하고 돌린 데이터셋이 없었던것
- sota 에 너무 많은 시간 투자한것
- validation dataset 으로 각 모델의 precision, recall 을 따로 뽑아서 좀 더 근거있는 ensemble 을 했어야하는것
- backbone 모델로 파라미터 조정해가며 많이 돌린 후 최종 파라미터로 큰 backbone 사용해보지 못한것

# 실험 리포트

## Introduction

총 10 종류(일반 쓰레기, 종이, 상자, 금속, 유리, 플라스틱, 스티로폼, 비닐봉지, 베터리, 옷)의 쓰레기가 찍힌 사진 속에서 각각의 쓰레기 위치를 최대한 정확히 localization하고 쓰레기의 종류를 classification 해야하는 문제

일반 object detection 문제와 달리 localizaition과 classification의 **기준이 굉장히 애매한 경우가 많은 object detection 문제** 

- 어느 물체까지를 쓰레기로 볼것인가? → bbox의 localization의 어려움 + classification의 어려움
- 과연 라벨링하는 여러 사람들이 모두 동일한 기준을 가지고 라벨링을 하였을까? → classification의 어려움
- 분류 기준이 다르게 적용된 데이터에 대해서는 오라벨 데이터로 판단해야 할까? → 오라벨로 판단한다면 오라벨 데이터가 다수 존재

### EDA

- **라벨링의 기준이 굉장히 애매하다.  
  →** 어디까지를 일반 사물로 보고 어디까지를 쓰레기로 볼것인가? 패트병에 붙어있는 비닐은? 비닐 봉지에 담겨있는 쓰레기는? 옷과 일반쓰레기의 구분 기준은?
- **오라벨된 데이터가 다수 존재한다.** 
  → 라벨링을 일일이 수정할 수 없기 때문에 오히려 데이터셋의 이러한 특징을 이용할 수 있는 방법을 고민 
  → 애초에 라벨링의 기준이 애매하고 오라벨 된 데이터가 많다면, 정확한 라벨링은 할 수 없더라도 pseudo labeling을 통해 절대적인 학습 데이터의 양을 늘려주는 것이 더 효과적일 것이라고 판단

## Related work & Method

### cv 전략

**Stratified Group KFold** 

object detection에서는 하나의 이미지안에 여러개의 class가 존재하므로 단순히 stratified kfold로는 클래스 별 분포를 맞춰 데이터를 나눠줄 수 없다. 그래서 일반적으로 object detection에서는 Stratified Group KFold를 사용한다. Stratified Group KFold를 사용하면 이미지 별로 그룹을 형성하고 그룹을 유지한 상태로 stratified  kfold를 적용할 수 있어, 결과적으로 이미지 중복없이 클래스의 분포를 균일하게 데이터셋을 나눌 수 있다. 

- `StratifiedKFold` to preserve the percentage of samples for **each class**. → 클래스별 분포 균등(General trash, paper …)
- `GroupKFold` to ensure that the same group will not appear in two different folds. → 이미지 중복을 막음, 이미지 별로 그룹을 형성
- `StratifiedGroupKFold` to keep the constraint of `GroupKFold` while attempting to return stratified folds. → GroupKFold를 유지하면서 클래스별 분포를 균등하게 맞춤

### pseudo labeling

Pseudo-labeling은 가장 확률 높은 라벨을 가상 라벨의 형태로 부여하는 기법으로서, 라벨값이 부족한 데이터의 한계를 극복하기 위한 방법이다.

Pseudo-labeling은 지도학습을 통해 1차적으로 학습된 모델(teacher model)을 이용하여, 태깅이 되지 않은 데이터에 대해 예측을 수행한다. 수행된 예측 결과를 이용해 가짜(pseudo)로 태깅(labeling)을 진행한다. 따라서 Pseudo-labeling을 수행하기 위해서는 학습된 모델이 있어야하고, 태깅되지 않은 데이터가 있어야 한다.

태깅되지 않은 데이터에 대해서 Pseudo-labeling을 한 후에는, 이 확장된 큰 데이터셋을 이용하여 2차 학습을 수행한다.

### Model

**Faster R-CNN** 

- 처음에는 mmdetection에 익숙해지기 위해 가장 접근하기 쉬운 모델로 실험 + 다양한 backbone을 쉽게 적용해 보기 위함
- 실험의 base 모델로 사용
- 2 stage의 방법론을 사용하는 만큼 기본 모델 중에서는 비교적 높은 mAP를 보여줌

**RetinaNet** 

- 모델의 다양성을 위해 1stage 모델도 사용
- 작은 물체 탐지의 정확도를 높이기 위해 FPN을 사용하는 모델 선정
- Focal Loss를 사용하여 class imbalance 문제를 완화

**Yolo** 

- ensemble을 고려하여 모델의 다양성을 위해 Yolo 계열의 모델을 선정하였고, Yolov3 에 다양한 탐지 테크닉(Anchor-free, Decoupled Head, SimOTA)을 추가하여 높은 성능을 보여주는 yolox를 선정
- 오라벨된 데이터에 대해 과적합 되는 것을 막기위해 **다양한 augmentation**을 적용해 보고 싶었고, 
  이러한 관점에서 기본적으로 다양한 augmentation(Mosaic, RandomAffine, MixUp …)이 적용되어 있는 yolox가 적절할 것으로 판단

**CascadeRCNN**

- pseudo labeling은 swin을 백본을 하는 cascade rcnn의 결과로 진행
- 최대한 큰 모델을 사용하기 위해 backbone은 transformer 기반의 swin_L 사용
  - 보통 teacher model을 student model 보다 큰 모델로 이용한다는 점을 고려
- cascade rcnn은 IoU threshold를 다양하게 가져 가며 학습하는 방식으로 전반적인 모델의 성능을 향상시키면서도 high quality detection을 수행할 수 있는 방법이다. 따라서 cascade rcnn을 teacher model로 사용하였을 때, 더 정확한 bbox를 더 많이 이용할 수 있을 것으로 기대했다.

## ****Experiments****

각 모델(Faster R-CNN, Retinanet, Yolox)의 성능을 최대한 향상시킨 뒤 모든 조건은 동일한 상태에서 pseudo labeling 된 데이터셋을 이용해 결과 비교 

### Pretrained weight & Resize

(val 기준)

| 모델         | BackBone   | 내용                         | mAP    | mAP50  |
| ------------ | ---------- | ---------------------------- | ------ | ------ |
| Faster R-CNN | Resnet50   | -                            | 0.1530 | 0.2730 |
| Faster R-CNN | Resnet50   | pretrianed                   | 0.2970 | 0.4560 |
| Faster R-CNN | Resnet50   | pretrained, resize=1024x1024 | 0.350  | 0.4890 |
| Faster R-CNN | Resnet101  | pretrained, resize=1024x1024 | 0.3670 | 0.4950 |
| Faster R-CNN | Resnext101 | pretrained, resize=1024x1024 | 0.3480 | 0.4800 |

- 확실히 현재 데이터셋으로만 학습을 하는 것 보다 coco 데이터셋으로 이미 사전 학습된 weight를 이용하는 것이 훨씬 성능이 좋았고 이미지 resolution을 512x512로 했을 때 보다 1024x1024로 했을 때 더 좋은 성능을 보여주었다.
- 백본도 크기가 커짐에 따라 성능이 향상되었지만 Resnext101의 경우 Resnet50 보다 오히려 성능이 하락하는 모습을 보여주었다.

### optimizer & lr

| 모델         | BackBone   | 내용            | mAP    | mAP50  |
| ------------ | ---------- | --------------- | ------ | ------ |
| Faster R-CNN | Resnet101  | SGD lr=2e-2     | 0.3670 | 0.4950 |
| Faster R-CNN | Resnet101  | Adam, lr=1e-4   | 0.3530 | 0.4970 |
| Faster R-CNN | Resnet101  | Adam, lr=5e-5   | 0.3810 | 0.5183 |
| Faster R-CNN | Resnet101  | Adam, lr=2.5e-5 | 0.3910 | 0.5250 |
| Retinanet    | Resnet101  | Adam, lr=2.5e-5 | 0.4030 | 0.5420 |
| Retinanet    | Resnext101 | Adam, lr=2.5e-5 | 0.4060 | 0.5310 |

- 기존 optimizer를 SGD lr=0.02 에서 Adam으로 바꿔주고 lr를 조정하며 실험
- Adam을 사용할 때는 lr을 생각보다 많이 작게 해주어야 학습 진행되었다.

### Pseudo labeling

- teacher model의 inference 결과를 시각화해서 관찰한 결과 confidience 0.65 기준으로 labeling을 해주었을 때 가장 불필요한 box를 제거하면서도 정확한 box를 살릴 수 있음을 확인하고 confidience가 0.65 이상인 box만 labeling에 사용

- train set과 val set의 경우 pseudo labeling dataset을 만든 후 stratified group kfold 이용

- Yolox의 경우 시간이 부족해 기본 설정으로만 실험 진행

| 모델         | BackBone  | 내용                            | mAP    | mAP50  |
| ------------ | --------- | ------------------------------- | ------ | ------ |
| Faster R-CNN | Resnet101 | 기존 dataset만 사용             | 0.3910 | 0.5250 |
| Faster R-CNN | Resnet101 | pseudo labeling 된 dataset 사용 | 0.4720 | 0.6300 |
| Retinanet    | Resnet101 | 기존 dataset만 사용             | 0.4030 | 0.5420 |
| Retinanet    | Resnet101 | pseudo labeling 된 dataset 사용 | 0.4670 | 0.6270 |
| Yolox        | darknet53 | 기존 dataset만 사용             | 0.3220 | 0.4260 |
| Yolox        | darknet53 | pseudo labeling 된 dataset 사용 | 0.4560 | 0.5850 |

- 실험 결과 pseudo labeling을 통해 데이터셋의 양을 2배 가까이 증가시켜 학습시킨 경우 엄청난 성능 향상을 보여줌

- 하지만 다른 모델에 비해 validation score와 LB score의 차이(0.6300 → 0.5830)가 크다는 것을 확인. (보통 0.01~0.02 정도의 차이가 있었음)
  → pseudo labeling 된 데이터를 학습하는 과정에서 오라벨링 된 데이터에 대해서도 학습이 진행된 것으로 보임 
  → 그럼에도 꽤 높은 성능향상을 보여주었고 이는 좋은 teacher model을 만들 수만 있다면 성능이 낮았던 모델들의 전반적인 성능 향상이 가능하다는 것을 확인함

    - **student model이 오라벨된 데이터에 대해 덜 학습될 수 있는 방법을 고민** (5가지 개선 아이디어)

      1. pseudo labeling 데이터는 수렴하는 순간까지만 사용해서 학습하기 (과적합 방지를 위해)
      2. 기존 정답 데이터로 한번 더 학습 진행 (fine-tuning) 
         → pseudo labeling 데이터 셋을 만들때 이 부분을 고려하지 못하고 만들어 validation set을 이용할 수 없게됨 (아쉬운 점)
      3. pseudo labeling의 정답 판단 기준을 높여보기 
      4. 대부분 모델의 inference 결과를 보았을 때 **작은 박스에 대한 정확도가 매우 낮은 것을 확인**하고, EDA를 통해 하위 25%크기의 작은 box에 대한 결과는 pseudo labeling에서 제외 하기
      5. ansemble된 결과를 가지고 pseudo labeling 시도해 보기

## ****Conclusion****

- 실험 결과 pseudo labeling된 데이터셋을 이용해 학습했을 경우 많은 성능 향상이 있음을 확인
- 하지만 teacher model보다 더 큰 모델에 대해서는 성능 하락하는 모습을 보임
- 결과적으로 단순히 pseudo labeling 만을 이용해서는 teacher model 보다 더 높은 성능을 내기는 힘들 것으로 판단
- 모델의 다양성을 필요로 하는 ensemble에서는 teacher model의 결과를 학습하는 pseudo labeling 방법이 효과적이지 못함을 확인

### 한계 및 아쉬운 점

- 처음 pseudo labeling dataset을 만들때 원본 데이터셋으로 fine-tuning 해볼 것을 고려하여 validation set은 psuedo labeling dataset에 포함되지 않도록 만들었어야 했는데, 그 부분을 미리 고려하지 못한점이 아쉽다.
- 또한 작은 박스에 대해 filtering을 거친 dataset에 대해서도 실험해보지 못한 점이 아쉽다. 전반적으로 시간이 부족해서 더 개선시킬 아이디어가 있었음에도 시도해 보지 못한점이 아쉽다.
- 또한 대회 특성상 ensemble을 하는 것이 최종 결과를 향상시키는데 많은 영향을 미쳤는데, pseudo labeling을 사용할 경우 teacher 모델의 결과를 학습하여 모델의 다양성이 감소하였고 모델의 다양성을 필요로 하는 ensemble에는 pseudo labeling이 불리할 것이라는 생각을 미리하지 못했던 점이 아쉽다.

## YOLOv8(Ultralytics) 실험(김대희)

| experiment #          | 1           | 2           | 3           | 4           | 5           | 6           | 7           | 8           | 9           | 10          | 11          |
| --------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| pretrained weight     | yolov8n     | yolov8n     | yolov8n     | yolov8n     | yolov8n     | yolov8n     | yolov8n     | yolov8n     | yolov8x     | yolov8x     | yolov8x     |
| custom folded dataset | not applied | not applied | not applied | not applied | not applied | not applied | not applied | not applied | not applied | not applied | not applied |
| epochs                | 100         | 100         | 100         | 100         | 200         | 200         | 200         | 200         | 200         | 200         | 200         |
| batch size            | 32          | 32          | 32          | 32          | 32          | 32          | -1          | -1          | -1          | -1          | -1          |
| imgsz                 | 640         | 1024        | 1024        | 1024        | 1024        | 1024        | 1024        | 1024        | 1024        | 1024        | 1024        |
| optimizer             | SGD         | SGD         | AdamW       | AdamW       | AdamW       | AdamW       | AdamW       | AdamW       | AdamW       | SGD         | SGD         |
| cosine learning rate  | false       | false       | false       | false       | false       | false       | false       | true        | false       | false       | true        |
| auto mixed precision  | true        | true        | true        | true        | true        | true        | true        | true        | true        | true        | true        |
| confidence threshold  | default     | default     | default     | default     | default     | default     | default     | default     | default     | default     | default     |
| IoU                   | 0.7         | 0.7         | 0.7         | 0.7         | 0.7         | 0.7         | 0.7         | 0.7         | 0.7         | 0.7         | 0.7         |
| max detection         | 300         | 300         | 300         | 300         | 100         | 100         | 100         | 100         | 100         | 300         | 300         |
| lr0                   | 0.01        | 0.01        | 0.001       | 0.001       | 0.001       | 0.001       | 0.001       | 0.001       | 0.001       | 0.01        | 0.01        |
| lrf                   | 0.01        | 0.01        | 0.01        | 0.01        | 0.01        | 0.01        | 0.01        | 0.01        | 0.01        | 0.01        | 0.01        |
| rotation degrees      | 0           | 0           | 0           | 30          | 180         | 180         | 180         | 180         | 180         | 0           | 0           |
| mixup                 | 0           | 0           | 0           | 0.3         | 0.3         | 0.5         | 0.5         | 0.5         | 0.5         | 0           | 0           |
| mAP50                 | 0.42897     | 0.43955     | 0.43348     | 0.43224     | 0.42544     | 0.41117     | 0.41587     |             | 0.52594     | 0.53287     | 0.53178     |
| mAP50-95              | 0.33532     | 0.33693     | 0.32972     | 0.30795     | 0.30606     | 0.29096     | 0.29219     |             | 0.40607     | 0.45442     | 0.44852     |
| leaderboard mAP       | 0.3716      |             |             |             |             |             |             |             |             | 0.4536      | 0.4590      |

YOLOv8 실험 후기

- 당연하게도, 큰 model은 학습 느리지만 score 높음
  - 학습 소요시간은 yolov8x이 yolov8n의 8 배 이상
  - 가장 작은 yolov8n 대신 yolov8x 사용 시 비약적 향상
- augmentation이 mAP score 향상에 도움이 되지 않았음
  - 기본적으로 YOLOv8 model은 mosaic 기법이 적용 되어 있음
  - 아래 각각이 mAP score 향상에 기여 하지 못 함
    - random rotation의 경우, (분석 추가)
    - mixup의 경우, (분석 추가)
- AdamW optimizer가 mAP score 향상에 도움이 되지 않았음
  - image classification에서는 F1 score 향상에 도움이 되었음
  - object detection에서는 SGD optimizer가 더 좋음
    - task에 의존적인지는 확실치 않음
- cosine learning rate scheduler가 mAP score 향상에 도움이 되지 않았음
  - 타 model에서는 score 향상에 도움이 되는 경우가 있음
  - 유의미한지 모르겠으나 수렴 시간은 다소 단축
    - 기존 200 epoch → 변경 160 epoch

HTC RCNN 실험 후기

- project code 전체를 그대로 서로 다른 hardware 등 환경에서 구동 하면 model이 완전히 다르게 학습 될 수 있음을 확인

  - 준하님 server와 내 server 각각에서 학습 한 같은 model의 validation score( 특히 mAP_m과 mAP_s) 최대 2 ~ 3% 차이

![yolov8](https://github.com/boostcampaitech5/level2_objectdetection-cv-13/assets/95335531/a249ffe0-b42b-4e7d-b288-4820a28b7128)

ensemble 실험 후기

- ensemble 방식 중 bbox 개수가 줄어드는 응용 nms 방식은 기본 nms 방식 보다 mAP 점수 낮음
- model 4 개 보다 model 6 개 ensemble mAP score가 더 좋지는 않음 → 마냥 ensemble source 개수에 비례 하지는 않음
- ensemble 결과들의 ensemble은 예상대로 mAP score가 더 좋아지지는 않음

## Ensemble 실험 (김령태)

| mAP    | 모델 description                                             | ID   |
| ------ | ------------------------------------------------------------ | ---- |
| 0.5830 | pseudo labling teather model : cascade rcnn student model : retinanet r101 | 1    |
| 0.6109 | cascade_rcnn_lr_0.00005_epoch100                             | 2    |
| 0.6187 | exp 16 HTC_swinL_AdamW_lr0.00005 cosine                      | 3    |
| 0.4613 | retinaswin - epoch28                                         | 4    |
| 0.4614 | baseline Faster RCNN                                         | 5    |
| 0.4590 | YOLOv8x_cosineLR                                             | 6    |
| 0.4536 | YOLOv8x_defaultLR                                            | 7    |
| 0.5999 | exp20_HTC_swinL                                              | 8    |

| 기법     | 모델                                                  | IoU_Thr | Sigma | Thresh (or Skip_Box_Thr) | Method (= 2, Gaussian) | mAP Score |
| -------- | ----------------------------------------------------- | ------- | ----- | ------------------------ | ---------------------- | --------- |
| Soft NMS | 0.5830(1번) & 0.6109(2번)                             | 0.4     | 0.5   | 0.001                    |                        | 0.5642    |
|          | 0.5830(1번) & 0.6109(2번) & 0.6187(3번) & 0.4613(4번) | 0.7     | 0.7   | 0.001                    |                        | 0.4567    |
|          |                                                       | 0.7     | 0.2   | 0.001                    |                        | 0.5969    |
|          |                                                       | 0.7     | 0.3   | 0.005                    |                        | 0.5603    |
|          |                                                       |         |       |                          |                        |           |
| WBF      | 0.5830(1번) & 0.6109(2번) & 0.6187(3번) & 0.4613(4번) | 0.7     |       | 0.3                      |                        | 0.6245    |
|          |                                                       | 0.7     |       | 0.2                      |                        | 0.6370    |
|          |                                                       |         |       |                          |                        |           |
| NMS      | 0.5830(1번) & 0.6109(2번)                             | 0.5     |       |                          |                        | 0.6122    |
|          | 0.5830(1번) & 0.6109(2번) & 0.6187(3번) & 0.4613(4번) | 0.7     |       |                          |                        | 0.6373    |
|          |                                                       | 0.8     |       |                          |                        | 0.6281    |
|          | 0.5830(1번) & 0.6109(2번) & 0.6187(3번)               | 0.7     |       |                          |                        | 0.6427    |
|          | 0.6109(2번) & 0.6187(3번)                             | 0.7     |       |                          |                        | 0.6424    |
|          | 0.6187(3번) & 0.4590(6번)                             | 0.7     |       |                          |                        | 0.6250    |
|          | 0.6109(2번) & 0.6187(3번) & 0.4590(6번)               | 0.65    |       |                          |                        | 0.6473    |
|          | 0.6187 & 0.6109 & 0.4590                              | 0.62    |       |                          |                        | 0.6476    |
|          | 0.6187 & 0.6109 & 0.4590 & 0.5999                     | 0.62    |       |                          |                        | 0.6499    |
|          |                                                       | 0.5     |       |                          |                        | 0.6504    |
|          | 6109 & 6187 & 5830 & 4590 & 4536 & 5779               | 0.62    |       |                          |                        | 0.6433    |
|          | 6187 & 6109  & 4590 & 5999 & 5830 & 5779              | 0.62    |       |                          |                        | 0.6472    |

### mAP 순 정렬

| 기법     | 모델                                                  | IoU_Thr | Sigma | Thresh (or Skip_Box_Thr) | Method (= 2, Gaussian) | mAP Score |
| -------- | ----------------------------------------------------- | ------- | ----- | ------------------------ | ---------------------- | --------- |
| NMS      | 0.6109(2번) & 0.6187(3번) & 0.4590(6번)               | 0.62    |       |                          |                        | 0.6476    |
| NMS      | 0.6109(2번) & 0.6187(3번) & 0.4590(6번)               | 0.65    |       |                          |                        | 0.6473    |
| NMS      | 0.5830(1번) & 0.6109(2번) & 0.6187(3번)               | 0.7     |       |                          |                        | 0.6427    |
| NMS      | 0.6109(2번) & 0.6187(3번)                             | 0.7     |       |                          |                        | 0.6424    |
| NMS      | 0.5830(1번) & 0.6109(2번) & 0.6187(3번) & 0.4613(4번) | 0.7     |       |                          |                        | 0.6373    |
| WBF      | 0.5830(1번) & 0.6109(2번) & 0.6187(3번) & 0.4613(4번) | 0.7     |       | 0.2                      |                        | 0.6370    |
| NMS      | 0.5830(1번) & 0.6109(2번) & 0.6187(3번) & 0.4613(4번) | 0.8     |       |                          |                        | 0.6281    |
| NMS      | 0.6187(3번) & 0.4590(6번)                             | 0.7     |       |                          |                        | 0.6250    |
| WBF      | 0.5830(1번) & 0.6109(2번) & 0.6187(3번) & 0.4613(4번) | 0.7     |       | 0.3                      |                        | 0.6245    |
| NMS      | 0.5830(1번) & 0.6109(2번)                             | 0.5     |       |                          |                        | 0.6122    |
| Soft NMS | 0.5830(1번) & 0.6109(2번) & 0.6187(3번) & 0.4613(4번) | 0.7     | 0.2   | 0.001                    |                        | 0.5969    |
| Soft NMS | 0.5830(1번) & 0.6109(2번)                             | 0.4     | 0.5   | 0.001                    |                        | 0.5642    |
| Soft NMS | 0.5830(1번) & 0.6109(2번) & 0.6187(3번) & 0.4613(4번) | 0.7     | 0.3   | 0.005                    |                        | 0.5603    |
| Soft NMS | 0.5830(1번) & 0.6109(2번) & 0.6187(3번) & 0.4613(4번) | 0.7     | 0.7   | 0.001                    |                        | 0.4567    |

### 주목할 점

**0.6476 vs 0.6473 : 같은 NMS, IoU_Thr를 0.62와 0.65로 대조 ⇒ 0.62가 미세하게 높았음**

**0.6373 vs 0.6281 : 같은 NMS, IoU_Thr를 0.8과 0.7로 대조 ⇒ 0.7이 유의미하게 높았음**

**0.6373 vs 0.6370 : IoU_Thr 0.7의 NMS와 WBF(Thresh = 0.2) 대조 ⇒ 거의 비슷한 결과**

**0.6370 vs 0.6245 : 같은 WBF, IoU_Thr를 Thresh만 0.2와 0.3으로 대조 ⇒ WBF의 Thresh가 낮은 쪽이 유의미하게 높음**

**0.6427 vs 0.6424 & 0.6427 vs 0.6122 (& 0.6473 vs 0.6427) (?) : 일반적으로 높은 mAP를 포함할 시 매우 상승하나, 낮은 mAP라 해도 기법에 따라 낮은 mAP의 포함도 유의미할 수 있음**

**결론**

**NMS vs WBF vs Soft NMS : 어느정도 Parameter를 변경해봐도 전반적으로 Soft NMS가 유의미하게 낮음, WBF와 NMS는 파라미터 조정에 따라 비슷한 수준을 보임**

**IoU_Threshold : 0.8 → 0.7, 0.65 → 0.62으로 낮추는 케이스 모두 상승하는 경향을 보임**
# Contributors ✨


<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/eogml88"><img src="https://avatars.githubusercontent.com/u/6427695?v=4?s=100" width="100px;" alt=""/><br /><sub><b>김대희</b></sub></a><br />
    <td align="center"><a href="https://github.com/kjs2109"><img src="https://avatars.githubusercontent.com/u/95335531?v=4" width="100px;" alt=""/><br /><sub><b>김주성</b></sub></a><br />
    <td align="center"><a href="https://github.com/RyeongTaeKim"><img src="https://avatars.githubusercontent.com/u/48413850?v=4" width="100px;" alt=""/><br /><sub><b>김령태</b></sub></a><br />
    <td align="center"><a href="https://github.com/jh58power"><img src="https://avatars.githubusercontent.com/u/48081459?v=4?s=100" width="100px;" alt=""/><br /><sub><b>황준하</b></sub></a><br />
    <td align="center"><a href="https://github.com/JunOnJuly"><img src="https://avatars.githubusercontent.com/u/97649331?v=4" width="100px;" alt=""/><br /><sub><b>성지훈</b></sub></a><br />
  </tr>
</table>
