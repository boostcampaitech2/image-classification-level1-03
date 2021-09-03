# [P-Stage] Image Classification

마스크 착용 상태 Image Classification

Naver Boostcamp AI Tech 팀 프로젝트 입니다.

<br>

# Table of Content

- [Competition Overview](#competition-overview)
- [Data Definition](#data-definition)
- [Evaluation Method](#evaluation-method)
- [Usage](#usage)
- [Archive Contents](#Archive-Contents)
- [Source Code](#source-code)

<br>

# Competition Overview
- COVID-19의 확산으로 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다.
- 이에, 우리나라는 COVID-19 확산 방지를 위해 제도적으로 많은 노력을 하고 있습니다.
- COVID-19의 치사율은 다른 전염병과 달리 비교적 낮은 편에 속하나, 강력한 전염력을 가지고 있습니다.
- 감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 쉽게 전파가 될 수 있기 때문에 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 
- 이를 위해 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.
- 따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 
- 이 시스템을 통해 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

<br>

# Data Definition
- 사람 이미지에 대하여, 마스크를 착용하였는지, 이상하게 착용하였는지, 미착용하였는지에 확인
- 전체 사람 데이터셋의 60%는 학습 데이터 셋으로 활용
- Input
    - 이미지 크기 : h , w = ( 512 , 384 )
    - 사람의 얼굴 이미지
- Output
    - 18개의 클래스 ( 마스크 착용여부, 성별, 나이 총 3개의 조합)
        | class | mask      | gender | age          |
        |-------|-----------|--------|--------------|
        | 0     | Wear      | Male   | <30          |
        | 1     | Wear      | Male   | >=30 and <60 |
        | 2     | Wear      | Male   | >=60         |
        | 3     | Wear      | Female | <30          |
        | 4     | Wear      | Female | >=30 and <60 |
        | 5     | Wear      | Female | >=60         |
        | 6     | Incorrect | Male   | <30          |
        | 7     | Incorrect | Male   | >=30 and <60 |
        | 8     | Incorrect | Male   | >=60         |
        | 9     | Incorrect | Female | <30          |
        | 10    | Incorrect | Female | >=30 and <60 |
        | 11    | Incorrect | Female | >=60         |
        | 12    | Not wear  | Male   | <30          |
        | 13    | Not wear  | Male   | >=30 and <60 |
        | 14    | Not wear  | Male   | >=60         |
        | 15    | Not wear  | Female | >=60         |
        | 16    | Not wear  | Female | >=60         |
        | 17    | Not wear  | Female | >=60         |

<br>

# Evaluation Method
- 결과에 대한 평가는 F1 Score 를 통해 진행합니다.
- F1 Score
    - ## $F_{1}=2 * \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}$
    - ### $precision = \frac{\text{TP}}{\text{TP} + \text{FP}}$
    - ### $recall = \frac{\text{TP}}{\text{TP} + \text{FN}}$



<br>


# Usage

>**Install Requirements**
```bash
# pip install -r requirements.txt
```

>**train.py**
```bash
# SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py [parser]
```
- parser
    - --seed : 난수 추출시 seed 값 설정
    - --epoch : epoch 수 설정
    - --dataset : dataset.py내의 사용할 class 설정
    - --augmentation : dataset.py내의 사용할 aumentation class 설정
    - --resize : resize할 이미지 크기 설정
    - --batch_size : batch size 설정
    - --valid_batch_size : validation batch size 설정
    - --model : model.py에서 사용할 model class 설정
    - --optimizer : torch.optim중 사용할 optimizer 설정
    - --lr : learning rate 설정
    - --val_ratio : validation으로 나눌 비율 설정
    - --criterion : loss.py에서 사용할 criterion 설정
    - --lr_decay_step : learning rate decay를 진행할 단위 step 설정
    - --log_interval : loss와 accuracy를 출력, logger에 저장할 interval(간격) 설정
    - --name : model이 저장될 directory 이름 설정
    - --patience : validation loss가 증가하는 것을 용인할 횟수
    - --mode : 학습 모델의 종류 설정 (회귀, 분류)
    - --data_dir : 불러올 이미지 데이터셋의 경로 설정
    - --model_dir : model이 저장될 전체 경로 설정


>**inference.py**
```bash
# SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py [parser]
```
- parser
    - --batch_size : batch size 설정
    - --resize : resize할 이미지 크기 설정
    - --model : model.py에서 사용할 model class 설정
    - --mode : 불러올 모델의 종류(회귀, 분류) 설정
    - --data_dir : 불러올 이미지 데이터셋의 경로 설정
    - --model_dir : best 모델의 경로 설정
    - --output_dir : output csv를 저장할 경로 설정

<br>

# Archive Contents
```
image-classification-level1-03/
├──dataset.py
├──efficientnet.py
├──evaluation.py
├──inference.py
├──loss.py
├──model.py
├──README.md
└──train.py
```

<br>

# Source Code

- `dataset.py` : 이미지 전처리 수행
- `efficientnet.py` : EfficientNet 모델 파일
- `loss.py` : Loss 관련 모듈
- `model.py` : 모델 생성
- `train.py` : 학습에 필요한 파라미터 설정 및 validation
- `inference.py` : test 이미지에 대한 라벨링 및 csv 파일 생성

<!-- 
<br></br>
##  문제정의 및 해결방법 <a name = 'Solution'></a>
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 각자의 wrap up report에서 기술하고 있습니다. 
    - [wrapup report](https://docs.google.com/document/d/1DRyilPNVsjNzxif05JKpSIUnwOjDEZjkJKhnANbj094/edit)    

- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다.  -->

