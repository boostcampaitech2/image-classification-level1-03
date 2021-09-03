[P=Stage] image-classification-level1-03
==========================================
마스크 착용 상태 Image Classification



Table of Content
==================
* [대회 개요](#Overview)
* [데이터 설명](#DataDefinition)
* [Usage](#usage)
* [Archive](#archive)
* []


대회 개요 <a name = 'Overview'></a>
===============
- COVID-19의 확산으로 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다.
- 이에, 우리나라는 COVID-19 확산 방지를 위해 제도적으로 많은 노력을 하고 있습니다.
- COVID-19의 치사율은 다른 전염병과 달리 비교적 낮은 편에 속하나, 강력한 전염력을 가지고 있습니다.
- 감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 쉽게 전파가 될 수 있기 때문에 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 
- 이를 위해 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.
- 따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 
- 이 시스템을 통해 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

데이터 개요 <a name='DataDefinition'></a>
===============
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



Usage
=====

>**Install Requirements**

```bash
# pip install -r requirements.txt
```

>**train.py**
```bash
# SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py
```

>**inference.py**
```bash
# SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py
```

Archive
===============
```
├──dataset.py
├──efficientnet.py
├──evaluation.py
├──inference.py
├──loss.py
├──model.py
├──README.md
├──train.py
```



<br></br>
##  소스 코드 설명 <a name = 'Code'></a>
- `dataset.py` : dataset.py 이미지 전처리 수행
- `efficientnet.py` : EfficientNet 모델 파일
- `loss.py` : Loss 관련 모듈
- `model.py` : 모델 생성
- `train.py` : 학습에 필요한 파라미터 설정 및 validation
- `inference.py` : test 이미지에 대한 라벨링 및 csv 파일 생성





<br></br>
##  문제정의 및 해결방법 <a name = 'Solution'></a>
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 각자의 wrap up report에서 기술하고 있습니다. 
    - [wrapup report](https://docs.google.com/document/d/1DRyilPNVsjNzxif05JKpSIUnwOjDEZjkJKhnANbj094/edit)    

- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다. 

