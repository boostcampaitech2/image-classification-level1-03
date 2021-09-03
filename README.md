# image-classification-level1-03
image-classification-level1-03 created by GitHub Classroom    

* [Usage](#usage)

## Usage

>**training**
```bash
# python train.py
```

>**inference**
```bash
# python inference.py --model_dir {model_path}
```
add hyeons branch
# Pstage 1 ] Image Classification
###### 😷 마스크 착용 상태 분류 Image Classification
###### 📌 본 프로젝트는 [_*Naver AI Boostcamp*_](https://www.edwith.org/bcaitech1/)에서 Team Project로 진행됐습니다.

<br></br>
## 📋 Table of content
+ [🏅최종결과](#Result)
+ [👁대회개요](#Overview)
+ [💾데이터개요](#Data)
+ [📝문제정의 및 해결방법](#'Solution')
+ [🗄폴더구조](#Directory)
+ [💻소스코드설명](#Code)


<br></br>
## 🏅 최종 결과 <a name = 'Result'></a>
- [[Image Classification]](http://boostcamp.stages.ai/competitions/1/overview/description)
    - Private LB (4/250)
        - F1 : `0.7691` 


<br></br>
## 👁 대회 개요 <a name = 'Overview'></a>
COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

- 평가방법 
    - F1 score

<br></br>
## 💾 데이터 개요 <a name = 'Data'></a>
마스크를 착용하는 건 COIVD-19의 확산을 방지하는데 중요한 역할을 합니다. 제공되는 이 데이터셋은 사람이 마스크를 착용하였는지 판별하는 모델을 학습할 수 있게 해줍니다. 모든 데이터셋은 아시아인 남녀로 구성되어 있고 나이는 20대부터 70대까지 다양하게 분포하고 있습니다. 간략한 통계는 다음과 같습니다.

- 전체 사람 명 수 : 4,500

- 한 사람당 사진의 개수: 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]

- 이미지 크기: (384, 512)

학습 데이터와 평가 데이터를 구분하기 위해 임의로 섞어서 분할하였습니다. 60%의 사람들은 학습 데이터셋으로 활용되고, 20%는 public 테스트셋, 그리고 20%는 private 테스트셋으로 사용됩니다

진행중인 대회의 리더보드 점수는 public 테스트셋으로 계산이 됩니다. 그리고 마지막 순위는 private 테스트셋을 통해 산출한 점수로 확정됩니다. private 테스트셋의 점수는 대회가 진행되는 동안 볼 수 없습니다.

입력값. 마스크 착용 사진, 미착용 사진, 혹은 이상하게 착용한 사진(코스크, 턱스크)


<br></br>
## 📝 문제정의 및 해결방법 <a name = 'Solution'></a>
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 각자의 wrap up report에서 기술하고 있습니다. 
    - [wrapup report](https://maihon.oopy.io/study/boostcamp/p-stage/classification/wrapup-report)    

- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다. 


<br></br>
## 🗄 폴더 구조 <a name = 'Directory'></a>
```
└── Image_Classification
     ├── experiments
     ├── infer.py
     ├── infer.sh
     ├── logs
     ├── pseudo_train.py
     ├── pseudo_train.sh
     ├── src
     │   ├── configs
     │   ├── dataset.py
     │   ├── earlyStop.py
     │   ├── loss.py
     │   ├── models.py
     │   ├── trainer.py
     │   └── uda_train.py
     └── infer.py
```

<br></br>
## 💻 소스 코드 설명 <a name = 'Code'></a>
- `pseudo_train.py` : pseudo labeling을 활용한 train 코드
- `earlyStop.py` : overfitting을 방지하기 위한 earlyStop 코드
- `loss.py` : 학습용 loss함수 정의
- `models.py` : 모델 정의
- `trainer.py` : train 코드를 간결하게 하기위한 train helper function
- `uda_train.py` : [[Paper]](https://arxiv.org/abs/1904.12848) 해당 논문에서 나오는 UDA기법을 적용해보려고 시도한 코드



<br></br>
## [훈련 & 추론방법]
- Train
    ```
    sh experiments/experiment_version.sh
    ```
    or 
    ```
    sh pseudo_train.sh
    ```

- Infer
    ```
    sh infer.sh
    ```
<br></br>
