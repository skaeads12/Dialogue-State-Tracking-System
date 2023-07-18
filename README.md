# Dialogue-State-Tracking-System
## 프로젝트 소개

본 저장소는 KLUE-WoS 데이터셋을 바탕으로 DST(Dialogue State Tracking) 시스템을 구현하였습니다. DST를 구현하기 위해 사용한 모델은 다음과 같습니다.

- BERT
- RoBERTa
- ELECTRA

3가지 모델은 Feature Extraction을 제외한 모든 모듈이 동일한 구성을 가지고 있으며, 모두 End-to-End로 구현되었습니다.

## 사용 방법

본 저장소에는 데이터셋이 포함되어있지 않습니다. 데이터셋은 https://github.com/KLUE-benchmark/KLUE/tree/main에서 WoS(Wizard of Seoul)을 참조해주세요.
실험에 사용한 파일의 트리 구조는 다음과 같습니다.

```
.
└── data
    ├── train.json
    ├── eval.json
    └── ontology.json
```

다음 명령어는 학습에 사용됩니다.

```console

python train.py -td data/train.json \      # train set dir
                -ed data/eval.json \       # develop set dir
                -od data/ontology.json \   # ontology dir
                -sd bert-result/ \         # save dir
                -pt klue/bert-base \       # pretrained tokenizer
                -pm klue/bert-base \       # pretrained model
                -e 3 \                     # num of epochs
                -b 64 \                    # batch size
                -ml 512                    # max length

```

다음 명령어는 시험에 사용됩니다. 본 실험에서는 WoS 데이터의 검증 데이터를 사용했습니다.

```console

python predict.py -td data/eval.json \                # test set dir
                  -od data/ontology.json \            # ontology dir
                  -sd bert-predictions.json \         # save dir
                  -pt bert-result/checkpoint-2757 \   # pretrained tokenizer
                  -pm bert-result/checkpoint-2757 \   # pretrained model
                  -b 64 \                             # batch size
                  -ml 512                             # max length

```

## 모델 구조

모델은 세 가지 모듈로 구성되어있습니다. 세부적인 구현 내용은 model.py를 참조하십시오.

- ENCODER
- Gate Classifier
- Span Predictor

### ENCODER

ENCODER는 BERT, RoBERTa, ELECTRA 모델을 사용하며, 고전적인 Encoding 방식을 사용합니다.

- Token, Position Embedding
- Self-Attention Layer, Fully Connected Layer
- Feed Forwarding

### Gate Classifier

Gate Classifier는 KLUE-WoS 데이터셋의 ontology에 따라 모든 클래스를 5개의 레이블로 예측하는 방식으로 구현되었습니다.
Gate Classification 규칙은 다음과 같습니다.

- Classifier: 관광-경치 좋은, 관광-교육적, 관광-도보 가능, ..., 택시-출발지 등 총 45개
- Labeling: none, yes, no, dontcare, span 등 총 5개

45개의 클래스에 대하여 5개의 레이블을 예측하고, 해당 레이블의 상태에 따라 최종 예측값을 결정합니다.

- **none**: 클래스 해당 없음
- **yes**: 해당 클래스 긍정(관광-경치 좋은, 관광-교육적 등에 해당)
- **no**: 해당 클래스 부정(관광-경치 좋은, 관광-교육적 등에 해당)
- **dontcare**: 해당 클래스 상관 없음(관광-경치 좋은, 관광-교육적 등에 해당)
- **span**: 해당 슬롯의 값이 발화에서 추출해야 하는 경우(관광-이름, 관광 종류 등에 해당)

### Span Predictor

Span Predictor는 일반적인 Span Prediction 방법에 따라 구현되었습니다. 이는 발화에서 정답을 찾아내야 하는 상황에서 사용됩니다.
클래스 개수에 따른 Span Predictor가 존재하며, 각각 2개의 Classifier를 가지고 있습니다.

- Predictor: 관광-경치 좋은, 관광-교육적, 관광-도보 가능, ..., 택시-출발지 등 총 45개
- Labeling: start_position, end_position 등 총 2개

### 모델 구조도

![dst](https://github.com/skaeads12/Dialogue-State-Tracking-System/assets/45366231/3c3f350c-9c9b-4a27-9b4c-78821ffe1015)

## LICENSE
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

