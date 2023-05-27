# speech grading mutimodal
### Data:

1. `**Good Dataset**`:

Tedlium Dataset: TED 에서 발표한 우수한 발표를 대상으로 제작된 데이터셋 (hugging face dataloader로 다운할 경우에 더욱 빠른 처리 가능해서 이를 통해서 다운 및 처리)

1. `**Poor Dataset**`

Mosi dataset: 영상에서의 감정, 음성의 감정, 그리고 텍스트에서의 감정 등 다양한 모달리티를 포함한 잘못된 발표 데이터셋을 제공합니다. 이 데이터셋은 발표자가 실수하거나 감정적으로 불안정할 때 발생하는 다양한 오류와 잘못된 발표를 포함(https://www.kaggle.com/datasets/mathurinache/cmu-mosi)

---

### Unimodal:

1. Text model: **Measurement of the order of presentation sentences, percentage of off-topic sentences, and grammatical accuracy**
    1. `**Bertopic**`: 
        1. 토픽 유사도를 이용해서, 전체 문장 대비 각 문장들의 토픽 유사도를 측정, 토픽에서 벗어날 경우에 코사인 유사도 값이 감소 (0에 수렴)
        2. bertopic dataset을 통한 pretrain 된 모델을 사용할 예정
        3. 토픽 모델이 잘 뽑혔는지에 있어서는 특정 지표보다는 heuristic한 방법론이 적절하다고 생각 → LDA 결과와 같이, 평가 지표가 모호
    2. `**CER**`:
        1. 기본 문장과, 그 문장에서 문법을 수정한 문장을 라벨로 사용하여 학습을 진행, 특정 문장에서 문법을 어떻게 수정해야 하는지에 대해서 학습, 이를 통해서 얼마나 기본 문장의 문법이 틀렸는지를 추출 가능
        2. 선 음성 인식 정확도를 평가하는데 사용되는 문자오류율인 CER를 LSTM 기반으로 구현, 음성인식 결과로 나온 텍스트 데이터와 문법 교정된 데이터를 input으로 넣으면 문맥적 정보를 포함한 문자 오류 비율을 출력
        3. lang8 데이터셋 사용
        4. 참고 노션: [https://www.notion.so/CER-Model-d9eefe13b7f84a21ae8338b1e63f5c73?pvs=4](https://www.notion.so/CER-Model-d9eefe13b7f84a21ae8338b1e63f5c73)
    3. `**Next Sentence Probability**`:
        1. roberta 모델을 활용하여, 다음 문장이 이전 문장 뒤에 적절히 위치하고 있는지에 대한 확률인 NSP를 통해서 발표 문장의 순서가 적절한지 평가
        2. 기존 pretrain된 roberta 모델을 hugging face로 끌어온 이후에, Good Dataset과 Poor Dataset을 결합하여 라벨링된 데이터셋을 통해서 fine tuning 진행
            1. hyper parameter: epoch: 30, k-fold Cross Validation: 5 → 아직 하이퍼파라미터 튜닝이 필요
            2. 더 많은 데이터셋으로 학습이 필요: epoch 27 정도에서 overfitting 발생: 데이터 셋이 적기 때문인듯???
            
2. Audio model: **Extracting intra-speech features to measure speech instability**
    1. `**speech to text api in google**`:
        1. 발표문의 오디오 데이터를 통해서 텍스트를 추출하는 api를 사용
        2. Calculate the cosine similarity using the extracted data and the real transcript.(Good Dataset: Average 0.8, Bad Dataset: Average 0.5 to 0.7)
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c2132d23-30c3-4773-8e11-fdd4065801a9/Untitled.png)
        
    2. `**openSMILE**`: open-source toolkit for audio feature extraction and classification of speech and music signals
        1. JitterLocal feature: 이 특징은 주파수 변동률을 나타냅니
        목소리의 강도 변동에 해당, 음성의 주파수 변화를 측정하여 발성의 안정성을 추정 → 다른 특징들에 비해서 잘한 발표와 못한 발표에서 특징 차이가 뚜렷하게 나타남
        2. dataset: Tedlium audio, Mosi audio 를 통해서 특징 추출
        
3. Vision model: **Point the gaze and measure the movement of the gaze**
    1. `**opencv**`: 
        1. 영상으로 처리하기 위해서는 많은 computing source 필요, 하지만 부족한 computing source에 대비하기 위해서 1초당 한장씩의 프레임 사진을 추출, 이를 모델 학습과 테스트에 사용하기 위해서 opencv모델을 통해서 전처리 진행
        2. Good Dataset, Bad Dataset 모두 처리 완료
    2. `**gazeNet + cubic spline model**`: 
        1. gazeNet 모델을 통해서 사진에서 gaze detection 진행, 이를 통해서 시선이 어디를 바라보고 있는지에 대한 수치를 얻을 수 있음
        2. 각 프레임당 gaze 의 위치를 pointing 진행, 이를 timeseries 모델과 같이 사용하기 위해서, cubic spline model을 사용해서 보간 진행중
        3. 진행사항: gazeNet python version을 3점대로 수정하는 작업을 진행 후에, gazeNet 처리중, 서있는 데이터셋에 대해서 작동을 잘하는지 판별 중이며, 앉아있는(시선이 잘 보이는) 데이터셋에는 작동 뛰어남.