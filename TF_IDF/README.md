# TF_IDF + roberta-large-mnli + logistic

### 파일 설명:

1. topic_train_preprocess.ipynb
    - `TF_IDF`:
        - Term Frequency - Inverse Document Frequency
        - <https://wikidocs.net/127853>
    - `roberta-large-mnli`:
        - <https://huggingface.co/roberta-large-mnli>
    - `descr`:
        - score: TF_IDF로 뽑아낸 topic과 text의 유사도를 계산
        - count: text의 단어 수를 계산
        - ['idx', 'score', 'count', 'label'] 형식으로 my_train_data.csv 작성

2. topic_train_logistic.ipynb
    - `logistic`
    - `descr`:
        - my_train_data.csv의 ['score', 'count'] + 'label' 학습
        - logistic_model.pkl 에 학습된 모델 저장

3. topic_train.ipynb
    - `descr`:
        - 사용할 model & dataset 경로 수정
        - ['idx', 'score', 'count'] 형식으로 my_test_data.csv 작성
        - ['score', 'count'] 출력
        - predict label 출력

4. logistic_model.pkl
    - `descr`:
        - test에 사용할 모델

---

### 사용 방법:

1. logistic_model.pkl 저장
2. topic_train.ipynb 에서 사용할 model & dataset 경로 수정
3. topic_train.ipynb 실행
