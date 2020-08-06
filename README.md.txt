Transformer 모델을 시계열 예측에 적용.

논문:

논문 설명. 정리 필요.

Time series 문제의 주요 과제

며칠을 보고 예측할건지. (lookback period)
며칠을 예측할건지(multiple time steps)
어떤 변수를 쓸 지.
past input 과 future input 을 어떻게 반영할지.
static(time-invariant) covariates 는 어떻게 반영할지.


이 모델의 기능

deep neural networks 가 나오면서 traditional time-series models 에 비해 성능 향상.
+ attention-based methods 로 과거의 특정 포인트를 enhance 할 수 있다.
여기에 past/future input 을 구분하고 static covariates 까지 반영할 수 있게 만들어졌다.

기존의 LIME 과 SHAP 을 이용한 explainability 는 시간을 반영하지 못했다.
얘들은 학습이 다 된(pre-trained) model로 구하는 methods.
attention weights 로 시간까지 반영한(long-term dependencies) explainability 를 얻을 수 있다.







코드 설명 간단히.
코드에 주석도 달까.
