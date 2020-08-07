# time-series-forecast-using-Transformer

# Transformer 모델을 시계열 예측에 적용.

논문: Bryan Lim and Nicolas Loeff. 2019. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.

github: https://github.com/google-research/google-research/tree/master/tft

# Time series 문제의 주요 과제

* 며칠을 보고 예측할건지. (lookback period)
* 며칠을 예측할건지(multiple time steps)
* 어떤 변수를 쓸 지.
* past input 과 future input 을 어떻게 반영할지.
* static(time-invariant) covariates 는 어떻게 반영할지.


# 이 모델의 기능

* deep neural networks 가 나오면서 traditional time-series models 에 비해 성능 향상.
+ attention-based methods 로 과거의 특정 포인트를 enhance 할 수 있다.
여기에 past/future input 을 구분하고 static covariates 까지 반영할 수 있게 만들어졌다.

* 기존의 LIME 과 SHAP 을 이용한 explainability 는 시간을 반영하지 못했다.
얘들은 학습이 다 된(pre-trained) model로 구하는 methods.
attention weights 로 시간까지 반영한(long-term dependencies) explainability 를 얻을 수 있다.

<img width="275" alt="FIG1" src="https://user-images.githubusercontent.com/49193062/89627930-7eca2780-d8d6-11ea-9315-458a7bb1823c.PNG">

이게 multi-horizon forecasting 문제의 일반적인 구조다.
Observed Inputs: target과 함께 나오는 inputs
Known Inputs: 예측 시점에 미래의 값을 미리 알 수 있는 inputs
Static Covariates: 시간에 관계없는 변하지 않는 features
Point Forecasts & Prediction Intervals: 예측값과 구간. 이 모델에서는 quantile regression 으로 각 time step 에서 10th, 50th, 90th percentiles를 구했다.

  <img width="579" alt="FIG2" src="https://user-images.githubusercontent.com/49193062/89627934-7f62be00-d8d6-11ea-93be-e6865249dc3d.PNG">

# Model Architecture
1. Gating Mechanisms
Gated Residual Network(GRN) 을 새롭게 제시했다.

<img width="257" alt="GRN" src="https://user-images.githubusercontent.com/49193062/89627939-7ffb5480-d8d6-11ea-8d9d-12ebebe1fd1f.PNG">

ELU 는 Exponential Linear Unit activation function 이다.
GRN 에는 두 가지 목적이 있다. 
non-linear processing을 적절히 적용하는 것과 relevant variables 에만 적용하는 것.
ELU -> GLU -> Residual Connection 을 적용하는 layer.

<img width="248" alt="GLU" src="https://user-images.githubusercontent.com/49193062/89627935-7f62be00-d8d6-11ea-8fa4-70c3785042e6.PNG">

GLU 가 GRN이 original input 에 기여하는 정도를 조절할 수 있게 해준다.
추가로 context vector 까지 적용할 수 있다.
Static Covariate 를 Variable Selection 과정이나 Static Enrichment 과정에 적용할 때 쓴다.
training 때 Dropout 적용한다.



2. Variable Selection Networks
instance-wise variable selection 을 제공한다.
significant variables / unnecessary variables 를 구분해낸다.
categorical 를 d dimensional vector 로 embedding.
continuous variables 를 d dimensional vector로 linear transformation.
skip connections 에 쓰일 수 있게 dimension 맞춰줄 것.



3. Static covariate Encoders
static metadata 의 representations 을 다소 복잡하게 만들었다.
4 가지 GRN encoders 로 서로 다른 context vectors 를 만들어 곳곳에 씀.



4. Interpretable Multi-Head Attention
서로 다른 time steps 의 long-term relationships 을 학습하기 위해 self-attention을 씀.

<img width="266" alt="attention1" src="https://user-images.githubusercontent.com/49193062/89627908-7b36a080-d8d6-11ea-9ade-3d58561bbf49.PNG">

일반적인 attention. value 를 scale 한다.

<img width="271" alt="attention2" src="https://user-images.githubusercontent.com/49193062/89627910-7b36a080-d8d6-11ea-8967-b3dda9502616.PNG">

A는 normalization function.
scaled dot-product attention 을 썼다.

<img width="294" alt="attention3" src="https://user-images.githubusercontent.com/49193062/89627912-7bcf3700-d8d6-11ea-90b2-42bd24a6f0c3.PNG">

keys, queries, values 에 head-specific weights 곱해주고 그걸 attention.
H들을 concatenate 해서 linearly combine.

<img width="326" alt="attention4" src="https://user-images.githubusercontent.com/49193062/89627914-7c67cd80-d8d6-11ea-8af1-d685dda94818.PNG">

각 head 에 다른 values 가 사용된 걸 고려하면, 
attention weights 만으론 feature 의 전반적인 importance를 분석하기에 충분하지 않다.
그래서 모든 heads의 value weights 를 share 하고, heads를 additive aggregation 한다.
Eq. (15) 에서, 각 head 가 다른 temporal patterns 를 학습하는 게 가능한 걸 볼 수 있다.



5. Decoder

5.1 Locality Enhancement with Sequence-to-Sequence Layer
anomalies, change-points, cyclical patterns 등 다양한 surrounding 정보를 읽어야 한다.
locality enhancement 를 위해 CNN을 쓰기도 하지만,
past/future inputs 를 encoder/decoder 구조로, seq2seq 모델을 쓰는 게 더 효과적.
여기에 LSTM 을 썼는데, positional encoding 을 대체할 수 있다. time ordering 에 inductive bias 를 줘서.
static metadata 도 적용하기 위해 context vector 를 첫 LSTM 의 cell state 와 hidden state 를 initialize 할 때 썼다.
거기에 gated skip connection 까지.

<img width="302" alt="attention5 1" src="https://user-images.githubusercontent.com/49193062/89627915-7c67cd80-d8d6-11ea-9c7c-104e42810c85.PNG">

5.2 Static Enrichment Layer

<img width="261" alt="attention5 2" src="https://user-images.githubusercontent.com/49193062/89627917-7d006400-d8d6-11ea-9a40-b368f1966325.PNG">

static covariates도 매우 중요하니 GRN 에 넣으면서 context vector 를 껴준다.

5.3 Temporal Self-Attention Layer
이후 self-attention 을 적용한다.
decoder masking 까지 해주고

<img width="297" alt="attention5 3" src="https://user-images.githubusercontent.com/49193062/89627918-7d98fa80-d8d6-11ea-983a-c0d1c5eb595c.PNG">

그 결과를 gating layer 에 넣어준다.


5.4 Position-wise Feed-Forward Layer

<img width="246" alt="attention5 4" src="https://user-images.githubusercontent.com/49193062/89627925-7e319100-d8d6-11ea-8af8-68d955061e62.PNG">

추가 non-linear processing.
이건 layer 전체 share한다.


6. Quantile Outputs

<img width="342" alt="quantile loss" src="https://user-images.githubusercontent.com/49193062/89627943-8093eb00-d8d6-11ea-9686-550633715b0a.PNG">

동시에 세 가지 점을 예측한다. 10th, 50th, 90th percentiles.
training 을 위해 quantile loss 를 사용한다.


Optional

<img width="732" alt="variable importance table" src="https://user-images.githubusercontent.com/49193062/89627901-796cdd00-d8d6-11ea-8896-3abf37792eb5.PNG">

<img width="351" alt="temporal patterns" src="https://user-images.githubusercontent.com/49193062/89627944-8093eb00-d8d6-11ea-9d4c-7343d5af2989.PNG">

variable selection weights 와 attention weights 를 사용해
variable importance 와 temporal patterns 을 알 수 있다.



# Codes

download_data: data 가져와서 1차 preprocess
hyperparam_optimization: hyperparameter random search
train_fixed_params: 위에서 찾은 hyperparameter로 
step-by-step: 예시로 layer 순차적으로 살펴보기
libs/tft_model: model architecture
data_formatters/base: 2차 preprocessing frame

