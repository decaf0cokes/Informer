# Informer

[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) (AAAI2021)

## Introduction

전기 사용량/주가 예측 등 Long Range의 시계열 데이터를 예측하는 **Long Sequence Time-Series Forecasting(LSTF)** 문제에서 LSTM이 자주 사용되었으나, LSTM은 *Increasingly* Long Range Dependency를 잡아내는 데 문제가 있다.

최근 Self-Attention을 기반으로 한 **Transformer**가 Long Range Dependency를 잘 학습하는 것으로 알려져 있고, Natural Language Processing(NLP) 분야에서 괄목할 만한 성능 향상을 이끌어냈다.

Transformer는 LSTF 문제에서도 엄청난 Potential을 보여주었으나, 다음과 같은 근본적인 문제점들을 갖는다.

- The **quadratic computation** of self-attention
- The **memory bottleneck** in stacking layers for long inputs
- **The speed plunge in predicting long outputs**

Ouput을 예측하는 데 많은 시간이 걸리는 것은 Encoder-Decoder 구조에서의 Decoder가 Inference시 통상적으로 한 번에 하나의 Item(NLP에서는 Token)만을 예측하기 때문이다.

(이는 Vaswani의 [Transformer](https://arxiv.org/abs/1706.03762)뿐 아니라 Encoder-Decoder, 특히 Decoder 구조를 갖는 [Seq2Seq](https://arxiv.org/abs/1409.3215), [GPT](https://arxiv.org/abs/2005.14165)와 같은 모델들에 모두 해당하는 문제라고 생각한다.)

해당 논문은 이러한 문제점들을 해결하기 위해 Efficient Architecture의 Transformer "***Informer***"를 제안한다.

## Informer

### ProbSparse Self-Attention

Canonical Self-Attention은 다음과 같이 표현될 수 있다. (Q, K, V = query(s), key(s), value(s) 및 q, k, v = *i*-th row in Q, K, V)

![Canonical_Self_Attention](./imgs/Canonical_Self_Attention.svg)

수식에서 f()는 query와 key간의 Attention을 계산하는 함수로, Transformer에서는 Scaled Dot-Product를 사용한다.

위와 같은 계산 방식은 **Quadratic Times**의 Dot-Product 연산을 비롯해 O(__len(Q)*len(K)__)의 메모리(Size of Attention Matrix)를 필요로 한다.

#### "본 논문에서는 Canonical Self-Attention이 Potentially Sparse하다는 점을 지적하며 Computation/Memory 측면에서 효율적인 ProbSparse Self-Attention을 제안한다."

ProbSparse Self-Attention의 핵심 아이디어는 "불필요한 query들에 대한 Attention을 계산하지 않겠다"는 것이다.

불필요한 query란, 모든 key들과의 Attention 값이 비슷하여 Q, K, V 간의 Self-Attention이 value들의 단순 합이 되도록 하는 query를 의미한다.

예를 들어, 특정 query q'가 5개의 key들과 [2, 2, 2, 2, 2]의 Attention을 갖는 경우 q', K, V 간의 Self-Attention은 value들의 단순 합이 되어버리며, 이 때 K가 긴 Sequence를 가진다면 Softmax를 취한 Attention값들이 모두 0에 수렴하게 된다(-> Sparse).

즉, 저자가 지적한 Sparse하다는 점은 Self-Attention 측면에서도 의미가 없는 경우이다.

본 논문에서는 불필요한 query를 걸러내기 위해 다음과 같이 query별 ***Sparsity Measurement***, *M*을 계산한다.

![Sparsity_Measurement_1](./imgs/Sparsity_Measurement_1.svg)<br/>

![Sparsity_Measurement_2](./imgs/Sparsity_Measurement_2.svg)

확률 p가 Uniform Distribution, q와 차이가 클수록 *M*값은 증가하며(KL Divergence를 사용했으므로), *M*값이 큰 c*ln(len(Q))개의 query들만을 선택하여 Attention을 계산한다.

ProbSparse Self-Attention은 Canonical Self-Attention에 비해 개선된 O(__len(K)*ln(len(Q))__)의 Computation과 Memory를 요구한다.

추가로, *M*값을 계산하는데 O(len(Q)*len(K))의 연산이 발생하는 점을 보완하기 위해, Long Tail Distribution에 의거하여  ln(len(K))개의 key들만으로 query별 Sparsity Measurement를 계산(재정의)한다.

![Sparsity_Measurement_3](./imgs/Sparsity_Measurement_3.svg)

### Self-Attention Distilling

![Self_Attention_Distilling](./imgs/Self_Attention_Distilling.png)

Encoder에서 ProbSparse Self-Attention 이후 1-D Convolution과 Stride-2 MaxPooling을 수행하여 Input Sequence를 절반으로 만들어 준다.

### Generative Decoder

![Decoder_Input](./imgs/Decoder_Input.svg)

Auto-Regressive한 방식의 Canonical Decoder와 달리, Target Sequence만큼의 Placeholder(0 Token)를 Input으로 넣어줌으로써 한 번의 Inference에 모든 Position 값을 예측한다.

Target Sequence 직전, 일정 기간의 Sequence를 추출하여 Start Token으로 사용한다.
