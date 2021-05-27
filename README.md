# Informer

[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) (AAAI2021)

### Introduction

전기 사용량/주가 예측 등 Long Range의 시계열 데이터를 예측하는 **Long Sequence Time-Series Forecasting(LSTF)** 문제에서 LSTM이 자주 사용되었으나, LSTM은 *Increasingly* Long Range Dependency를 잡아내는 데 문제가 있다.

최근 Self-Attention을 기반으로 한 **Transformer**가 Long Range Dependency를 잘 학습하는 것으로 알려져 있고, Natural Language Processing(NLP) 분야에서 괄목할 만한 성능 향상을 이끌어냈다.

Transformer는 LSTF 문제에서도 엄청난 Potential을 보여주었으나, 다음과 같은 근본적인 문제점들을 갖는다.

- The **quadratic computation** of self-attention
- The **memory bottleneck** in stacking layers for long inputs
- **The speed plunge in predicting long outputs**

Ouput을 예측하는 데 많은 시간이 걸리는 것은 Encoder-Decoder 구조에서의 Decoder가 Inference시 통상적으로 한 번에 하나의 Item(NLP에서는 Token)만을 예측하기 때문이다.

(이는 Vaswani의 [Transformer](https://arxiv.org/abs/1706.03762)뿐 아니라 Encoder-Decoder, 특히 Decoder 구조를 갖는 [Seq2Seq](https://arxiv.org/abs/1409.3215), [GPT](https://arxiv.org/abs/2005.14165)와 같은 모델들에 모두 해당하는 문제라고 생각한다.)

해당 논문은 이러한 문제점들을 해결하기 위해 Efficient Architecture의 Transformer "***Informer***"를 제안한다.
