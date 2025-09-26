# Fast exact top-k on TPU
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oliverdutton/fast_exact_topk_tpu/blob/main/fast_exact_topk_tpu.ipynb)

<img src="https://github.com/oliverdutton/fast_exact_topk_tpu/blob/main/fast_exact_topk_tpu.png" width="840" height="600">


An implementation of exact top-k that's ~11x faster than the XLA version[^1].
[^1]: (v6e TPU, top-64, batch_size=32, vocab_size=201,088: XLA 1320us, this work 120us).

## Background
### Top-k in LLMs
In the last layer of LLMs, the hidden state is 'unembedded' to logits for each word in the models vocab [e.g. for GPT-OSS this vocab size is 201,088]. Those logits are then sampled. Often to ensure what's sampled is sensical it's advised to hard restrict sampling to just words with the largest logits (usually the top 50). That XLA top-k operation can take far longer than the actual matmul on TPU which means TPU is not going brrrr which is sad.

### TPUs
TPUs are great machines, their hardware is very cool. For instance, the VPU, it 'is a 2D vector arithmetic unit of shape (8, 128) where the 128 dimension is referred to as lane axis and the dimension of 8 is referred to as the sublane axis.' [(JAX scaling-book)](https://jax-ml.github.io/scaling-book/tpus/). Due to this 2D array structure imprinted in the hardware, operations between lanes after slow, between sublanes are okay and between full chunks is fastest. [(Pallas TPU docs)](https://docs.jax.dev/en/latest/pallas/tpu/details.html). This leads to different optimal algorithms for TPU than other accelerators which this code addresses.

## Algorithm
### Explanation
To better match TPU hardware and minimize ops across the lane axis you can split the vocab in 128 blocks and do top-m on every block very efficiently through bubble sort. Using conditionals you can increment m until the 128 top-m's contain the overall top-k elements, then do top-k on that much smaller filtered top-m subset.[^2] We use early stopping and conditionals to ensure both maximum speed and that the top-k is exact.

[^2]: In practice, we increment get the 128 top-(m+1)'s, increasing m until the largest of the 128 (m+1)'th largest values is too small to be in the top-k and hence up until top-m must contain the top-k.

The concept is not new and approximate top-k has been addressed in many previous works e.g. [Approximate Top-k for Increased Parallelism](https://arxiv.org/pdf/2412.04358) and [Faster Approx. top-K: Harnessing the full power of two stages](https://arxiv.org/pdf/2506.04165) however this this work is focussed on exact top-k.

### Code structure
`fast_exact_topk_tpu.py` contains the implementation, while a  [colab](https://colab.research.google.com/github/oliverdutton/fast_exact_topk_tpu/blob/main/fast_exact_topk_tpu.ipynb) contains some tensorboard profiling and some calculations about how many iterations convergence requires.

## Apology
Many things are missing including tests, typing, fusing into the matmul and multi-TPU. Apologies.

