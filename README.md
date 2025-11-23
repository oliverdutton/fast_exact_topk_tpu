[Archived] Moved to [tallax](https://github.com/oliverdutton/tallax), accelerated JAX for TPU using Pallas. Runtime improved further to 40µs on v5e, 33x speedup.

# Fast Exact Top-K on TPU

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oliverdutton/fast_exact_topk_tpu/blob/main/fast_exact_topk_tpu.ipynb)

<img src="https://github.com/oliverdutton/fast_exact_topk_tpu/blob/main/fast_exact_topk_tpu.png" width="840" height="600">

This repository provides a Pallas implementation of an exact `top-k` operation that is **~11x faster** than the standard XLA version on TPUs.

| Hardware       | XLA (Baseline) | This Work (Pallas) | Speedup |
| :------------- | :------------- | :----------------- | :------ |
| v5e            | 1320µs         | 120µs              | ~11x    |
| v6e            | 1010µs         | 116µs              | ~8.7x   |


*Tested on various TPUs with `k=64`, `batch_size=32`, and `vocab_size=201,088`.*

---

## The Problem: Top-K as a Bottleneck

In the final layer of a Large Language Model (LLM), the model produces a logit for every word in its vocabulary (e.g. 201,088 for GPT-OSS). During text generation, a sampling method is used to select the next word. A common technique, **top-k sampling**, restricts the choice to the `k` words with the highest logits to ensure coherent output.

This `top-k` operation can be a bottleneck, taking longer than the massive matrix multiplication that precedes it. This inefficiency prevents the hardware from being fully utilized, and accelerators not going brrr makes me sad.

This work addresses that bottleneck with an algorithm designed specifically for the TPU's architecture.

---

## The Algorithm: A TPU-Native Approach

### General Idea

TPUs achieve maximum performance when computations avoid data shuffling across the hardware's 128 "lanes" ([see below if unfamilar](#background-on-tpus)). Our algorithm minimizes this by splitting the problem into two stages:

1.  **Block-wise Candidate Search**: First, we partition the full vocabulary into 128 blocks. Within each block, we perform a highly efficient local search for the `top-m`. This avoids  data shuffling across the hardware's 128 lanes.
2.  **Final Top-K Selection**: We then gather the candidates from all blocks (a set of $128 \times m$ elements) and perform a final `top-k` operation on this much smaller, filtered subset.


### Ensuring Exactness

While many works focus on *approximate* `top-k`[^6] [^7] for performance gains, this implementation provides a fast **exact** result.

[^6]: [Approximate Top-k for Increased Parallelism](https://arxiv.org/pdf/2412.04358) 
[^7]: [Faster Approx. top-K: Harnessing the full power of two stages](https://arxiv.org/pdf/2506.04165)

To guarantee the result is **exact** and not an approximation, the value of `m` is determined dynamically. The algorithm iteratively increases `m`, checking after each iteration if the collected candidates are sufficient to contain the full global `top-k`.

The process stops once there are more than `k` values greater (or equal to) the highest of the `m`-th highest value across the 128 blocks. At this point, we can be certain that no element outside our candidate pool (of the block `top-(m-1)`s) could possibly be in the final `top-k` set. This use of conditional early stopping ensures correctness while maximizing speed.

---

## Repository Structure

* `fast_exact_topk_tpu.py`: The core implementation of the Pallas `block top-k` kernel and followup `top-k` on filtered candidate pool.
* [Jupyter Notebook](https://colab.research.google.com/github/oliverdutton/fast_exact_topk_tpu/blob/main/fast_exact_topk_tpu.ipynb): A notebook containing TensorBoard profiling and calculations for the number of iterations required for convergence.

---

## Limitations and Future Work

This is an early-stage implementation. Contributions are welcome! Key areas for future development include:
* Adding comprehensive unit tests.
* Improving type hinting and code documentation.
* Fusing the `top-k` kernel directly with the preceding `matmul` operation.
* Extending support for multi-TPU device configurations.
* Try pack bfloat16 value and uint16 index
 into 32 bits and do comparisons in float32

---
## Background on TPUs
TPUs are great machines, their hardware is awesome. For instance, the VPU, it 'is a 2D vector arithmetic unit of shape (8, 128) where the 128 dimension is referred to as lane axis and the dimension of 8 is referred to as the sublane axis'[^3]. Due to this 2D array structure imprinted in the hardware, operations between lanes after slow, between sublanes are okay and between full chunks is fastest[^4]. This means algorithms designed for other accelerators can be inefficient on TPU.

[^3]: [JAX scaling-book](https://jax-ml.github.io/scaling-book/tpus/)
[^4]: [Pallas TPU docs](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
