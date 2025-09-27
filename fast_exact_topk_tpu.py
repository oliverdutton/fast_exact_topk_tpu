# !pip install tensorboard tensorboard-plugin-profile
# Pip install will require a restart, then comment the code


import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def blockwise_topk(
  logits, k, block_topk_val=None, block_topk_index=None, start_k=0, num_blocks=128, mode="jax"
):
  """Compute blockwise top-k."""
  ntokens = logits.shape[0]

  if start_k != 0 and (block_topk_val is None or block_topk_index is None):
    raise ValueError(
      "start_k must be 0, unless precomputed top-(start_k) in a buffer is provided in block_topk_val and block_topk_index"
    )
  if mode == "jax":
    block_topk_val = [
      jnp.full((ntokens, num_blocks), float("-inf"), dtype=logits.dtype)
      for i in range(k)
    ]
    # TODO?: Could use uint16 when vocab size < 4M if hardware supports
    block_topk_index = [
      jnp.full((ntokens, num_blocks), -1, dtype=jnp.int32) for i in range(k)
    ]
  elif mode == "pallas":
    if block_topk_val is None or block_topk_index is None:
      raise ValueError(
        "Pass through of block_topk_val and tok_index expected for pallas topk."
      )

  def while_body(i, while_carry):
    block_topk_val, block_topk_index = while_carry

    if mode == "pallas":
      vals_carry = logits[..., pl.dslice(num_blocks * i, num_blocks)]
    elif mode == "jax":
      vals_carry = jax.lax.dynamic_slice_in_dim(
        logits, num_blocks * i, num_blocks, axis=1
      )
    else:
      raise ValueError(
        "mode must be either `pallas` and a memory ref or `jax` and an array"
      )

    index_carry = jnp.full((ntokens, num_blocks), i, jnp.int32)

    for i in range(k):
      if i < start_k:
        # Nothing will be exchanged into the completed block topk, we just need
        # to invalidate it from flowing downward. So we check if it's already
        # found and invalidate if so.
        vals_carry = jnp.where(index_carry == block_topk_index[i], float("-inf"), vals_carry)
      else:
        # Sinking sort
        mask = vals_carry > block_topk_val[i]
        # TODO: Consider packing bfloat16 val and uint16 index into single uint32
        # and packed sort as in
        # https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/topk_details/_topk_forward.py
        block_topk_val[i], vals_carry = (
          jnp.where(v, vals_carry, block_topk_val[i]) for v in (mask, ~mask)
        )
        block_topk_index[i], index_carry = (
          jnp.where(v, index_carry, block_topk_index[i]) for v in (mask, ~mask)
        )
    return (block_topk_val, block_topk_index)

  (block_topk_val, block_topk_index) = jax.lax.fori_loop(
    0, logits.shape[-1] // num_blocks, while_body, (block_topk_val, block_topk_index)
  )
  return block_topk_val, block_topk_index


# Pallas kernel scaffolding
def topk_blockwise_superset_kernel(
  logits_ref,
  block_topm_val_refs,
  block_topm_index_refs,
  max_m_ref,
  flag_ref,
  num_blocks: int = 128,
  k: int = 64,
  m_schedule: tuple[int] | None = None,
):
  """Compute blockwise top-m's until they contain global top-k."""
  ### Initialize refs
  shape = block_topm_val_refs[0].shape
  for i in range(len(block_topm_val_refs)):
    block_topm_val_refs[i][...] = jnp.full(shape, float("-inf"), dtype=logits_ref.dtype)
    block_topm_index_refs[i][...] = jnp.full(shape, -1, dtype=jnp.int32)

  block_token = logits_ref.shape[0]
  for i in range(block_token):
    # Worst case m = k
    max_m_ref[pl.program_id(0) * block_token + i] = k

  # flag for termination of while loop
  flag_ref[0] = 0

  ### Run increasing block topk, until sure overall topk present
  if m_schedule is None:
    m_schedule = (5, 8, 12)
  # Ensure worst case of all k in one block is covered
  m_schedule = (0,) + m_schedule + (k,)

  for completed_m, m in zip(m_schedule, m_schedule[1:]):

    @pl.when(flag_ref[0] == 0)
    def _():
      topk_vals, topk_indexs = blockwise_topk(
        logits_ref,
        # bf16, bf16 -> i1 mask not supported on v5e so we cast to f32
        # TODO: check v6e, bf16 comparitor and make model specific
        block_topk_val=jax.tree.map(
          lambda ref: ref[...].astype(jnp.float32), block_topm_val_refs
        ),
        block_topk_index=jax.tree.map(lambda ref: ref[...], block_topm_index_refs),
        k=m,
        start_k=completed_m,
        mode="pallas",
      )

      for i in range(completed_m, m):
        block_topm_val_refs[i][...] = topk_vals[i].astype(block_topm_val_refs[i].dtype)
        block_topm_index_refs[i][...] = topk_indexs[i].astype(
          block_topm_index_refs[i].dtype
        )

      # Stopping criterion check
      # To find top-k values of a set, we can split into N subsets,
      # and sort the largest, 2nd-largest, 3-rd largest, ..., m-th largest values for each subset
      # When in the superset of top-(m-1) subsets there are more than k values
      # larger (or equal than) the largest m'th largest value from the subsets
      # then the top-(m-1) subsets must contain the top-k of the set.
      # We run a schedule of m's until we have that full top-k found.
      pivot_point = topk_vals[m - 1].max(-1, keepdims=True)
      n_larger = (
        sum([(v >= pivot_point) for v in topk_vals[: m - 1]])
        .astype(jnp.float32)
        .sum(-1)
      )
      # flag SMEM used to check if all searches terminated
      flag_ref[0] = 0
      for i in range(block_token):
        blockwise_topm_contains_topk = n_larger[i] >= k
        flag_ref[0] += blockwise_topm_contains_topk
        # Store when the criteria was hit for each query
        token_index = pl.program_id(0) * block_token + i
        max_m = max_m_ref[token_index]
        max_m_ref[token_index] = jnp.where(
          blockwise_topm_contains_topk & (max_m == k), m - 1, max_m
        )

      # If not all terminated, reset the flag say we need to search deeper
      @pl.when(flag_ref[0] != block_token)
      def _():
        flag_ref[0] = 0


# Pallas function
def blockwise_topk_pallas(logits, k, num_blocks=128, block_token=None, m_schedule=None):
  num_tokens, vocab_size = logits.shape
  if block_token is None:
    block_token = min(32, num_tokens)
  if num_tokens % block_token != 0:
    raise ValueError("token block size must be a multiple of num tokens")

  out_shape = (
    [jax.ShapeDtypeStruct((num_tokens, num_blocks), logits.dtype) for i in range(k)],
    # uint16 fits vocab size of up to 2**16 * 128 = 8.4M. But not used to avoid unforseen issues.
    [jax.ShapeDtypeStruct((num_tokens, num_blocks), jnp.int32) for i in range(k)],
    jax.ShapeDtypeStruct((num_tokens,), jnp.int32),
    jax.ShapeDtypeStruct((1,), jnp.int32),  # scratch for termination flag
  )
  out_specs = jax.tree.map(
    lambda _: pl.BlockSpec((block_token, num_blocks), lambda i: (i, 0)), out_shape[:2]
  )
  out_specs += (
    pl.BlockSpec(memory_space=pltpu.SMEM),
    pl.BlockSpec(memory_space=pltpu.SMEM),
  )
  return pl.pallas_call(
    functools.partial(
      topk_blockwise_superset_kernel,
      k=k,
      num_blocks=num_blocks,
      m_schedule=m_schedule,
    ),
    in_specs=(pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),),
    out_shape=out_shape,
    grid=(num_tokens // block_token),
    out_specs=out_specs,
    compiler_params=pltpu.TPUCompilerParams(vmem_limit_bytes=2**26),
  )(logits)


def topk_on_filtered_subset(topk_val, topk_index, k):
  num_blocks = topk_val[0].shape[-1]
  topk_logits, local_indices = jax.lax.top_k(jnp.concatenate(topk_val, axis=-1), k=k)

  @jax.vmap
  def unravel_indices(local_indices, topk_index):
    m, col = jnp.unravel_index(local_indices, (k, num_blocks))
    row = jnp.stack(topk_index)[m, col]
    flat_index = row * num_blocks + col
    return flat_index

  topk_flat_indices = unravel_indices(local_indices, topk_index)
  return topk_logits, topk_flat_indices


@functools.partial(
  jax.jit,
  static_argnames=(
    "k",
    "num_blocks",
    "m_stage1_schedule",
    "m_stage2_schedule",
    "block_token",
  ),
)
def topk_optimized(
  logits,
  k=64,
  num_blocks=128,
  m_stage1_schedule=None,
  m_stage2_schedule=None,
  block_token=None,
):
  topk_val, topk_index, termination_m, _ = blockwise_topk_pallas(
    logits,
    k=k,
    block_token=block_token,
    m_schedule=m_stage1_schedule,
    num_blocks=num_blocks,
  )

  # top-k the smallest number of values we can, by taking max m required
  # such that all queries to have full top-k
  # We compile for a range of shapes, then use jax.lax.cond to run just one.
  # in practice 8 nearly always sufficient
  if m_stage2_schedule is None:
    m_init = 8
    m_stage2_schedule = (
      [-1] + [m_init * (2**i) for i in range(int(math.log2(k // m_init)) + 1)] + [k]
    )

  # Buffer for output to be written in to
  topk_logits, topk_flat_indices = jax.tree.map(
    jnp.zeros_like, topk_on_filtered_subset(topk_val[:1], topk_index[:1], k=k)
  )
  max_m = termination_m.max()
  for lower_m, upper_m in zip(m_stage2_schedule, m_stage2_schedule[1:]):
    topk_logits, topk_flat_indices = jax.lax.cond(
      (max_m > lower_m) & (max_m <= upper_m),
      lambda *args: topk_on_filtered_subset(
        topk_val=topk_val[:upper_m], topk_index=topk_index[:upper_m], k=k
      ),
      lambda *args: args,
      topk_logits,
      topk_flat_indices,
    )
  return topk_logits, topk_flat_indices
