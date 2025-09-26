# !pip install tensorboard tensorboard-plugin-profile
# Pip install will require a restart, then comment the code


import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def block_topk(logits, k, topk_val=None, topk_index=None, start_k=0, num_lanes=128, mode='jax'):
  ntokens = logits.shape[0]

  if mode == 'jax':
    topk_val = [jnp.full((ntokens, num_lanes), float('-inf'), dtype=logits.dtype) for i in range(k)]
    # TODO?: Could use uint16 when vocab size < 4M if hardware supports
    topk_index = [jnp.full((ntokens, num_lanes), -1, dtype=jnp.int32) for i in range(k)]
  elif mode=='pallas':
    if topk_val is None or topk_index is None:
      raise ValueError("Pass through of topk_val and tok_index expected for pallas topk.")

  def while_body(i, while_carry):
    topk_val, topk_index = while_carry

    if mode == 'pallas':
      vals_carry = logits[..., pl.dslice(num_lanes*i, num_lanes)]
    elif mode == 'jax':
      vals_carry = jax.lax.dynamic_slice_in_dim(logits, num_lanes*i, num_lanes, axis=1)
    else:
      raise ValueError("mode must be either `pallas` and a memory ref or `jax` and an array")

    index_carry = jnp.full((ntokens, num_lanes), i, jnp.int32)

    for depth in range(k):
      if depth < start_k:
        # Nothing will be exchanged into the completed block topk, we just need
        # to invalidate it from flowing downward. So we check if it's already
        # found and invalidate if so.
        vals_carry = jnp.where(index_carry == topk_index[depth], float('-inf'), vals_carry)
      else:
        # Sinking sort
        mask = vals_carry > topk_val[depth]
        # TODO: Consider packing bfloat16 val and uint16 index into single uint32 and packed sort as in https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/topk_details/_topk_forward.py
        topk_val[depth], vals_carry = (
            jnp.where(m, vals_carry, topk_val[depth]) for m in (mask, ~mask))
        topk_index[depth], index_carry = (
            jnp.where(m, index_carry, topk_index[depth]) for m in (mask, ~mask))
    return (topk_val, topk_index)

  (topk_val, topk_index) = jax.lax.fori_loop(
      0,
      logits.shape[-1] // num_lanes,
      while_body,
      (topk_val, topk_index)
  )
  return topk_val, topk_index

# Pallas kernel scaffolding
def block_topk_kernel(logits_ref, topk_val_refs, topk_index_refs, max_depth_ref, flag_ref, num_lanes=128, k=64, depth_schedule=None):
  ### Initialize refs
  shape = topk_val_refs[0].shape
  for i in range(len(topk_val_refs)):
    topk_val_refs[i][...] = jnp.full(shape, float('-inf'), dtype=logits_ref.dtype)
    topk_index_refs[i][...] = jnp.full(shape, -1, dtype=jnp.int32)

  block_token = logits_ref.shape[0]
  for i in range(block_token):
    # Worst case depth
    max_depth_ref[pl.program_id(0) * block_token + i] = k

  # flag for termination of while loop
  flag_ref[0] = 0

  ### Run increasing block topk, until sure overall topk present
  if depth_schedule is None:
    depth_schedule = (0, 5, 8, 12, k)

  for completed_depth, depth in zip(depth_schedule, depth_schedule[1:]):
    @pl.when(flag_ref[0] == 0)
    def _():

      topk_vals, topk_indexs = block_topk(
          logits_ref,
          # bf16, bf16 -> i1 mask not supported on v5e so we cast to f32
          # TODO: check v6e, bf16 comparitor and make model specific
          topk_val=jax.tree.map(lambda ref: ref[...].astype(jnp.float32), topk_val_refs),
          topk_index=jax.tree.map(lambda ref: ref[...], topk_index_refs),
          k=depth,
          start_k=completed_depth,
          mode='pallas',
      )

      for i in range(completed_depth, depth):
        topk_val_refs[i][...] = topk_vals[i].astype(topk_val_refs[i].dtype)
        topk_index_refs[i][...] = topk_indexs[i].astype(topk_index_refs[i].dtype)

      # Stopping criterion check
      # To find top-k values of a set, we can split into N subsets,
      # and sort the largest, 2nd-largest, 3-rd largest, ..., m-th largest values for each subset
      # When in the superset of top-(m-1) subsets there are more than k values
      # larger (or equal than) the largest m'th largest value from the subsets
      # then the top-(m-1) subsets must contain the top-k of the set.
      # We run a schedule of m's until we have that full top-k found.
      pivot_point = topk_vals[depth-1].max(-1, keepdims=True)
      n_larger = sum(
          [(v >= pivot_point) for v in topk_vals[:depth-1]]
      ).astype(jnp.float32).sum(-1)
      # flag SMEM used to check if all searches terminated
      flag_ref[0] = 0
      for i in range(block_token):
        topk_all_present = n_larger[i] > k
        flag_ref[0] += topk_all_present
        # Store when the criteria was hit for each query
        token_index = pl.program_id(0) * block_token + i
        block_topk_depth = max_depth_ref[token_index]
        max_depth_ref[token_index] = jnp.where(
            topk_all_present & (block_topk_depth == k),
            depth - 1,
            block_topk_depth)

      # If not all terminated, reset the flag say we need to search deeper
      @pl.when(flag_ref[0] != block_token)
      def _():
        flag_ref[0] = 0

# Pallas function
def block_topk_pallas(logits, k, num_lanes=128, block_token=None, depth_schedule=None):
  num_tokens, vocab_size = logits.shape
  if block_token is None:
    block_token = min(32, num_tokens)
  if num_tokens % block_token != 0:
    raise ValueError('token block size must be a multiple of num tokens')

  out_shape = (
          [jax.ShapeDtypeStruct((num_tokens, num_lanes), logits.dtype) for i in range(k)],
          [jax.ShapeDtypeStruct((num_tokens, num_lanes), jnp.int32) for i in range(k)], # uint16 fits vocab size of up to 2**16 * 128 = 8.4M. But not used to avoid unforseen issues.
          jax.ShapeDtypeStruct((num_tokens,), jnp.int32), # block_topk required to be certain to contain topk vals
          jax.ShapeDtypeStruct((1,), jnp.int32), # scratch for stopping boolean
  )
  out_specs = jax.tree.map(
      lambda _: pl.BlockSpec((block_token, num_lanes), lambda i: (i, 0)),
      out_shape[:2]
  )
  out_specs += (
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(memory_space=pltpu.SMEM),
  )
  return pl.pallas_call(
      functools.partial(
          block_topk_kernel,
          k=k,
          num_lanes=num_lanes,
          depth_schedule=depth_schedule,
      ),
      in_specs=(
          pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),
      ),
      out_shape=out_shape,
      grid=(num_tokens // block_token),
      out_specs=out_specs,
      debug=False,
      compiler_params=pltpu.TPUCompilerParams(vmem_limit_bytes=2**26),
  )(logits)

def topk_on_filtered_subset(topk_val, topk_index, k):
  num_lanes = topk_val[0].shape[-1]
  topk_logits, local_indices = jax.lax.top_k(
      jnp.concatenate(topk_val, axis=-1),
      k=k
  )

  @jax.vmap
  def unravel_indices(local_indices, topk_index):
    depth, col = jnp.unravel_index(local_indices, (k, num_lanes))
    row = jnp.stack(topk_index)[depth, col]
    flat_index = row * num_lanes + col
    return flat_index

  topk_flat_indices = unravel_indices(local_indices, topk_index)
  return topk_logits, topk_flat_indices


@functools.partial(jax.jit, static_argnames=('k', 'base_cutoff', 'block_token', 'depth_schedule', 'num_lanes'))
def topk_optimized(logits, k=64, base_cutoff=8, block_token=None, depth_schedule=None, num_lanes=128):
  topk_val, topk_index, depths, _ = block_topk_pallas(logits, k=k, block_token=block_token, depth_schedule=depth_schedule, num_lanes=num_lanes)

  # top-k the smallest number of values we can, by taking max depth required
  # such that all queries in logits are guaranteed to have top-k
  # We compile for a range of shapes, then use jax.lax.cond to run just one.
  # in practice 8 nearly always sufficient
  cutoff_schedule = [-1]+[
      base_cutoff * (2**i) for i in range(int(math.log2(k // base_cutoff))+1)
  ] + [k]

  # Buffer for output to be written in to
  topk_logits, topk_flat_indices = jax.tree.map(
      jnp.zeros_like,
      topk_on_filtered_subset(topk_val[:1], topk_index[:1], k=k)
  )
  for min_cutoff, cutoff in zip(cutoff_schedule, cutoff_schedule[1:]):
    max_depth = depths.max()
    topk_logits, topk_flat_indices = jax.lax.cond(
        (max_depth > min_cutoff) & (max_depth <= cutoff),
        lambda *args: topk_on_filtered_subset(topk_val=topk_val[:cutoff], topk_index=topk_index[:cutoff], k=k),
        lambda *args: args,
        topk_logits, topk_flat_indices
    )
  return topk_logits , topk_flat_indices
