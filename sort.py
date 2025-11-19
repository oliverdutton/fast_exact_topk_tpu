#@title alu
"""
Optimized Bitonic Sort Implementation for TPU using JAX/Pallas.
"""
!pip install -U jax[tpu]
from google.colab import runtime
import functools
import itertools
from functools import lru_cache
import gzip
import json
import os
from glob import glob
from collections.abc import Callable, Sequence
import pandas as pd
import math

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
INTERPRET='TPU' not in jax.devices()[0].device_kind

NUM_SUBLANES = 8
NUM_LANES = 128

@lru_cache
def _log2(value: int) -> int:
  """Calculate log base 2 of an integer."""
  log_result = 0
  n = value
  while n > 1:
    n = n // 2
    log_result += 1
  return log_result

# JAX lowering for Pallas doesnt support integer unroll
def unrolled_fori_loop(length: int, body_fn, init_val, unroll: int):
  """Execute a for loop with manual unrolling for better performance."""
  unroll = min(length, unroll)

  def unrolled_body(i, carry):
    i *= unroll
    for j in range(unroll):
      carry = body_fn(i + j, carry)
    return carry

  carry = jax.lax.fori_loop(0, length // unroll, unrolled_body, init_val)
  for j in range(length % unroll):
    carry = body_fn((length // unroll) * unroll + j, carry)
  return carry

gather_sublane = lambda x, index: jax.lax.gather(x, index[...,None], jax.lax.GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,),start_index_map=(0,),operand_batching_dims=(1,),start_indices_batching_dims=(1,),), slice_sizes=(1,1))
gather_lane = jax.vmap(lambda x, index: x[index])

def _get_dtype_info(x):
  dtype=x.dtype
  if jnp.issubdtype(dtype, jnp.floating):
    info = jnp.finfo(x)
  elif jnp.issubdtype(dtype, jnp.integer):
    info = jnp.iinfo(x)
  else:
    raise ValueError('Only int and float supported')
  return info

def pack_value_with_index(val, index):
  """
  Pack bfloat16 value and int32 index into a single float32.
  This allows sorting while preserving original indices.
  """
  assert index.dtype == jnp.int32
  # BF16 values in F32 have empty lower 16 bits in mantissa where we pack the index
  return (
      val.astype(jnp.float32).view(jnp.int32) | index
      ).view(jnp.float32)


def unpack_value_and_index(packed):
  """Extract the original value and index from packed representation."""
  val = (packed.view(jnp.int32) & ~0xFFFF
      ).view(jnp.float32).astype(jnp.bfloat16)
  index = packed.view(jnp.int32) & 0xFFFF
  return val, index

def float_to_sortable_int(x: jnp.ndarray, standardize=True) -> jnp.ndarray:
    """
    Transforms float32 bits into a sortable int32 representation.

    Positive floats are mapped to [INT_MIN, -1].
    Negative floats are mapped to [INT_MAX, 0] (with order reversed).
    """
    if standardize:
      # nans, one less than extreme vals to ensure compatibility with padding
      info = jnp.iinfo(jnp.int32)
      nan_val = sortable_int_to_float(info.min+1 if descending else info.max-1)
      x = jnp.where(jnp.isnan(x), nan_val, x) # nan standardized
      x = jnp.where(x==0, 0, x) # +/-0 standardized
    i = x.view(jnp.int32)
    return jnp.where(i<0, i^0x7FFFFFFF, i)

def sortable_int_to_float(i: jnp.ndarray) -> jnp.ndarray:
    """
    Performs the inverse transformation from a sortable int32
    back to the original float32 bits.
    """
    return jnp.where(i<0, i^0x7FFFFFFF, i).view(jnp.float32)

# ============================================================================
# Tile Management
# ============================================================================

def split_array_to_tiles(arr):
  """Split a 2D array into a flat list of tiles."""
  num_rows, num_cols = arr.shape
  tile_rows = num_rows // NUM_SUBLANES
  tile_cols = num_cols // NUM_LANES

  tiles = []
  for row in range(tile_rows):
    for col in range(tile_cols):
      tile = arr[
          row * NUM_SUBLANES: (row + 1) * NUM_SUBLANES,
          col * NUM_LANES: (col + 1) * NUM_LANES,
      ]
      tiles.append(tile)
  return tiles


def join_tiles_to_array(target_shape, tiles):
  """Reconstruct a 2D array from a flat list of tiles."""
  num_rows, num_cols = target_shape
  tile_rows, tile_cols = tiles[0].shape
  grid_cols = num_cols // tile_cols

  rows = []
  for i in range(len(tiles) // grid_cols):
    row_tiles = tiles[i * grid_cols: (i + 1) * grid_cols]
    rows.append(jnp.concatenate(row_tiles, axis=-1))

  return jnp.concatenate(rows, axis=-2)


def iota_tile(dim):
  return lax.broadcasted_iota(jnp.int32, (NUM_SUBLANES, NUM_LANES), dim)

def create_bit_indicator(bit_position: int, index=None):
  """Create a boolean mask indicating which elements have a specific bit set.
  
  provides int format, so uses ALU rather than mask operations
  """
  if index is None:
    index = iota_tile(1) 
  if type(bit_position)==int:
    # returning bool
    bit = (index & (1 << bit_position))
    return bit > 0
  # returning int
  return (index >> bit_position) & 1

def compare(lefts, rights, is_descending: jax.Array | None, is_right_half=None,
num_keys=None, handle_nans=True, has_unique_key=False):
  
  num_arrs = len(lefts)
  if num_keys is None:
    # sort based on all keys
    num_keys = num_arrs

  def _compare(i, left, right):
    handle_subtile_ties = (
      is_right_half is not None
      and not has_unique_key and num_arrs!=num_keys and i==num_keys-1)
        
    if handle_subtile_ties:
      left, right = (
        jnp.where(is_right_half, right, left),
        jnp.where(is_right_half, left, right)
      )
            
    mask = left > right if type(is_descending)==bool and is_descending else right > left
    if jnp.issubdtype(dtype, jnp.floating) and handle_nans:
      
      
      
    mask = mask.astype(jnp.int32)
    
    if is_right_half is not None and not handle_subtile_ties:
      mask = jnp.bitwise_xor(mask, is_right_half.astype(jnp.int32))
    return mask

  masks = tuple(_compare(i, left, right) for i, (left, right) in enumerate(zip(lefts, rights, strict=True)))
  
  ties = [(
    (left==right) | (jnp.isnan(left) & jnp.isnan(right))
    if jnp.issubdtype(dtype, jnp.floating) and handle_nans else (left==right))
  for left, right in zip(lefts, rights, strict=True)]
  
  mask = masks[0]
  for k in range(1, num_keys):
    # break ties in primary key with secondary key comparison (and so on)
    # mask = mask | (ties[k-1] & masks[k])
    # alternative using i32 instead of i1
    mask = jnp.where(ties[k-1], masks[k], mask)
    ties[k] &= ties[k-1]
  
  if is_descending is not None and type(is_descending)!=bool:
    # can be done using i32 or i1 mask
    # chosen by optimizing runtime
    if num_arrs>1:
      mask = mask.astype(bool)
      is_descending = is_descending.astype(bool)  
    mask = mask ^ is_descending

  return jax.tree.map(
    lambda left, right: (
      # compares two tiles
      jnp.where(mask, left, right),
      jnp.where(mask, right, left), # equiv to ~swap, left, right
    ) if is_right_half is None else (
      # compares within a tile
      jnp.where(mask, left, right)
    ),
    lefts, rights
  )

def transpose_list_of_lists(tree):
  outer = jax.tree.structure(type(tree)('*')*len(tree))
  inner = jax.tree.structure(type(tree[0])('*')*len(tree[0]))
  return jax.tree.transpose(outer, inner, tree)

def compute_crosstile_substage(
    refs,
    substage: int,
    stage: int,
    unroll: int = 16,
    dim1_offset: int = 0,
    num_keys: int | None = None,
):
  """
  Perform a substage of sort involving comparisons between tiles

  Args:
      array_ref: Reference to array being sorted
      aux_ref: array to be sorted according to value in array_ref (e.g. array indices)
      substage: Current substage within the stage
      stage: Current sorting stage
      unroll: Loop unrolling factor
  """
  assert (unroll % 2) == 0, 'Static sort order requires even unroll factor'

  num_pairs = refs[0].shape[-1] // 2 ** (substage + 1)
  unroll = min(unroll, num_pairs)

  @pl.loop(0, pl.cdiv(num_pairs, unroll))
  def process_pairs(loop_idx):
    pair_length = 2 ** (substage + 1)
    slice_length = unroll * pair_length
    ref_slices = [ref.at[:, pl.dslice(loop_idx * slice_length, slice_length)] for ref in refs]

    outs = [[] for _ in refs]
    for i in range(unroll):
      pair_offset = (loop_idx * unroll + i) * pair_length
      half_length = 2 ** substage

      # Slice subarrays to be compared
      lefts  = [v[:, i * pair_length: i * pair_length + half_length] for v in ref_slices]
      rights = [v[:, i * pair_length + half_length: i * pair_length + 2 * half_length] for v in ref_slices]

      is_descending = create_bit_indicator(stage, dim1_offset + pair_offset)

      # Store the sorted pairs for the main array
      for i, vs in enumerate(compare(lefts, rights, is_descending=is_descending, num_keys=num_keys)):
        outs[i].extend(vs)

    # Concatenate all sorted pairs and write the entire slice back at once
    for ref_slice, out in zip(ref_slices, outs, strict=True):
      ref_slice[...] = jnp.concatenate(out, axis=-1)

def _pad(x, descending=False, f32_sorted_in_i32=True):
  # pad to multiple of NUM_SUBLANES in dim0
  # and power of two in final dim
  dim0, dim1 = x.shape
  pad_dim0 = pl.cdiv(dim0, NUM_SUBLANES) * NUM_SUBLANES
  pad_dim1 = max(2**math.ceil(math.log2(dim1)), NUM_LANES)
  if f32_sorted_in_i32 and jnp.issubdtype(x.dtype, jnp.floating):
    # we'll sort in i32, both max and min i32 transform to nans in f32 
    v = jnp.iinfo(jnp.int32)
    pad_val = float_to_sortable_int(
      jnp.array(v.min if descending else v.max, jnp.int32), standardize=False)
  else:
    info = _get_dtype_info(x)
    pad_val = info.min if descending else info.max
  return jnp.pad(x, ((0, pad_dim0-dim0),(0, pad_dim1-dim1)), mode='constant',
  constant_values=pad_val)

def convert_to_sublane_sort_format(arr):
  arrs = [arr[:, i*NUM_LANES:(i+1)*NUM_LANES]
  for i in range(pl.cdiv(arr.shape[1], NUM_LANES))]
  arr = jnp.concatenate(arrs, axis=0).T # (128,n*b)
  if arr.shape[1] < NUM_LANES:
    arr = _pad(arr)
  tiles = split_array_to_tiles(arr) # 16 * (n*b//128) (8, 128)
  return tiles

def convert_from_sublane_sort_format(tiles, shape):   
  b, m = shape
  assert m >= NUM_LANES
  n = m // NUM_LANES
  dim1 = len(tiles)*NUM_SUBLANES
  arr = join_tiles_to_array((NUM_LANES, dim1), tiles) #(128,n*b)
  if dim1 != n*b:
    # unpad
    arr = arr[...,:n*b]
  arr = arr.T
  return jnp.concatenate(
      [arr[i*b:(i+1)*b] for i in range(arr.shape[0] // b)], axis=1)

def compute_substage_by_permute(substage, arrs_tiles, *, stage, permute_dim, dim1_offset, num_keys: int, b:int):
  if permute_dim==0: # sublane
    assert b is not None
    index = iota_tile(0)
    global_base_index = iota_tile(0) + ((iota_tile(1) // b) * NUM_LANES)
    tile_rows = NUM_LANES // NUM_SUBLANES
    tile_cols = len(arrs_tiles[0]) // tile_rows
  elif permute_dim==1: #lane
    index = global_base_index = iota_tile(1)
    tile_rows = b // NUM_SUBLANES
    tile_cols = len(arrs_tiles[0]) // tile_rows
  else:
    raise ValueError('dim must be 0 or 1, (sublane or lane)')
  is_right_half = create_bit_indicator(substage, index)

  permutation = jnp.bitwise_xor(index, 1 << substage)
  arrs_tiles_permuted = jax.tree.map(lambda tile: (
    gather_sublane(tile, permutation) if permute_dim==0 else gather_lane(tile, permutation)), arrs_tiles)

  outs_tiles = [[] for _ in arrs_tiles]

  for tile_idx, (lefts, rights) in enumerate(zip(*map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)), strict=True)):
    if permute_dim==0:
      tile_row = tile_idx // tile_cols
      tile_col = tile_idx % tile_cols
      tile_offset = tile_row * NUM_SUBLANES + tile_col * (NUM_LANES * (NUM_LANES // b))
    else: #1, lane
      tile_offset = (tile_idx % tile_cols) * NUM_LANES
    
    is_descending = create_bit_indicator(stage,
      dim1_offset + tile_offset + global_base_index, )   
    if type(stage)==int:
      # performance optimizations for early, staticly compiled, stages
      if stage < _log2(NUM_SUBLANES): #1,2
        # bitonic sort, as stage is small cycle is intrasubtile
        is_descending = create_bit_indicator(stage, global_base_index)
      elif stage < _log2(NUM_LANES): #3,4,5,6
        # single value per tile as all vals in tile are multiples of NUM_LANES
        is_descending = create_bit_indicator(stage, tile_offset)

    for i, o in enumerate(compare(lefts, rights, is_descending=is_descending, is_right_half=is_right_half, num_keys=num_keys)):
        outs_tiles[i].append(o)

  return outs_tiles

def compute_start_index(i, separation, slice_length=1):
  # when theres pairs if arrays of certain lengths which are subsliced
  if slice_length > separation:
    raise ValueError(f'Separation must be at least slice length, {separation=} {slice_length=}')
  slices_per_pair = separation // slice_length
  pair_idx = i // slices_per_pair
  slice_idx = i % slices_per_pair
  return pair_idx * 2 * separation + slice_idx * slice_length

def compute_substage_by_crosstile_comparison(arrs_tiles, substage, b, dim1_offset=0, stage=None, num_keys: int | None = None):
  #return
  global_base_index = iota_tile(0) + ((iota_tile(1) // b) * NUM_LANES)
  num_tiles = len(arrs_tiles[0])
  tile_rows = NUM_LANES // NUM_SUBLANES
  tile_cols = num_tiles // tile_rows

  # compare explicit entries from lists of tiles
  separation = (2**substage // NUM_SUBLANES) * tile_cols
  outs_tiles = [[None for _ in t] for t in arrs_tiles]
  for i in range(num_tiles // 2):
    idx = compute_start_index(i, separation=separation)

    tile_row = idx // tile_cols
    tile_col = idx % tile_cols
    pair_offset = tile_row * NUM_SUBLANES + tile_col * (NUM_LANES * (NUM_LANES // b))
    lefts, rights = (transpose_list_of_lists(arrs_tiles)[j] for j in (idx, idx+separation))
    
    is_descending = create_bit_indicator(stage, 
      dim1_offset + pair_offset + global_base_index, )
    
    if type(stage)==int and stage<_log2(NUM_LANES):
      # stage 4,5,6
      is_descending = create_bit_indicator(stage, pair_offset)
    # Store the sorted pairs for the main array
    for i, (o_left, o_right) in enumerate(compare(lefts, rights, 
    is_descending=is_descending, num_keys=num_keys)):
      outs_tiles[i][idx] = o_left
      outs_tiles[i][idx+separation] = o_right

  assert all(
    not any([v is None for v in out_tiles])
    for out_tiles in outs_tiles)
  return outs_tiles

def _compute_subtile_substages(
    arrs_tiles,
    num_substages: int,
    stage: int,
    b:int,
    use_lane_permute: bool,
    dim1_offset: int = 0,
    num_keys: int | None = None,
):
  """Execute multiple substages of bitonic sort where compared values are from the same tile."""
  assert num_substages <= _log2(NUM_LANES)

  for substage in range(num_substages)[::-1]:    
    if use_lane_permute:
      arrs_tiles = compute_substage_by_permute(substage, arrs_tiles,
      stage=stage, permute_dim=1, # lane permute
      b=b, dim1_offset=dim1_offset, num_keys=num_keys)
  
    # from the (b,n*128) array, its transposed to (128, b*n) and split into (8, 128) tiles
    # allowing comparison between full tiles or using sublane permutes insteas of lane permutes which is much faster
    elif substage >= _log2(NUM_SUBLANES):
      # inter tile (64,32,16,8)
      arrs_tiles = compute_substage_by_crosstile_comparison(arrs_tiles,
      substage=substage, b=b, dim1_offset=dim1_offset,
      stage=stage, num_keys=num_keys)
    else:      
      # intra tile (4,2,1)
      arrs_tiles = compute_substage_by_permute(substage, arrs_tiles,
      stage=stage, permute_dim=0, # sublane permute
      b=b, dim1_offset=dim1_offset, num_keys=num_keys)
  return arrs_tiles


def compute_subtile_substages(
    refs,
    *,
    num_substages: int,
    stage: int,
    num_keys: int,
    unroll: int = 256,
    dim1_offset: int = 0,
    slice_dim1: int = None,
):
  """Orchestrate subtile sorting operations with proper blocking."""
  shape = refs[0].shape
  if slice_dim1 is None:
    slice_dim1 = min(unroll * NUM_LANES, shape[1])

  unroll_dim0 = (unroll * NUM_LANES) // slice_dim1
  slice_dim0 = min(unroll_dim0 * NUM_SUBLANES, shape[0])
  unroll = (slice_dim0 * slice_dim1) // (NUM_SUBLANES * NUM_LANES)

  grid_dim0 = shape[0] // slice_dim0
  grid_dim1 = shape[1] // slice_dim1

  @pl.loop(0, grid_dim0 * grid_dim1)
  def process_block(loop_idx):
    block_row = loop_idx // grid_dim1
    block_col = loop_idx % grid_dim1

    ref_slices = [ref.at[
        pl.dslice(block_row * slice_dim0, slice_dim0),
        pl.dslice(block_col * slice_dim1, slice_dim1)
    ] for ref in refs]

    slice_shape = ref_slices[0].shape
    b = slice_shape[0]

    # benchmarking showed transpose then sublane permutes always faster than lane permutes
    use_lane_permute = False

    arrs_tiles = jax.tree.map(
      split_array_to_tiles if use_lane_permute else
      convert_to_sublane_sort_format, ref_slices)

    if stage is not None:
      # run single stage
      arrs_tiles = _compute_subtile_substages(
          arrs_tiles,
          num_substages=num_substages,
          stage=stage,
          dim1_offset=dim1_offset + (block_col * slice_dim1),
          b=b,
          num_keys=num_keys,
          use_lane_permute=use_lane_permute,
      )
    else:
      # special mode, where we
      # run all stages 1 to num_substages.
      # this allows compiler to fuse stages
      num_stages = num_substages
      for stage_ in range(1, num_stages+1):
        arrs_tiles = _compute_subtile_substages(
            arrs_tiles,
            num_substages=stage_,
            stage=stage_,
            dim1_offset=dim1_offset + (block_col * slice_dim1),
            b=b,
            num_keys=num_keys,
            use_lane_permute=use_lane_permute,
        )
        
        
        outs = [
        join_tiles_to_array(slice_shape, tiles) if use_lane_permute else
        convert_from_sublane_sort_format(tiles, shape=slice_shape) for tiles in arrs_tiles]
        if INTERPRET:
          l = 2**stage_
          jax.debug.print(f'{stage_=}, ''{}', 
          [jnp.argsort(outs[0][0,i*l:(i+1)*l]) for i in range(4)])

    outs = [
    join_tiles_to_array(slice_shape, tiles) if use_lane_permute else
    convert_from_sublane_sort_format(tiles, shape=slice_shape) for tiles in arrs_tiles]
    
    for ref_slice, out in zip(ref_slices, outs, strict=True):
      ref_slice[...] = out
  
       
def _max_int(a,b):
  '''Max of two int values, accepts dynamic and static ints.'''
  if not all(map(lambda v: type(v) == int, (a,b))):
    return jnp.maximum(a,b)
  return max(a,b)

def _all_concrete_ints(*args):
  return all(map(lambda v: type(v) == int, args))

def compute_stages(
    start_stage: int,
    end_stage: int,
    refs,
    unroll_crosstile: int = 128,
    unroll_subtile: int = 128,
    dim1_offset: int = 0,
    num_keys: int | None = None,
    start_stage_static_lower_bound: int | None = None
):
  """Execute a range of bitonic sorting stages."""
  log_n = _log2(refs[0].shape[1])

  # to allow for dynamic val of start_stage and compiler fusion of stages 1-7
  # we support providing a lower bound for start_stage
  if start_stage_static_lower_bound is None:
    start_stage_static_lower_bound = start_stage

  # run stages 1 to 7 (if large enough), compiler fused
  if start_stage_static_lower_bound==1:
    compute_subtile_substages(
      refs,
      num_substages=min(_log2(NUM_LANES), end_stage),
      stage=None,
      dim1_offset=dim1_offset,
      unroll=unroll_subtile,
      num_keys=num_keys,
    )
  elif (_all_concrete_ints(start_stage, end_stage)
    and start_stage<=_log2(NUM_LANES) and end_stage==start_stage+1):
    compute_subtile_substages(
      refs,
      num_substages=start_stage,
      stage=start_stage,
      dim1_offset=dim1_offset,
      unroll=unroll_subtile,
      num_keys=num_keys,
    )
    return
  else:
    assert start_stage_static_lower_bound > _log2(NUM_LANES), 'stages 1 to _log2(NUM_LANES) only triggered as fully unrolled code block'

  # run stages 8 and upwards
  @pl.loop(_max_int(start_stage, _log2(NUM_LANES)+1), end_stage)
  def run_stage(stage):
    for substage in range(_log2(NUM_LANES), log_n)[::-1]:
      @pl.when(stage > substage)
      def _():
        compute_crosstile_substage(
          refs,
          substage=substage,
          stage=stage,
          unroll=unroll_crosstile,
          dim1_offset=dim1_offset,
          num_keys=num_keys,
        )
    # run subtile comparisons
    compute_subtile_substages(
      refs,
      num_substages=_log2(NUM_LANES),
      stage=stage,
      dim1_offset=dim1_offset,
      unroll=unroll_subtile,
      num_keys=num_keys,
    )


def bitonic_sort(
    refs,
    stage_ref,
    k: int = None,
    descending: bool = False,
    num_keys: int | None = None,
    log_n: int | None = None,
):
  """Core bitonic sort implementation."""
  shape = refs[0].shape
  assert len(shape)==2

  if k is None:
    k = shape[1]
  if log_n is None:
    log_n = _log2(shape[1])
  if 2**_log2(shape[1]) != shape[1]:
    raise ValueError("Size along sort dimension must be a power of 2")

  log_k = _log2(k)
  if 2**log_k != k:
    raise ValueError("k must be a power of 2")

  if num_keys is None:
    num_keys = len(refs)

  # to keep track of global index for bitonic sort order (if array is being subsort)
  # the second term controls whether final stage is descending or ascending
  dim1_offset = pl.program_id(1) * shape[1] + int(descending)*pl.num_programs(1) * shape[1]
  if stage_ref is None:
    # Execute bitonic sort of refs
    compute_stages(
      1, log_n + 1, refs,
      num_keys=num_keys,
      dim1_offset=dim1_offset,
    )
  else:
    # Run a single stage
    stage = stage_ref[0]
    compute_stages(
      stage, stage + 1,
      refs,
      dim1_offset=dim1_offset,
      # code only used for stages that do not fit in VMEM
      start_stage_static_lower_bound=log_n,
    )


def is_32bit(x):
  return x.dtype.itemsize == 4

def to_32bit_dtype(operand_dtype):
 for dtype_class, dtype_32bit in {jnp.floating: jnp.float32,
 jnp.integer: jnp.int32,
 jnp.bool: jnp.int32}.items():
   if jnp.issubdtype(operand_dtype, dtype_class):
     return dtype_32bit
 raise ValueError('dtype not recognized')

def same_shape_dtype(ref1, ref2):
  return (ref1.dtype == ref2.dtype) and (ref1.shape==ref2.shape)

def sort_kernel(
    in_refs,
    stage_ref,
    out_refs,
    refs, # scratch, refs operated on
    indices_ref,
    *,
    descending: bool,
    is_stable: bool = False,
    num_keys: int | None = None,
    log_n: int | None = None,
):
  """Pallas kernel for sorting."""
  shape = in_refs[0].shape
  k = out_refs[0].shape[-1]
  if num_keys is None:
    num_keys = len(in_refs)

  return_argsort = len(out_refs) > len(in_refs)
  assert len(out_refs) == (len(in_refs) + int(return_argsort))

  use_indices = is_stable or return_argsort
  indices = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

  pack_bf16_and_aux_in_f32 = not is_stable and  in_refs[0].dtype==jnp.bfloat16 and (
    (len(in_refs)==1 and return_argsort and shape[1] < 2**16) or
    (len(in_refs)==2 and in_refs[1].dtype==jnp.uint16 and num_keys==1 and not return_argsort)
  )
  if pack_bf16_and_aux_in_f32:
    # we can pack bf16 val and uint16 values into f32 for a fast sort without affecting (not stable) comparison. NaN handling undefined.
    # TODO: Unpack by cast and bitmasks rather than astype
    aux = indices if len(in_refs)==1 else in_refs[1][...].astype(jnp.int32)
    refs[0][...] = pack_value_with_index(in_refs[0][...], aux)
    bitonic_sort(
    refs,
    stage_ref,
    k=max(k, NUM_LANES),
    descending=descending,
    num_keys=num_keys,
    log_n=log_n)
    outs = unpack_value_and_index(refs[0][...])
    for ref, out in zip(out_refs, outs, strict=True):
      ref[...] = out[...,:k].astype(ref.dtype)
    return

  if descending and is_stable:
    # order must be maintained, so the indices key must be sorted ascending while other keys descending, so we flip the sign on indices to sort, then flip back before write out
    indices = indices.shape[1]-indices

  # try and reuse in/out VMEM which is already allocated, then the scoped VMEM is DCE'd and VMEM usage reduced
  for i in range(len(in_refs)):
    # try reuse buffer
    if same_shape_dtype(in_refs[i], refs[i]):
      refs[i] = in_refs[i]
    else:
      refs[i][...] = in_refs[i][...].astype(refs[i].dtype)

  if use_indices:
    # try reuse buffer
    if same_shape_dtype(indices_ref, out_refs[-1]):
      indices_ref = out_refs[-1]
    indices_ref[...] = indices
    refs.insert(num_keys, indices_ref)

  bitonic_sort(
    refs,
    stage_ref,
    k=max(k, NUM_LANES),
    descending=descending,
    # indices in array break ties, so sort is stable
    num_keys=num_keys+int(is_stable),
    log_n=log_n,
    )

  if use_indices:
    refs.pop(num_keys)
  if return_argsort:
    if descending and is_stable:
      # flip back the ascending indices on ties trick for stable sort
      indices_ref[...] = indices_ref.shape[1]-indices_ref[...]
    # move indices ref to the end so it matches out refs ordering
    refs.append(indices_ref)

  for ref, out_ref in zip(refs, out_refs, strict=True):
    # save a copy if compute was done in the output buffer
    if ref is not out_ref:
      out_ref[...] = ref[..., :k].astype(out_ref.dtype)

def canonicalize_operand(operand):
  operands = jax.tree.leaves(operand)
  shapes = [x.shape for x in operands]
  if len(set(shapes))!=1:
    raise ValueError(f'Inputs must all have the same shape, but found {shapes=}')
  shape = shapes[0]
  if len(shape)!=2:
    raise ValueError('Only 2D inputs supported')
  return operands, shape

@functools.partial(
    jit,
    static_argnames=("k", "block_token", "block_seq", "return_argsort",
    "descending", "num_keys", "is_stable", "log_n")
)
def _sort_pallas_vmem(
    operand: jax.Array | Sequence[jax.Array],
    k: int | None=None,
    block_token: int | None=None,
    block_seq: int | None=None,
    return_argsort: bool=False,
    descending: bool=False,
    num_keys: int | None = None,
    is_stable: bool = False,
    stage: int | None = None,
    log_n: int | None = None,
) -> tuple[jax.Array, ...]:
  """
  High-level interface for Pallas-based sorting on TPU.

  Args:
      x: Input array to sort (2D)
      k: Return only the first k elements from sorted arrays
      block_size: Token blocking size for memory efficiency
      return_argsort: Whether to return argsort of operand, if so this will be returned last
      descending: Sort in descending order
      log_n: length of sorted axis if array is padded
  """

  operands, shape = canonicalize_operand(operand)

  if k is None:
    k = shape[-1]
  if block_token is None:
    block_token = min(max(NUM_SUBLANES, (2**14) // shape[0]), shape[0])
  if block_seq is None:
    block_seq = shape[1]
  if k!=shape[1] and block_seq!=shape[1]:
    raise ValueError('k is not compatible with subsorting')
  block_shape = (block_token, block_seq)

  out_shapes = jax.tree.map(
    lambda v: jax.ShapeDtypeStruct((shape[0], k), v.dtype),
    tuple(operands)
  )
  if return_argsort:
    out_shapes += (
      jax.ShapeDtypeStruct((shape[0], k), jnp.int32),)

  in_specs = (
    [pl.BlockSpec(block_shape, lambda i,j: (i, j)) for _ in operands],
    pl.BlockSpec(memory_space=pltpu.SMEM) if stage is not None else None
  )
  out_specs = tuple(pl.BlockSpec((block_token, min(k, block_seq)),
  lambda i,j: (i, j)) for _ in out_shapes)

  scratch_shapes=(
    [pltpu.VMEM(block_shape, to_32bit_dtype(ref.dtype)) for ref in operands],
    pltpu.VMEM(block_shape, jnp.int32),
  )
  if stage is not None:
    stage = stage[None]

  return pl.pallas_call(
      functools.partial(sort_kernel, descending=descending, num_keys=num_keys,
      is_stable=is_stable, log_n=log_n),
      out_shape=(out_shapes,),
      in_specs=in_specs,
      out_specs=(out_specs,),
      scratch_shapes=scratch_shapes,
      grid=(shape[0] // block_token, shape[1] // block_seq),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=int(0.9 * 2**27),
          #allow_input_fusion=(True,True),
          ),
      interpret=INTERPRET,
  )(operands, stage)[0]


class AsyncCopyAggregator:
  """Bundles multiple async copy operations as a single copy operation."""

  def __init__(self, copy_descriptors):
    self.copy_descriptors = tuple(copy_descriptors)

  def wait(self):
    """Wait for all copy operations to complete."""
    for descriptor in self.copy_descriptors:
      descriptor.wait()


def _substage_hbm_kernel(
    input_hbm_refs,
    substage_ref,
    stage_ref,
    output_hbm_refs,
    input_semaphores,
    output_semaphores,
    input_vmem_refs,
    scratch_vmem_refs, # actually operated on
    output_vmem_refs,
    *,
    num_keys: int,
):
  """Kernel for running a substage which do not fit in VMEM."""
  # Handle sublane dimension indexing
  sublane_block = input_vmem_refs[0].shape[-2]
  sublane_slice = pl.dslice(pl.program_id(0) * sublane_block, sublane_block)
  input_hbm_refs, output_hbm_refs = jax.tree.map(
    lambda ref: ref.at[sublane_slice], (input_hbm_refs, output_hbm_refs))

  substage = substage_ref[0]
  stage = stage_ref[0]
  slice_length = input_vmem_refs[0].shape[-1]
  pair_length = 2 ** (substage + 1)
  slices_per_pair = (pair_length // 2) // slice_length

  def compute_start_index(i):
    pair_idx = i // slices_per_pair
    pair_subslice_idx = i % slices_per_pair
    return pair_idx * pair_length + pair_subslice_idx * slice_length

  def perform_dma(i, is_load):
    """Perform DMA operation (load or store)."""
    buffer_slot = lax.rem(i, 2)
    left_start = compute_start_index(i)
    right_start = left_start + (pair_length // 2)
    sems = input_semaphores if is_load else output_semaphores
    copies = []
    for i_ref, (hbm_ref, vmem_ref) in enumerate(zip(*(
        (input_hbm_refs, input_vmem_refs) if is_load else (
          output_hbm_refs, output_vmem_refs)),
        strict=True
      )):
        for vmem_slot, start in enumerate((left_start, right_start)):
          # Compiler fails to recognize start indices are multiples of num_lanes, so we tell the compiler explicitly
          start = pl.multiple_of(start, NUM_LANES)
          hbm_ref_slice = hbm_ref.at[:, pl.dslice(start, slice_length)]
          vmem_ref_slice = vmem_ref.at[buffer_slot, vmem_slot]
          sem = sems.at[buffer_slot, vmem_slot, i_ref]
          src, dst = (hbm_ref_slice, vmem_ref_slice) if is_load else (vmem_ref_slice, hbm_ref_slice)
          copies.append(
            pltpu.async_copy(
              src_ref=src,
              dst_ref=dst,
              sem=sem,
          ))
    return AsyncCopyAggregator(copies)

  load_dma = functools.partial(perform_dma, is_load=True)
  store_dma = functools.partial(perform_dma, is_load=False)

  def compute(loop_idx):
    """Perform comparison and swap logic."""
    start_idx = compute_start_index(loop_idx)
    slot = lax.rem(loop_idx, 2)

    refs = []
    for input_ref, scratch_ref in zip(input_vmem_refs, scratch_vmem_refs):
      if same_shape_dtype(input_ref, scratch_ref):
        refs.append(tuple(input_ref[slot]))
      else:
        scratch_ref[slot] = input_ref[slot].astype(scratch_ref.dtype)
        refs.append(tuple(scratch_ref[slot]))

    is_descending = create_bit_indicator(stage, start_idx)
    outputs = compare(
      *transpose_list_of_lists(refs), is_descending=is_descending, num_keys=num_keys)
    for (output_ref, (o_left, o_right)) in zip(output_vmem_refs, outputs):
      output_ref[slot, 0] = o_left.astype(output_ref.dtype)
      output_ref[slot, 1] = o_right.astype(output_ref.dtype)

  num_iterations = input_hbm_refs[0].shape[-1] // (2 * slice_length)
  assert num_iterations > 0

  # Pipeline: Load -> Compute -> Store
  initial_load = load_dma(0)
  if num_iterations > 1:
    next_load = load_dma(1)

  initial_load.wait()
  compute(0)

  if num_iterations == 1:
    store_dma(0).wait()
    return

  next_load.wait()

  @pl.loop(1, num_iterations - 1)
  def pipeline_iteration(loop_idx):
    store_op = store_dma(loop_idx - 1)
    load_op = load_dma(loop_idx + 1)
    compute(loop_idx)
    store_op.wait()
    load_op.wait()

  store_op = store_dma(num_iterations - 2)
  compute(num_iterations - 1)
  store_op.wait()
  store_dma(num_iterations - 1).wait()


@functools.partial(
    jax.jit,
    static_argnames=('block_shape', 'num_keys')
)
def compute_substage_hbm(
    operand,
    substage: int,
    stage: int,
    num_keys:int,
    block_shape=None,
):
  """Runs a substage without loading the full lane dimension into VMEM."""
  operands, shape = canonicalize_operand(operand)
  if block_shape is None:
    block_shape = (NUM_SUBLANES, 2**(16-_log2(len(operands))))

  #checkify.check(substage >= LOG_LANES, 'Intra tile comparisons not supported')
  #slice_length = block_shape[-1]
  #checkify.check(slice_length <= 2**substage, 'invalid slice length, sections of length {} (2**substage) sliced into chunks of size {}', 2**substage, slice_length)
  #checkify.check(substage < stage, 'substage greater than stage is not valid, substage={}, stage={}', substage, stage)

  # HBM-VMEM transfers handled manually as loading and storing two blocks from the same array (inplace) is not expressible in BlockSpecs
  input_specs = (
      [pl.BlockSpec(memory_space=pltpu.ANY) for _ in operands],
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(memory_space=pltpu.SMEM),
  )

  output_shape = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tuple(operands))
  num_refs = len(operands)
  input_vmems = jax.tree.map(lambda x: pltpu.VMEM((2, 2, *block_shape), x.dtype), operands)
  scratch_vmems = jax.tree.map(lambda x: pltpu.VMEM((2, 2, *block_shape), to_32bit_dtype(x.dtype)), operands)

  return pl.pallas_call(
      functools.partial(
          _substage_hbm_kernel,
          num_keys=num_keys,
      ),
      # indexing in outer loop over sublane dimension is handled inside the kernel, as pltpu.ANY memory space doesnt support block specs
      grid=(operands[0].shape[0] // block_shape[0],),
      out_shape=(output_shape,),
      in_specs=input_specs,
      out_specs=(tuple(input_specs[0]),),
      # (2,2) = (slot, left/right array for swap)
      scratch_shapes=(
          pltpu.SemaphoreType.DMA((2, 2, num_refs)),
          pltpu.SemaphoreType.DMA((2, 2, num_refs)),
          input_vmems,
          scratch_vmems,
          input_vmems, # output_vmems
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=int(0.9 * 2**27)),
      interpret=INTERPRET,
  )(operands, substage[None], stage[None])[0]


@functools.partial(jax.jit, static_argnames=('k', 'num_vmem_substages', 'descending', 'return_argsort', 'is_stable', 'num_keys', 'block_token', 'contains_nans'))
def lax_sort_pallas(
    operand: jax.Array | Sequence[jax.Array],
    num_keys: int = 1,
    is_stable: bool = False,
    k: int | None=None,
    return_argsort: bool=False,
    descending: bool=False,
    num_vmem_substages: int | None=None,
    block_token: int | None=None,
    contains_nans: bool = True,
) -> tuple[jax.Array, ...]:
  """
  Sort large arrays using a hybrid HBM-VMEM approach.

  This function handles arrays larger than VMEM by breaking them into
  subsections, sorting in VMEM, then merging with HBM-based operations.

  Args:
      x: Input array to sort
      num_vmem_substages: log2 of max size that fits in VMEM (default: 2^19)
      descending: Sort in descending order
  """
  operands, shape = canonicalize_operand(operand)
  dtypes = [x.dtype for x in operands]
  if return_argsort:
    dtypes.append(jnp.int32)
  num_stages = _log2(shape[1])

  operands = [_pad(x, descending=descending) for x in operands]
  is_padded = shape != operands[0].shape
  if is_padded and not all(jnp.issubdtype(dtype, jnp.floating) for dtype in dtypes[:num_keys]):
    # must be stable to avoid padding entries leaking into the array
    is_stable = True

  use_indices = return_argsort or is_stable

  operands = [float_to_sortable_int(x) 
    if jnp.issubdtype(x.dtype, jnp.floating) and i<num_keys else x 
    for i,x in enumerate(operands)]
  
  if num_vmem_substages is None:
    # heuristic to fit 128MB VMEM
    num_vmem_substages = 18 - math.ceil(math.log2(
      len(operands)+int(use_indices)+sum(not is_32bit(x) for x in operands)*0.5
    ))
    #print(f'{num_vmem_substages=}')    

  # If array fits in VMEM, use simple sort
  if num_stages <= num_vmem_substages:
    operands = _sort_pallas_vmem(operands,
    descending=descending, num_keys=num_keys, is_stable=is_stable,
    return_argsort=return_argsort, k=k, block_token=block_token,
    log_n=math.ceil(math.log2(shape[1])))

    operands = [sortable_int_to_float(x)
    if jnp.issubdtype(dtype, jnp.floating) and jnp.issubdtype(x.dtype, jnp.integer)
    else x for x, dtype in zip(operands, dtypes)]
    # unpad
    return tuple(x[:shape[0], :shape[1]] for x in operands)

  if use_indices:
    indices = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    if descending and is_stable:
      # order must be maintained, so the indices key must be sorted ascending while other keys descending, so we flip the sign on indices to sort, then flip back before write out
      indices = indices.shape[1]-indices
    indices_index = num_keys
    operands.insert(num_keys, indices)
    num_keys += 1

  def _run_stage(stage, operands):
    """Execute a complete sorting stage. Mixing HBM and VMEM implementations"""
    def _compute_substages_hbm_body(i, operands):
      substage = stage - 1 - i
      return compute_substage_hbm(
          operands, substage, stage, num_keys=num_keys
      )

    # First: HBM-based substages for cross-VMEM-block operations
    operands = jax.lax.fori_loop(0, stage - num_vmem_substages, _compute_substages_hbm_body, operands)

    # Then: VMEM-based substages for within-block operations
    return _sort_pallas_vmem(operands, block_seq=2**num_vmem_substages,
    stage=stage, descending=descending, num_keys=num_keys, is_stable=False)

  # Initial bitonic sorting of VMEM-sized blocks up to VMEM-sized subsequences
  operands = _sort_pallas_vmem(tuple(operands), block_seq=2**num_vmem_substages,
    stage=None, descending=descending, num_keys=num_keys, is_stable=False)

  # Merge blocks through successive stages
  operands = jax.lax.fori_loop(
      num_vmem_substages, num_stages+1,
      _run_stage, operands
  )

  operands = list(operands)
  if use_indices:
    indices = operands.pop(indices_index)
  if return_argsort:
    if descending and is_stable:
      # flip back the ascending indices on ties trick for stable sort
      indices = indices.shape[1]-indices
    # move aindices ref to the end so it matches out refs ordering
    operands.append(indices)
  if k is not None:
    operands = [x[...,:k] for x in operands]    
  
  operands = [sortable_int_to_float(x)
    if jnp.issubdtype(dtype, jnp.floating) and jnp.issubdtype(x.dtype, jnp.integer)
    else x for x, dtype in zip(operands, dtypes)]
  # unpad
  operands = [x[:shape[0], :shape[1]] for x in operands]
  return tuple(operands)


topk_xla = jax.jit(jax.lax.top_k, static_argnames=("k",))
approx_topk_xla = jax.jit(jax.lax.approx_max_k, static_argnames=("k",))
sort_xla = jax.jit(jnp.sort, static_argnames=('stable',))
argsort_xla = jax.jit(jnp.argsort, static_argnames=('stable',))

lax_sort_xla = jax.jit(jax.lax.sort, static_argnames=( 'is_stable', 'num_keys'))

@jax.jit
def add_one(x):
  return x+1

def benchmark(_run):
  def run():
    return jax.block_until_ready(_run())
  run()
  with jax.profiler.trace("/content/"):
    run()

  path = sorted(glob("/content/plugins/profile/*/**.json.gz"), key=os.path.getmtime)[-1]
  trace = json.load(gzip.open(path))
  df = pd.DataFrame(trace["traceEvents"])
  df = df[~df.name.isna()]
  df['name'] = df.name.apply(lambda s: s.split('(')[0])
  print(df[df.name.str.contains("jit_")][['name', 'dur']].to_string(index=False))


@jax.jit
def exact_match(xs, ys):
  _all = lambda equality_op: jnp.array(jax.tree.leaves(jax.tree.map(lambda x,y: equality_op(x,y).all(), xs, ys))).all()
  nans_match = _all(lambda x, y: jnp.isnan(x)==jnp.isnan(y))
  non_nans_match = _all(lambda x,y: jnp.where(jnp.isnan(x), True, x==y))
  return nans_match & non_nans_match

def _match_descending(x):
    x = x[...,::-1]
    # moves lhs nans to rhs
    return jax.vmap(jnp.roll)(x, -jnp.isnan(x).sum(1))

@functools.partial(jax.jit, static_argnames=('k', 'num_vmem_substages', 'descending', 'return_argsort', 'is_stable', 'num_keys', 'block_token', 'contains_nans'))
def equiv_xla_based_sort(
    operand: jax.Array | Sequence[jax.Array],
    num_keys: int = 1,
    is_stable: bool = False,
    k: int | None=None,
    return_argsort: bool=False,
    descending: bool=False,
    #unused
    num_vmem_substages: int | None=None,
    block_token: int | None=None,
    contains_nans: bool = True,
) -> tuple[jax.Array, ...]:
  del num_vmem_substages, block_token, contains_nans
  operands = jax.tree.leaves(operand)
  if return_argsort:
    operands.append(
      jax.lax.broadcasted_iota(jnp.int32, operands[0].shape, 1))
  if descending and is_stable:
    operands.insert(num_keys, -jax.lax.broadcasted_iota(jnp.int32, operands[0].shape, 1))
    num_keys+=1  
  outs = jax.lax.sort(operands, num_keys=num_keys, is_stable=is_stable)    
  if descending and is_stable:
    outs = list(outs)
    outs.pop(num_keys-1)
  if descending:
    outs = tuple(map(_match_descending, outs))  
  if k is not None:
    outs = tuple(x[...,:k] for x in outs)    
  return outs
  
  
def check_lax_sort_pallas(
    operand: jax.Array | Sequence[jax.Array],
    k: int | None=None,
    block_token: int | None=None,
    return_argsort: bool=False,
    descending: bool=False,
    num_keys: int | None = None,
    is_stable: bool = False,
    print_outputs: bool = False,
):
  kwargs = dict(k=k, block_token=block_token, return_argsort=return_argsort, descending=descending, num_keys=num_keys, is_stable=is_stable)
  out_pallas = lax_sort_pallas(operand, **kwargs)

  # now build the lax output equivalent to check it matches
  if is_stable:
    # exact match is only acceptable solution
    out_xla = equiv_xla_based_sort(operand, **kwargs)
    print('Matches XLA: ', exact_match(out_pallas, out_xla))
  else:
    # check stably ordered output is the same as output, and that the output fully ordered is the same as input (to check output is a perkutation of input)

    # Valid unstable sorts of operand do not alter when stable sorted
    out_pallas_stable_sorted = equiv_xla_based_sort(
      out_pallas,
      num_keys=num_keys,
      is_stable=True,
      descending=descending,
    )
    print('out_pallas==stablesort(out_pallas): ', exact_match(out_pallas, out_pallas_stable_sorted))
    narrs = len(out_pallas)

    operands_fully_sorted = equiv_xla_based_sort(
      operand, **{**kwargs, 'num_keys':narrs})
    out_pallas_fully_sorted = equiv_xla_based_sort(
      out_pallas, **{**kwargs, 'num_keys':narrs, 'return_argsort':False})
    print('out_pallas is permute of input: ', exact_match(operands_fully_sorted, out_pallas_fully_sorted))
  
  def _run():
    return (
      lax_sort_pallas(operand, **kwargs),
      equiv_xla_based_sort(operand, **kwargs)
    )
  if not INTERPRET:
    benchmark(_run)
  if print_outputs:
    o_pallas, o_xla = _run()
    print(f'Pallas: {o_pallas}\nXLA: {o_xla}')

ntoken = 1
for num_operands in range(1,2):
  print(f'{num_operands=}')
  for num_keys in range(1, num_operands+1):
    for n in (
      #*(2**i for i in range(3,15)),
      2**9,#2**10,
      #2**13,
      #2**8, 2**9, 2**10, 2**11,
      #2**11,
      #2**13, 2**16,
      #2**19,
      #2**22,
      ):
      for dtype in (
        jnp.float32,
        #jnp.bfloat16
        jnp.int32,
      ):
        #x, aux, second_aux = (jax.random.normal(jax.random.key(0), (3, ntoken, n), jnp.float32)*1000).astype(dtype)
        
        x, aux, second_aux = jax.random.randint(jax.random.key(0), (3, ntoken,n), jnp.iinfo(jnp.int32).min, jnp.iinfo(jnp.int32).max, jnp.int32).view(dtype)
        for kwargs in (
          dict(),
          #dict(descending=True),dict(descending=True, k=128), dict(k=128), dict(return_argsort=True, is_stable=True),         
          dict(return_argsort=True),dict(is_stable=True),
          ):
          print(f'\n{(x.shape, x.dtype)}\n{num_operands=} {num_keys=} {kwargs=}')
          check_lax_sort_pallas((x, aux, second_aux)[:num_operands], num_keys=num_keys, **kwargs,
          #print_outputs=True,
          )
          #import sys; sys.exit(0)
#runtime.unassign()