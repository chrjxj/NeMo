# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import gc
import inspect
import logging
import math
import traceback
import warnings
from collections import namedtuple
from contextlib import ExitStack
from enum import Enum
from typing import Any, List, Optional, Tuple

import torch

# FIXME: Remove the following dependencies
from megatron.core import parallel_state
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.fp8_utils import is_float8tensor, modify_underlying_storage, quantize_param_shard
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.utils import is_submodule, is_te_min_version, log_on_each_pipeline_stage
from torch.distributed import _coalescing_manager
from torch.distributed.tensor._dtensor_spec import DTensorSpec

try:
    from transformer_engine.pytorch import fp8_model_init
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    HAVE_TE = True
except Exception:
    HAVE_TE = False


logger = logging.getLogger(__name__)


def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """Alternate to ``assert`` when in the backward context to print the error
    message ``s`` since otherwise, it is swallowed.
    """
    if not cond:
        print(s)
        traceback.print_stack()
        if raise_assertion_error:
            raise AssertionError(s)


def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_allocated = tensor._typed_storage()._size() == size.numel()
            if not already_allocated:
                tensor_storage_size = tensor._typed_storage()._size()
                _p_assert(
                    tensor_storage_size == 0,
                    "Tensor storage should have been resized to be 0 but got PLACEHOLDEr",
                )
                tensor._typed_storage()._resize_(size.numel())


def _free_storage(tensor: torch.Tensor):
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_freed = tensor._typed_storage()._size() == 0
            if not already_freed:
                _p_assert(
                    tensor.storage_offset() == 0,
                    "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                    f"storage offset: {tensor.storage_offset()}\n"
                    f"storage size: {tensor._typed_storage()._size()}\n"
                    f"tensor shape: {tensor.shape}",
                )
                tensor._typed_storage()._resize_(0)


TensorItemIndex = namedtuple('TensorItemIndex', ['global_data_index', 'size', 'item_id', 'bucket_id', 'shape'])
BucketIndex = namedtuple('BucketIndex', ['bucket_id', 'global_data_index', 'size', 'items'])
ShardBucketIndex = namedtuple(
    'ShardBucketIndex',
    ['bucket_id', 'global_data_index', 'local_data_index', 'bucket_data_index', 'size'],
)


@dataclasses.dataclass
class BucketingPolicy:
    """
    A policy for bucketing in Fully Sharded Data Parallel (FSDP) training.

    Attributes:
        suggested_bucket_size (int): The suggested size of each bucket in num of elements.
        fsdp_unit_modules (list): A list of module classes that are treated as a
            single unit for FSDP bucketing.
        data_parallel_sharding_strategy (str): The strategy used for sharding
            data parallel modules.

    Note:
        This policy is used to configure the bucketing behavior in FSDP training.
    """

    suggested_bucket_size: Optional[int] = 40_000_000
    fsdp_unit_modules: List[torch.nn.Module] = dataclasses.field(default_factory=list)
    data_parallel_sharding_strategy: str = 'no_shard'


def _pad(number_to_be_padded: int, divisor: int) -> int:
    return int(math.ceil(number_to_be_padded / divisor) * divisor)


def build_data_parallel_buffer_index(
    elements: List[torch.Size],
    data_parallel_rank: int,
    data_parallel_world_size: int,
    is_data_distributed: bool,
    ddp_config: DistributedDataParallelConfig,
    bucket_id: int = 0,
) -> Tuple[int, List[tuple], List[tuple], List[tuple]]:
    """
    Assuming that all input tensor elements contiguously compose a global
    buffer, give the index range of every tensor, the bucket in the buffer,
    and the (distributed) shard within the bucket. Note that the global bucket
    buffer is only temporarily allocated, but is abstractly tracked via indices
    deduced from the number of raw parameters assigned to this buffer / bucket.

    Args:
        elements (List[torch.Size]): List of input tensor.
        data_parallel_rank (int): Rank of the current process in the data parallel group.
        data_parallel_world_size (int): World size of the data parallel group.
        bucket_id (int, optional): The id of the bucket. Defaults to 0.

    Returns:
        Tuple[int, List[tuple], List[tuple], List[tuple]]: The index range of every tensor,
            every bucket and every in bucket local buffer.
    """

    def _pad_if_needed(data_index: int) -> int:
        """
        Pads data indices if using distributed optimizer (to ensure uniform sharding).
        """
        if ddp_config.data_parallel_sharding_strategy != 'no_shard':
            # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
            # This also helps cuBLAS pick more efficient algorithms for GEMMs.
            # We now ensure that all buckets start at a memory address that is 256-byte
            # aligned (128 values since params and grads use >= 16-bit precision).
            return _pad(data_index, math.lcm(data_parallel_world_size, 128))
        return data_index

    def add_item(item_id, item, bucket, item_index_map, bucket_id):
        # Add the parameter shape to the bucket.
        bucket.append(item)
        # New running bucket size with parameter buffer added.
        bucket_size = sum([it.numel() for it in bucket])
        # The item index map contains information on where each parameter item will
        # be stored in the tensor data buffer in a bucket.
        item_index_map.append(
            TensorItemIndex(
                # Global data index of the starting idx of this parameter 
                # = running global data index + updated bucket size - the parameter size.
                global_data_index=data_index + bucket_size - item.numel(),
                # Number of tensor elements in the parameter.
                size=item.numel(),
                # Index of the parameter to be buffered in the list of parameter shapes.
                item_id=item_id,
                # ID of the bucket that this parameter belongs to.
                bucket_id=bucket_id,
                # Shape of the parameter.
                shape=item,
            )
        )

    # For all bucket parameters, add information on the parameter to the item index map,
    # and add the size of the parameter to the bucket.
    item_index_map = []
    bucket = []
    data_index = 0
    for item_id, item in enumerate(elements):
        add_item(item_id, item, bucket, item_index_map, bucket_id)

    # Final bucket allocated size. Pad if necessary.
    bucket_size = sum([it.numel() for it in bucket])
    bucket_size = _pad_if_needed(bucket_size)

    # Bucket index contains information on what tensor items are in this bucket.
    bucket_index = BucketIndex(
        bucket_id=bucket_id,
        global_data_index=data_index,
        size=bucket_size,
        items=list(filter(lambda x: x.bucket_id == bucket_id, item_index_map)),
    )

    """
    Rank-Specific Buffer Sharding
    """
    # Calculate the shard size and the starting index of this shard in the global bucket.
    # Each rank / process will have a different shard size and starting index regardless
    # of whether the buffer is sharded or not, i.e. a "virtual shard" for unsharded buffers.
    shard_size = bucket_index.size // data_parallel_world_size
    bucket_data_index = shard_size * data_parallel_rank

    # Calculate the global data index of the starting index of this shard in the global bucket.
    global_data_index = bucket_index.global_data_index + bucket_data_index

    if is_data_distributed:
        # Sharded Data Buffer - This index stores the location (start) and size (end) of the
        # buffer shard in the global bucket.
        shard_bucket_index = ShardBucketIndex(
            bucket_id=bucket_id,
            global_data_index=global_data_index,    # Location of the buffer shard in the global bucket.
            local_data_index=0,                     # When the buffer is sharded, the local index of the data in this shard starts at 0.
            bucket_data_index=bucket_data_index,    # Location of the buffer shard relative to the global starting index of the bucket.
            size=shard_size,                        # Size of the bucket shard.
        )
    else:
        # Virtual sharding for bijections with other sharded buffers. But the buffer
        # itself is not actually sharded and contains the entire global bucket.
        shard_bucket_index = ShardBucketIndex(
            bucket_id=bucket_id,
            global_data_index=global_data_index,
            # When the buffer is not sharded, the local index of the data in this "virtual" shard begins at the
            # location of the buffer shard in the global bucket, because the entire bucket is stored in this buffer.
            local_data_index=global_data_index,
            bucket_data_index=bucket_data_index,
            size=shard_size,
        )

    # Return the tensor item index map in the buffer,
    # the bucket index with information on what items this bucket contains,
    # and the sharded bucket index.
    return item_index_map, bucket_index, shard_bucket_index


@dataclasses.dataclass
class Bucket:
    """
    A container for holding data in Fully Sharded Data Parallel (FSDP) training.

    Attributes:
        data (torch.Tensor): A tensor containing the data elements
            grouped together in a bucket.
        data_operation_event (Optional[torch.cuda.Event]): An optional CUDA event
            used to synchronize data operations.
        status (Any): An optional status object used to track the state of the bucket.

    Note:
        Buckets are used to optimize communication in FSDP training by
            grouping small tensors together.
    """

    data: torch.Tensor
    data_operation_event: Optional[torch.cuda.Event] = None
    status: Any = None


class TemporaryBucketAllocator:
    """
    A utility class for managing temporary buckets (buffers) used in FSDP
    operations like parameters unshard and gradients reduction.

    This allocator handles the dynamic allocation and deallocation of temporary memory buffers
    needed during FSDP (Fully Sharded Data Parallel) operations, particularly for parameters
    unshard and gradients reduction. It helps optimize memory usage by allowing temporary
    buckets to be released when no longer needed.

    Key Features:
        - Dynamic allocation of temporary buckets for FSDP operations
        - Memory-efficient management of temporary buffers
        - Support for both parameters unshard and gradients reduction operations
        - Automatic cleanup of unused buckets to save memory

    Usage:
        ```python
        # Create an allocator instance
        allocator = TemporaryBucketAllocator(name="gpt_parameters")

        # Allocate a temporary bucket
        temp_bucket = allocator.allocate(size=1024, dtype=torch.float32)

        # Use the temporary bucket for FSDP operations
        # ... perform all-gather or reduce-scatter ...

        # Free the bucket when done
        allocator.free(temp_bucket)
        ```

    Note:
        It's important to release temporary buckets after use to prevent memory leaks
        and optimize memory usage during training.
    """

    def __init__(self):
        self.buckets = {}

    def allocate(self, bucket_id: int, size: int, dtype: torch.dtype, device: torch.device) -> Bucket:
        """
        allocate a temporary bucket.
        """
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
        return self.buckets[bucket_id]

    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        if bucket_id in self.buckets:
            _free_storage(self.buckets[bucket_id].data)
            del self.buckets[bucket_id]


class StorageResizeBasedBucketAllocator(TemporaryBucketAllocator):
    """
    A specialized temporary bucket allocator that resizes the storage of temporary buckets
    based on the required size.
    """

    def __init__(self):
        super().__init__()
        self.buckets = {}  # {bucket_id: Bucket}

    def allocate(self, bucket_id: int, size: int, dtype: torch.dtype, device: torch.device) -> Bucket:
        """
        allocate a temporary bucket.
        """
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
        bucket = self.buckets[bucket_id]
        _alloc_storage(bucket.data, torch.Size([size]))
        return bucket

    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        if bucket_id in self.buckets:
            _free_storage(self.buckets[bucket_id].data)


class RotaryBucketAllocator(TemporaryBucketAllocator):
    """A specialized temporary bucket allocator that implements a circular buffer recycling strategy
    to minimize memory fragmentation in FSDP operations.

    RotaryBucketAllocator extends TemporaryBucketAllocator by maintaining a limited pool of
    pre-allocated buffers that are reused in a circular manner. This approach helps prevent
    memory fragmentation that typically occurs with frequent allocation and deallocation of
    temporary buffers during FSDP operations.

    Key Features:
        - Circular buffer recycling strategy for memory efficiency
        - Reduced memory fragmentation compared to dynamic allocation
        - Pre-allocated buffer pool for faster access
        - Automatic buffer reuse without explicit deallocation

    Usage:
        ```python
        # Create a rotary allocator
        allocator = RotaryBucketAllocator(name="gpt_parameters")

        # Get a temporary buffer from the pool
        temp_bucket = allocator.allocate(size=1024, dtype=torch.float32)

        # Use the temporary bucket for FSDP operations
        # ... perform all-gather or reduce-scatter ...

        # Free the bucket when done, make it in idle buffer pool
        allocator.free(temp_bucket)
        ```
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.num_global_buffer = 0
        self.idle_buffer = []  # [buffer_id]
        self.using_buffer = {}  # {bucket_id: buffer_id}
        self.buckets = {}

    def allocate(self, bucket_id: int, size: int, dtype: torch.dtype, device: torch.device) -> Bucket:
        """
        allocate a temporary bucket.
        """

        def _get_global_buffer(buffer_id: int):
            return parallel_state.get_global_memory_buffer().get_tensor(
                [size], dtype=dtype, name=self._get_gbuf_name(buffer_id)
            )

        if bucket_id in self.using_buffer:
            buffer_id = self.using_buffer[bucket_id]
            return Bucket(data=_get_global_buffer(buffer_id))

        if len(self.idle_buffer) == 0:
            # allocate new buffer
            buffer_id = self.num_global_buffer
            self.num_global_buffer += 1
            self.idle_buffer.append(buffer_id)

        buffer_id = self.idle_buffer.pop(0)
        self.using_buffer[bucket_id] = buffer_id
        return Bucket(data=_get_global_buffer(buffer_id))

    def _get_gbuf_name(self, buffer_id: int):
        return f"{self.name}_{buffer_id}"

    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        if bucket_id in self.using_buffer:
            buffer_id = self.using_buffer.pop(bucket_id)
            self.idle_buffer.append(buffer_id)


@dataclasses.dataclass
class DTensorParam:
    """
    A class that represents a DTensor parameter with its associated metadata.

    Attributes:
        param_data (torch.Tensor): The original PyTorch parameter data.
        sharding_spec (Optional[DTensorSpec]): The sharding specification for the DTensor.
    """

    param_data: torch.Tensor
    sharding_spec: Optional[DTensorSpec] = None
    sharded_param: Optional[torch.Tensor] = None


class DataParallelBuffer:
    """
    A class that manages the data parallel buffer for Fully Sharded Data Parallel (FSDP) training.
    It has two operating modes given a bucket of module parameters:

        - Sharded: The bucket is sharded across the data parallel group, and each rank will manage a
            shard of the bucket that is persistently stored in this buffer.
        - Unsharded: The bucket is not sharded, and the entire bucket is persistently stored in this
            buffer. Virtual shards of this unsharded buffer can be retrieved from each rank when needed.

    This design supports interoperability of sharded and unsharded buffers, such as ZeRO-1 and ZeRO-2,
    where buffers associated with sharded parameters can be utilized with buffers associated with unsharded
    parameters through the use of "virtual" or rank-specific shards for the unsharded buffers.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        params: List[torch.nn.Parameter],
        is_data_distributed: bool,
        bucket_id: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        temporary_bucket_allocator: Optional[TemporaryBucketAllocator] = None,
        init_meta_only: bool = False,
        is_dtype_float8: bool = False,
        gradient_scaling_factor: Optional[float] = None,
    ) -> None:
        self.ddp_config = ddp_config
        self.params = params
        _param_dtype = {p.dtype for p in self.params}
        assert len(_param_dtype) == 1, f'params have different dtypes: {_param_dtype}'
        self.is_data_distributed = is_data_distributed
        self.bucket_id = bucket_id
        self.dtype = dtype if dtype else next(iter(_param_dtype))
        self.device = device
        self.data_parallel_group = data_parallel_group
        self.dp_rank = torch.distributed.get_rank(group=self.data_parallel_group)
        self.dp_world_size = torch.distributed.get_world_size(group=self.data_parallel_group)
        self.temporary_bucket_allocator = (
            temporary_bucket_allocator if temporary_bucket_allocator else TemporaryBucketAllocator()
        )
        self.is_dtype_float8 = is_dtype_float8
        self.gradient_scaling_factor = gradient_scaling_factor

        # Build the data parallel buffer index, which contains information on where each parameter tensor will
        # be stored in this sharded / distributed buffer.
        (self.item_index_map, self.bucket_index, self.shard_bucket_index) = build_data_parallel_buffer_index(
            [p.shape for p in self.params],
            self.dp_rank,
            self.dp_world_size,
            is_data_distributed,
            ddp_config,
            bucket_id=bucket_id,
        )

        # Allocate a buffer Tensor to persistently store the data for this (shard of) the buffer.
        self.data_size = self.bucket_index.size if not is_data_distributed else self.shard_bucket_index.size
        if init_meta_only:
            self.data = None
        else:
            self.data = torch.empty(self.data_size, dtype=self.dtype, device=device)

        # Count all parameters in this buffer and store their enumerated index.
        self.param_idx = {p: i for i, p in enumerate(self.params)}

        # TODO(@cspades): What does this placeholder bucket hold? Doesn't seem to be used anywhere.
        self.placeholder_bucket = None
        self.placeholder_items = {}

    def fetch_bucket(self, dtype: Optional[torch.dtype] = None, and_allocate_params_data: bool = False) -> Bucket:
        """
        Fetch a communication buffer for data-parallel operations.

        The size of the bucket is defined by the `DataParallelBuffer` instance.
        If `and_allocate_params_data` is True, this method resets the parameter
        data stored in the `DataParallelBuffer` instance.

        Args:
            dtype (Optional[torch.dtype], optional): The data type of the tensor
                to fetch a buffer for. Defaults to None.
            and_allocate_params_data (bool, optional): Whether to allocate and
                reset parameter data. Defaults to False.

        Returns:
            Bucket: The communication buffer for the specified data type.
        """
        if dtype is None:
            dtype = self.dtype
        bucket_index = self.bucket_index

        if not self.is_data_distributed and dtype == self.dtype:
            bucket = Bucket(
                data=self.data[bucket_index.global_data_index : bucket_index.global_data_index + bucket_index.size]
            )
        else:
            # Bucket (unsharded) needs to be retrieved. If the temporary bucket cache
            # does not have the bucket corresponding to the bucket_id, it will allocate
            # a new Bucket with an empty tensor. Otherwise, it will simply return the
            # pre-allocated bucket with pre-existing data.
            bucket = self.temporary_bucket_allocator.allocate(
                bucket_id=bucket_index.bucket_id,
                size=bucket_index.size,
                dtype=dtype,
                device=self.device,
            )

            if and_allocate_params_data:
                for p in self.params:
                    item_id = self.param_idx[p]
                    # Attach the bucket's data to the parameters managed by this buffer.
                    if is_float8tensor(p):
                        p._data = self.get_item_from_bucket(bucket, item_id).view(p.shape)
                    else:
                        p.data = self.get_item_from_bucket(bucket, item_id).view(p.shape)

                # TODO(@cspades): Why doesn't something like this work?
                # for p in self.params:
                #     item_id = self.param_idx[p]
                #     item_index = self.item_index_map[item_id]
                #     # If the parameter is sharded, we need to use the shape of the unsharded parameter.
                #     param_shape = p.shape if item_index.size == p.numel() else item_index.shape
                #     # Attach the bucket's data to the parameters managed by this buffer.
                #     item_data = self.get_item_from_bucket(bucket, item_id).view(param_shape)
                #     if is_float8tensor(p):
                #         p._data = item_data
                #     else:
                #         p.data = item_data

        return bucket

    def free_bucket_storage(self, and_free_params_data: bool = False):
        """
        Release the storage of a temporary communication bucket.

        If the bucket is temporary, this method frees its storage.
        If `and_free_params_data` is True, this method also releases the storage
            of the parameter data stored in the `DataParallelBuffer` instance.

        Args:
            and_free_params_data (bool, optional): Whether to also release the
                storage of the parameter data. Defaults to False.

        Returns:
            None
        """
        if not self.is_data_distributed:
            # Only free the allocated bucket if the buffer is sharded.
            # Otherwise, the buffer contains the entire bucket.
            return

        # Free the memory backing the temporarily-allocated bucket associated with this buffer.
        self.temporary_bucket_allocator.free(self.bucket_index.bucket_id)

        if and_free_params_data:
            if self.placeholder_bucket is None:
                # Create a new placeholder bucket.
                self.placeholder_bucket = Bucket(
                    data=torch.empty(self.bucket_index.size, dtype=self.dtype, device=self.device)
                )
                for p in self.params:
                    item_id = self.param_idx[p]
                    # Index the placeholder bucket's data for all the parameters managed by this buffer.
                    self.placeholder_items[item_id] = self.get_item_from_bucket(self.placeholder_bucket, item_id).view(
                        p.shape
                    )
                # Deallocate the memory backing the placeholder bucket (as well as the placeholder item parameters).
                _free_storage(self.placeholder_bucket.data)
            # Attach the placeholder bucket's data to the parameters managed by this buffer.
            for p in self.params:
                item_id = self.param_idx[p]
                if is_float8tensor(p):
                    p._data = self.placeholder_items[item_id]
                else:
                    p.data = self.placeholder_items[item_id]

    def _get_item_slice_in_shard(self, item_id: int) -> Tuple[int, int]:
        """
        Return the coordinates of the slice of the item that is contained
        in this buffer shard. In other words, this returns the coordinates
        of all of the data in this item that is stored in this shard.
        
        Maps to the global coordinates of the item in the bucket when added to
        the starting coordinate of the item in the bucket, and maps to the local
        coordinates of the item in the shard when added to the difference between
        the starting coordinate of the item and the starting coordinate of the
        shard in the global bucket (i.e. mapping from item coordinates to global
        coordinates to shard coordinates).
        """
        item_index = self.item_index_map[item_id]
        shard_bucket_index = self.shard_bucket_index

        # Define the boundaries of the item in the global buffer,
        # as well as the boundaries of the shard in the buffer.
        # The tensor and shard boundaries may not align, so we
        # need to find their intersection, i.e. the slice of the
        # item that is contained in this shard.
        item_global_start = item_index.global_data_index
        item_global_end = item_index.global_data_index + item_index.size
        shard_bucket_start = shard_bucket_index.global_data_index
        shard_bucket_end = shard_bucket_index.global_data_index + shard_bucket_index.size

        # If the item is not in the shard, return 0, 0.
        if item_global_start > shard_bucket_end or item_global_end < shard_bucket_start:
            return (0, 0)

        # Find the slice of the item that is contained in this buffer shard relative
        # to the starting index of the item in the global bucket. If the item starts
        # before the shard, then the offset to reach the start of the slice of the item
        # in the shard from the starting index of the item is the difference between
        # the start of the shard and the start of the item. Otherwise, the offset is 0,
        # because the start of the item is within the shard.
        start = max(item_global_start, shard_bucket_start) - item_global_start
        # If the item ends after the shard, then the offset to reach the end of the
        # slice of the item in the shard from the starting index of the item is the
        # difference between the end of the shard and the start of the item. Otherwise,
        # the offset is just the size of the item, because the end of the item is
        # contained within the shard.
        end = min(item_global_end, shard_bucket_end) - item_global_start

        # Return the boundaries of the item in the shard relative to the global
        # start of the item.
        return (start, end)

    # pylint: disable=missing-function-docstring
    def locate_item_in_global_item(self, item_id: int) -> Tuple[int, int]:
        """
        TODO(@cspades): Identical to _get_item_slice_in_shard, except it handles
        the case where the buffer is not sharded, in which case we shouldn't
        constrain the item's coordinates to the virtual shard boundaries.
        """
        item_index = self.item_index_map[item_id]
        if not self.is_data_distributed:
            return (0, item_index.size)

        slice_start, slice_end = self._get_item_local_shard_index(item_id)
        if slice_start == slice_end:
            return (0, 0)

        # This reverts the mapping in _get_item_local_shard_index from the
        # output of _get_item_slice_in_shard...
        # TODO(@cspades): Call and return _get_item_slice_in_shard instead
        # of _get_item_local_shard_index and applying these offsets! (DRY)
        local_shard_index_to_global_index_offset = (
            self.shard_bucket_index.global_data_index - self.shard_bucket_index.local_data_index
        )
        slice_start += local_shard_index_to_global_index_offset
        slice_end += local_shard_index_to_global_index_offset
        return (
            slice_start - item_index.global_data_index,
            slice_end - item_index.global_data_index,
        )

    def _get_item_local_shard_index(self, item_id: int) -> Tuple[int, int]:
        """
        Return the local coordinates of the slice of this buffer's shard that
        contains the item with the given ID. In other words, this returns the
        coordinates of all of the data in this shard associated with the item.
        
        Maps to the global coordinates of the item in the bucket when added to
        the starting coordinate of the shard in the global bucket, and maps to
        the coordinates of the item contained in the shard when added to the
        difference between the starting coordinate of the shard and the starting
        coordinate of the item in the global bucket (i.e. mapping from shard
        coordinates to global coordinates to item coordinates).
        """
        # Get the coordinates of the slice of the item that is contained in this shard.
        slice_start, slice_end = self._get_item_slice_in_shard(item_id)
        if slice_start == slice_end:
            # The item does not intersect this shard.
            return (0, 0)

        item_index = self.item_index_map[item_id]
        shard_bucket_index = self.shard_bucket_index

        """
        Compute the offset that maps the coordinates of the slice of the item in this shard to the local coordinates
        of the slice of this shard that contains the item, for retrieval of the item's data stored in this shard.
            - If distributed, then evaluates to item_start - shard_start (because shard_local_data_index = 0).
            - If not distributed, then evaluates to item_start (because shard_local_data_index = shard_global_data_index).
              This maps the coordinates of the slice of the item in this shard to the global coordinates of the
              slice of the item in the bucket because the unsharded buffer entirely backs the global bucket.
        """
        offset = (
            item_index.global_data_index - shard_bucket_index.global_data_index + shard_bucket_index.local_data_index
        )

        # Return the local coordinates of the slice of the item contained in this (sharded or unsharded) buffer.
        return (offset + slice_start, offset + slice_end)

    def _get_item_local_index(self, item_id: int) -> Tuple[int, int]:
        """
        Return the local coordinates of the slice of this buffer's data that
        contains the item with the given ID.
        """
        if not self.is_data_distributed:
            # Return the boundary indices of the item in the bucket buffer.
            # Shortcut case where the buffer / bucket is not sharded, so we
            # can retrieve the untruncated item tensor from the buffer without
            # calculating the intersection of the item and the shard.
            item_index = self.item_index_map[item_id]
            # Note: Buffer coordinates = bucket coordinates when the buffer is not sharded.
            return (item_index.global_data_index, item_index.global_data_index + item_index.size)
        # Otherwise, return the local coordinates of the slice of this
        # buffer's shard that intersects the specified item tensor.
        return self._get_item_local_shard_index(item_id)

    def set_item(self, item_id: int, item_data: torch.Tensor) -> None:
        """
        Update a Tensor item managed by the `DataParallelBuffer` instance,
        i.e. store (a shard of) the Tensor in this buffer's datastore.

        The storage of the item is mapped to the communication bucket.
        This method updates the item data and ensures consistency with the bucket.

        Args:
            item_id (int): The ID of the tensor item to update.
            item_data (torch.Tensor): The new data for the tensor item.

        Returns:
            None
        """
        # When fully sharded, we need to get the slice of the item to be stored in this shard.
        # Otherwise, we can just flatten the entire item since this buffer contains the entire bucket.
        if self.is_data_distributed:
            # Get the coordinates of the slice of the item that is contained in this shard.
            slice_start, slice_end = self._get_item_slice_in_shard(item_id)
            # Flatten the item data and get the slice of the item to place in the shard.
            item_data = item_data.flatten()[slice_start:slice_end]
        # Get the local coordinates of the slice of this buffer's shard that
        # intersects the specified item tensor.
        local_index_start, local_index_end = self._get_item_local_index(item_id)
        # Copy the slice of the item associated with this sharded buffer into the
        # slice of this buffer's shard that intersects the specified item tensor.
        shard = self.data[local_index_start:local_index_end]
        if shard.numel() > 0:
            shard.data.copy_(item_data.flatten())

    def get_item(self, item_id: int, only_shard: bool = False) -> torch.Tensor:
        """
        Retrieve a tensor item managed by the `DataParallelBuffer` instance,
        i.e. get all the item data stored in this sharded or unsharded buffer.

        The storage of the item is mapped to the communication bucket.
        If `only_shard` is True, returns only the shard of the item corresponding
            to the current process / rank, a "virtual shard" for unsharded buffers.
        Otherwise, returns the entire item, which could be a bucket shard or bucket.

        Args:
            item_id (int): The ID of the tensor item to retrieve.
            only_shard (bool, optional): Whether to return only the shard of the
                item. Defaults to False.

        Returns:
            torch.Tensor: The retrieved tensor item.
        """
        if only_shard:
            # Get segment of the item saved in the shard associated with this rank.
            # Used in situations where the buffer is unsharded but another buffer
            # associated with this buffer's data is sharded, so you need to retrieve
            # a "virtual shard" of the item corresponding to this process / rank
            # from this unsharded buffer.
            start, end = self._get_item_local_shard_index(item_id)
        else:
            # Retrieve all item data stored in this buffer. Buffer could be sharded or unsharded.
            # When sharded, return the intersection of the item and the bucket shard stored in this buffer.
            # When unsharded, return the entire item in the unsharded bucket stored in this buffer.
            start, end = self._get_item_local_index(item_id)

        return self.data[start:end]

    def get_item_from_bucket(self, bucket: Bucket, item_id: int):
        """
        Get Tensor item data from the given bucket specified by the item ID.
        """
        item_index = self.item_index_map[item_id]
        bucket_index = self.bucket_index
        start_index = item_index.global_data_index - bucket_index.global_data_index
        end_index = start_index + item_index.size
        item = bucket.data[start_index:end_index]
        return item

    def get_shard_from_bucket(self, bucket: Bucket):
        """
        Get the shard from the provided bucket associated with the sharding strategy of this buffer.
        """
        shard_bucket_index = self.shard_bucket_index
        offset = shard_bucket_index.bucket_data_index
        shard_size = shard_bucket_index.size
        shard = bucket.data[offset : offset + shard_size]
        return shard

    def get_shard_from_local_buffer(self) -> torch.Tensor:
        """
        Get the shard or virtual shard of the bucket stored in this buffer.
        """
        index = self.shard_bucket_index
        # If the buffer is sharded, return the shard stored in this buffer.
        # Otherwise, return the virtual shard of the bucket associated with this buffer,
        # corresponding to the process / rank of this buffer.
        return self.data[index.local_data_index : index.local_data_index + index.size]


@dataclasses.dataclass
class ParameterGroup:
    """
    A group of model parameters with associated metadata for data-parallel training.

    This dataclass encapsulates a list of PyTorch parameters and additional information
    necessary for managing data-parallel operations, such as data type, gradient requirements,
    and buffer assignments.
    """

    params: List[torch.nn.Parameter]
    dtype: Optional[torch.dtype] = None
    is_expert_param: bool = False
    requires_grad: Optional[bool] = None
    fsdp_unit_id: Optional[int] = None
    data_parallel_world_size: Optional[int] = None
    model_weight_buffer: Optional[DataParallelBuffer] = None
    main_weight_buffer: Optional[DataParallelBuffer] = None
    main_grad_buffer: Optional[DataParallelBuffer] = None


def _get_parameter_groups(
    module: torch.nn.Module,
    policy: BucketingPolicy,
    meta_device_init_fp8_params: dict,
    bucket_group_by_fsdp_unit: bool = True,
):
    """
    Get the parameter group for the given module and parameters.
    """

    # Step 0: Register new FSDP unit modules.
    param_to_name = {p: name for name, p in module.named_parameters()}
    # fsdp_units is a list of lists of parameter names, one list per FSDP unit module.
    fsdp_units = []
    if policy.fsdp_unit_modules:
        param_to_id = {}
        for i, p in enumerate(module.parameters()):
            param_to_id[p] = i
        fsdp_modules = []
        # Loop through all sub-modules of the module.
        for m in module.modules():
            # Skip nested FSDP module, i.e. FSDP modules already have their sub-module parameters registered.
            if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
                continue
            # If the sub-module is a FSDP unit module, add its parameter (names) to the list of FSDP units.
            if isinstance(m, tuple(policy.fsdp_unit_modules)):
                fsdp_units.append([param_to_name[p] for p in m.parameters()])
                fsdp_modules.append(m)

    def _does_param_require_new_bucket(param):
        """
        Split shared embedding parameters into separate bucket if using distributed
        optimizer that makes use of reduce-scatters instead of all-reduces.
        This ensures that the first and last pipeline stage partition optimizer state
        for the shared embedding parameters the same way across DP replicas, allowing
        the DP reduce-scatter to be before the embedding all-reduce.
        """
        return getattr(param, "shared_embedding", False) and policy.data_parallel_sharding_strategy != "no_shard"

    is_expert_parameter = lambda p: not getattr(p, 'allreduce', True)

    # Step 1: Group the parameters according to their execution order and attributes.
    # FSDP unit module parameters are split into multiple parameter sub-groups.
    # All parameters in the module are assigned a parameter group, even non-FSDP modules.
    parameter_groups = []
    for name, param in module.named_parameters():

        # We need this information to correctly dynamically allocate Tensors!
        param_attrs = dict(
            dtype=(
                "float8" if is_float8tensor(param) or meta_device_init_fp8_params.get(name, False) else param.dtype
            ),
            is_expert_param=is_expert_parameter(param),
            requires_grad=param.requires_grad,
            fsdp_unit_id=None,
        )

        # For all the new FSDP unit parameters collected, assign an ID number
        # associated with which unit module the parameter belongs to.
        for fsdp_unit_id, fsdp_unit in enumerate(fsdp_units):
            if name in fsdp_unit:
                param_attrs["fsdp_unit_id"] = fsdp_unit_id
                break

        found_group = False
        # Check if the parameter already belongs to a group.
        for param_group in parameter_groups:
            group_attrs = {key: value for key, value in param_group.__dict__.items() if key in param_attrs}
            # Parameters are grouped by their attributes and FSDP unit module ID.
            if group_attrs == param_attrs:
                param_group.params.append(param)
                found_group = True
                break

        # If the parameter does not belong to any group, create a new group for it.
        if not found_group:
            parameter_groups.append(ParameterGroup([param], **param_attrs))

    # Step 2: Bucket the parameters based on the guide bucket size.
    # Parameter groups can be split into multiple buckets based on bucket size.
    suggested_bucket_size = policy.suggested_bucket_size
    bucket_groups = []
    for group in parameter_groups:
        bucket = []

        # Bucket attributes.
        basic_attrs = {
            key: value
            for key, value in group.__dict__.items()
            if key in ['dtype', 'is_expert_param', 'requires_grad', 'fsdp_unit_id']
        }
        for param in group.params:
            # TODO(@cspades): Understand this.
            if _does_param_require_new_bucket(param):
                if len(bucket) > 0:
                    # Append the current bucket to the list of bucket groups.
                    bucket_groups.append(ParameterGroup(bucket, **basic_attrs))
                # Create a new bucket for the parameter.
                bucket_groups.append(ParameterGroup([param], **basic_attrs))
                bucket = []
                continue

            # Append the parameter to the current bucket.
            bucket.append(param)
            # If the current bucket has reached the suggested bucket size,
            # append the bucket as a parameter group to the list of bucket groups
            # and create a new bucket. Used to control the size of parameter
            # groups that are not members of FSDP unit modules.
            if (
                group.fsdp_unit_id is None
                and suggested_bucket_size
                and sum([p.numel() for p in bucket]) >= suggested_bucket_size
            ):
                # Create a new parameter group from a subset of the original
                # parameter group's parameters.
                bucket_groups.append(ParameterGroup(bucket, **basic_attrs))
                bucket = []
                continue

        # Append the parameter group bucket to the list of bucket groups.
        if bucket:
            bucket_groups.append(ParameterGroup(bucket, **basic_attrs))

    # Map each parameter to its bucket group ID.
    param_to_param_group = {}
    for group_id, group in enumerate(bucket_groups):
        for param in group.params:
            param_to_param_group[param] = group_id

    # Step 3: Generate the groups of collective buckets, where each group aggregates
    # the collectives per FSDP unit. This improves performance by reducing
    # the number of collective calls and increasing per-collective efficiency.
    bucket_group_of_bucket = {}
    # This initializes the mapping from bucket ID to the full group of bucket IDs
    # that are associated with this bucket ID.
    for bucket_id in range(len(bucket_groups)):
        # Every bucket group associated with a bucket ID should contain the bucket ID.
        bucket_group_of_bucket[bucket_id] = [bucket_id]

    # Set aggregate buckets by FSDP units, i.e. buckets pertaining to the same
    # FSDP unit module and are either expert or non-expert parameters should
    # end up in the same bucket group for NCCL.
    # Non-FSDP unit parameters will be assigned to the identity bucket group.
    if bucket_group_by_fsdp_unit:
        bucket_group_map = {}

        # Assign bucket IDs to bucket groups from the same FSDP unit module.
        for bucket_id, param_group in enumerate(bucket_groups):
            if param_group.fsdp_unit_id is None:
                # Ignore parameter groups without FSDP unit IDs.
                # These come from the parameter group processing loop
                # which loops over all module parameters and groups by
                # everything else if the fsdp_unit_id is not set.
                continue
            # Create an FSDP unit ID sub-classified by expert / non-expert parameters.
            # Then index this pair in bucket_group_map.
            id = (param_group.fsdp_unit_id, param_group.is_expert_param)
            if id not in bucket_group_map:
                bucket_group_map[id] = []
            bucket_group_map[id].append(bucket_id)

        # For each aggregated bucket group based on FSDP unit module and parameter type,
        # overwrite the previously initialized bucket group associated with the bucket ID.
        for bucket_group in bucket_group_map.values():
            for bucket_id in bucket_group:
                bucket_group_of_bucket[bucket_id] = bucket_group
 
    # Return the full list of split bucket / parameter groups, the mapping from parameters to their bucket group ID,
    # and the mapping from bucket ID to the full group of bucket IDs that are NCCL-aggregated with this bucket ID.
    return (bucket_groups, param_to_param_group, bucket_group_of_bucket)


class ParamAndGradBuffer:
    """A class that manages parameter grouping, buffer allocation, and
    communication operations for data-parallel distributed training.

    This class provides functionality to:
    1.  Group parameters based on their data types and communication group sizes.
    2.  Create contiguous buffers for model weights, gradients, and high-precision
        main weights.
    3.  Handle parameter unsharding, gradient reduction, and weight
        synchronization operations.

    Key Features:
        - Efficient parameter grouping based on data types and communication patterns
        - Memory-efficient contiguous buffer allocation
        - Support for mixed-precision training with main weights
        - Distributed operations including parameters all-gather and gradients
            reduce-scatter/all-reduce
        - Synchronized weight updates between model and main weights

    Note:
        This class is designed for distributed training scenarios where efficient
        parameter management and communication are crucial for performance.

    Args:
        ddp_config (DistributedDataParallelConfig): The distributed data parallel
            configuration.
        module (torch.nn.Module): The module whose parameters are to be grouped
            and flatten.
        bucketing_policy (BucketingPolicy): The bucketing policy.
        data_parallel_group (torch.distributed.ProcessGroup): The data parallel group.
        expert_data_parallel_group (Optional[torch.distributed.ProcessGroup]):
            The expert data parallel group.
        preserve_fp32_weights (bool): Whether to preserve FP32 weights.
        grad_reduce_in_fp32 (bool): Whether to reduce gradients in FP32.
        gradient_scaling_factor (Optional[float]): The gradient scaling factor.
        expert_gradient_scaling_factor (Optional[float]): The expert gradient
            scaling factor.
        device (torch.device): The parameter and gradient buffer device.
        only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad (bool):
            Whether to only create the gradient buffer and main weight buffer
            for parameters that require gradients. Default is True.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        bucketing_policy: BucketingPolicy,
        data_parallel_group: torch.distributed.ProcessGroup,
        expert_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        preserve_fp32_weights: bool = True,
        grad_reduce_in_fp32: bool = True,
        gradient_scaling_factor: Optional[float] = None,
        expert_gradient_scaling_factor: Optional[float] = None,
        device: torch.device = torch.device('cuda'),
        only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad: bool = True,
        reset_parameters_for_meta_device_init_module: bool = False,
    ):
        self.ddp_config = ddp_config
        self.module = module
        self.bucketing_policy = bucketing_policy
        self.param_to_name = {p: name for name, p in self.module.named_parameters()}
        self.preserve_fp32_weights = preserve_fp32_weights
        self.grad_reduce_in_fp32 = grad_reduce_in_fp32
        self.data_parallel_group = data_parallel_group
        self.expert_data_parallel_group = expert_data_parallel_group
        self.params = list(module.parameters())
        self.gradient_scaling_factor = gradient_scaling_factor
        self.expert_gradient_scaling_factor = expert_gradient_scaling_factor
        self.device = device
        self.only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad = (
            only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad
        )
        self.reset_parameters_for_meta_device_init_module = reset_parameters_for_meta_device_init_module

        # Mark fp8 param.
        meta_device_init_fp8_params = {}
        if reset_parameters_for_meta_device_init_module:
            for m in module.modules():
                if not isinstance(m, TransformerEngineBaseModule):
                    continue
                for name, param in m.named_parameters(recurse=False):
                    # The fp8 param initialized from the meta device may NOT be
                    # an fp8 tensor, according to the internal logic of the TE
                    # to determine whether this parameter is fp8 or not.
                    fp8_meta_index = m.param_init_meta[name].fp8_meta_index
                    if m.primary_weights_in_fp8 and fp8_meta_index is not None:
                        meta_device_init_fp8_params[self.param_to_name[param]] = True

        # Get the parameter groups.
        (self.parameter_groups, self.param_to_param_group, self.bucket_group_of_bucket) = _get_parameter_groups(
            module, bucketing_policy, meta_device_init_fp8_params
        )
        self._init_each_parameter_group_buffers(meta_device_init_fp8_params)

        # Initialize the optimizer named parameters.
        self.optimizer_named_parameters = self._init_optimizer_named_parameters()

        self._log_parameter_groups()

    def _log_parameter_groups(self):
        """
        Log the parameter groups for all pipeline stages.
        """
        # Log buckets for all PP stages.
        bucket_groups = self.parameter_groups
        param_to_name = self.param_to_name
        log_strs = []
        log_strs.append(f'Number of parameter groups for FSDP: {len(bucket_groups)}')
        for index, group in enumerate(bucket_groups):
            numel = 0
            for param in group.params:
                numel += param.numel()
            log_strs.append(
                f"Params for group {index+1} ({numel} elements, dtype: {group.dtype}, "
                f"fsdp_unit_id: {group.fsdp_unit_id}, "
                f"has_weight_buffer: {group.model_weight_buffer is not None}, "
                f"has_grad_buffer: {group.main_grad_buffer is not None}, "
                f"has_main_weight_buffer: {group.main_weight_buffer is not None}):"
            )
            for param in group.params:
                log_strs.append(f'\t{param_to_name[param]}')
        if torch.distributed.get_rank() == 0:
            logger.info('\n'.join(log_strs))

    def _init_each_parameter_group_buffers(self, meta_device_init_fp8_params):
        """
        Initialize the buffers for each parameter group.
        """
        # FSDP Strategy: DP, ZeRO-1, ZeRO-2, or ZeRO-3.
        data_parallel_sharding_strategy = self.ddp_config.data_parallel_sharding_strategy
        if data_parallel_sharding_strategy == 'no_shard':
            is_model_weight_buffer_distributed = False
            is_main_weight_buffer_distributed = False
            is_grad_buffer_distributed = False
        elif data_parallel_sharding_strategy == 'optim':
            is_model_weight_buffer_distributed = False
            is_main_weight_buffer_distributed = True
            is_grad_buffer_distributed = False
        elif data_parallel_sharding_strategy == 'optim_grads':
            is_model_weight_buffer_distributed = False
            is_main_weight_buffer_distributed = True
            is_grad_buffer_distributed = True
        elif data_parallel_sharding_strategy == 'optim_grads_params':
            is_model_weight_buffer_distributed = True
            is_main_weight_buffer_distributed = True
            is_grad_buffer_distributed = True
        else:
            raise ValueError(f'Invalid data_parallel_sharding_strategy: {data_parallel_sharding_strategy}')

        self.memory_allocator_for_model_weight_buffer = StorageResizeBasedBucketAllocator()
        self.buffer_all_in_one = True

        preserve_fp32_weights = self.preserve_fp32_weights
        grad_reduce_in_fp32 = self.grad_reduce_in_fp32
        buffer_size = {torch.float32: 0, torch.float16: 0, torch.bfloat16: 0, "float8": 0}
        
        # For all bucket groups (partitioned parameter groups)...
        for group_id, group in enumerate(self.parameter_groups):
            dp_group = self.data_parallel_group if not group.is_expert_param else self.expert_data_parallel_group
            group.data_parallel_world_size = torch.distributed.get_world_size(group=dp_group)
            gradient_scaling_factor = (
                self.gradient_scaling_factor if not group.is_expert_param else self.expert_gradient_scaling_factor
            )
            # Check if the parameter group is FP8.
            one_param = group.params[0]
            is_dtype_float8 = is_float8tensor(one_param) or meta_device_init_fp8_params.get(
                self.param_to_name[one_param], False
            )
            if is_dtype_float8:
                param_dtype = torch.uint8
                grad_dtype = torch.bfloat16
            else:
                param_dtype = group.params[0].dtype
                grad_dtype = param_dtype

            # Check if the parameter group requires a grad buffer or main weight buffer.
            should_create_grad_buffer_or_main_weight_buffer = (
                not self.only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad or group.requires_grad
            )

            # Initialize the model weight buffer from bucket parameters.
            if data_parallel_sharding_strategy != 'no_shard':
                group.model_weight_buffer = DataParallelBuffer(
                    self.ddp_config,
                    group.params,
                    is_data_distributed=is_model_weight_buffer_distributed and group.data_parallel_world_size > 1,
                    dtype=param_dtype,
                    device=self.device,
                    data_parallel_group=dp_group,
                    init_meta_only=True,
                    is_dtype_float8=is_dtype_float8,
                    temporary_bucket_allocator=self.memory_allocator_for_model_weight_buffer,
                    bucket_id=group_id,
                )

            # Initialize the main weight buffer.
            if should_create_grad_buffer_or_main_weight_buffer and preserve_fp32_weights:
                group.main_weight_buffer = DataParallelBuffer(
                    self.ddp_config,
                    group.params,
                    is_data_distributed=is_main_weight_buffer_distributed and group.data_parallel_world_size > 1,
                    dtype=torch.float32,
                    device=self.device,
                    data_parallel_group=dp_group,
                    init_meta_only=True,
                    bucket_id=group_id,
                )

            # Initialize the main grad buffer.
            if should_create_grad_buffer_or_main_weight_buffer:
                group.main_grad_buffer = DataParallelBuffer(
                    self.ddp_config,
                    group.params,   # Proxy because the number of gradient parameters is the same as the number of model parameters.
                    is_data_distributed=is_grad_buffer_distributed and group.data_parallel_world_size > 1,
                    dtype=torch.float32 if grad_reduce_in_fp32 else grad_dtype,
                    device=self.device,
                    data_parallel_group=dp_group,
                    init_meta_only=True,
                    is_dtype_float8=not grad_reduce_in_fp32 and grad_dtype is torch.uint8,
                    gradient_scaling_factor=gradient_scaling_factor,
                    bucket_id=group_id,
                )

                # Track number of elements in the main grad buffer by dtype.
                if grad_reduce_in_fp32:
                    buffer_size[torch.float32] += group.main_grad_buffer.data_size
                elif group.main_grad_buffer.is_dtype_float8:
                    buffer_size["float8"] += group.main_grad_buffer.data_size
                else:
                    buffer_size[group.main_grad_buffer.dtype] += group.main_grad_buffer.data_size

        reset_context_args = {"init_param_with_fp8": self.ddp_config.fp8_param_gather}
        module_reset_flag = {}
        if self.reset_parameters_for_meta_device_init_module:
            self.param_to_direct_module = {}
            for name, m in self.module.named_modules():
                for p in m.parameters(recurse=False):
                    self.param_to_direct_module[p] = (name, m)

            meta_params_numel = 0
            cuda_params_numel = 0
            cpu_params_numel = 0
            for group in self.parameter_groups:
                for p in group.params:
                    if p.is_meta:
                        meta_params_numel += p.numel()
                    elif p.device.type == 'cuda':
                        cuda_params_numel += p.numel()
                    else:
                        cpu_params_numel += p.numel()
            log_str = (
                f"Meta params numel: {meta_params_numel / 1_000_000:.2f} M, "
                f"CUDA params numel: {cuda_params_numel / 1_000_000:.2f} M, "
                f"CPU params numel: {cpu_params_numel / 1_000_000:.2f} M"
            )
            log_on_each_pipeline_stage(logger, logging.INFO, log_str)

        # Initialize the model weight buffer data of each parameter group.
        # Specifically, replace the Torch module's parameter data with tensors
        # whose memory managed by the model weight buffer, and store a shard
        # of all the parameters across ranks in the model weight buffer.
        for group in self.parameter_groups:
            wbuf = group.model_weight_buffer
            if wbuf:
                # Manually instantiate an empty tensor into the model weight buffer.
                wbuf.data = torch.empty(wbuf.data_size, dtype=wbuf.dtype, device=self.device)
                # Fetch an empty global bucket from the model weight buffer.
                bucket = wbuf.fetch_bucket()
            mbuf = group.main_weight_buffer
            if mbuf:
                # Manually instantiate an empty tensor into the main weight buffer.
                mbuf.data = torch.empty(mbuf.data_size, dtype=mbuf.dtype, device=self.device)
            for item_id, p in enumerate(group.params):
                # Model Weight (Low-Precision) Buffer Initialization
                if wbuf:
                    if self.reset_parameters_for_meta_device_init_module and p.is_meta:
                        m_name, m = self.param_to_direct_module[p]
                        if not module_reset_flag.get(m_name, False) and hasattr(m, "reset_parameters"):
                            old_params = list(m.parameters(recurse=False))

                            # If the GPU memory over threshold, empty cache to leave
                            # some memory for initialization of the model on the
                            # CUDA device.
                            if check_gpu_memory(threshold=0.5):
                                gc.collect()
                                torch.cuda.empty_cache()

                            m.to_empty(device=self.device, recurse=False)
                            if (
                                HAVE_TE
                                and is_te_min_version("0.9.0")
                                and not isinstance(m, TransformerEngineBaseModule)
                            ):
                                reset_context_args["with_cuda_rng_tracker"] = True
                            with ResetParametersContext(**reset_context_args):
                                m.reset_parameters()
                            module_reset_flag[m_name] = True
                            new_params = list(m.parameters(recurse=False))

                            self._reset_parameters(old_params, new_params)
                            p = group.params[item_id]

                            # After resetting parameters, delete fp8 transpose cache
                            # if we do not need keep cache.
                            if not self.ddp_config.keep_fp8_transpose_cache_when_using_custom_fsdp:
                                for _param in m.parameters(recurse=False):
                                    if is_float8tensor(_param):
                                        _param._transpose_invalid = True
                                        _param._transpose = None
                    assert not p.is_meta, (self.param_to_name[p], module_reset_flag)

                    # Copy the model weight parameter tensor into the buffer.
                    # When distributed, this shards and preserves the data across all ranks.
                    wbuf.set_item(item_id, p.data)

                    # Retrieve the newly allocated parameter data from the global bucket.
                    # Attach the bucket-allocated parameter data to the module parameter,
                    # to use the bucket-allocated data for autograd and NCCL.
                    new_param_data = wbuf.get_item_from_bucket(bucket, item_id).view(p.shape)
                    if is_float8tensor(p):
                        # TODO(@cspades): Megatron / TE, understand if necessary.
                        modify_underlying_storage(p, new_param_data)
                    else:
                        # Detach the bucket-allocated parameter data from the computational graph
                        # before copying the old parameter data into the new parameter data
                        # to prevent backpropagation into a deleted parameter / Tensor.

                        # Copy the values of the original parameter data into the bucket-allocated
                        # parameter data. Detach the module parameter because parameters that require
                        # gradients in the computational graph do not support in-place operations.
                        old_param_data = p.data
                        p.data = new_param_data
                        assert old_param_data._base is None
                        p.data.detach().copy_(old_param_data)
                        del old_param_data

                # Main Weight (High-Precision) Buffer Initialization
                if mbuf:
                    if hasattr(p, 'get_high_precision_init_val'):
                        # TODO(@cspades): Megatron / TE, understand if necessary.
                        mbuf.set_item(item_id, p.get_high_precision_init_val())
                        p.clear_high_precision_init_val()
                    else:
                        # Insert a copy of the model weight parameter tensor into
                        # the (high-precision) main weight buffer.
                        # Nothing else needs to be done, because the main weights
                        # do not require autograd operations, only possibly sharding.
                        mbuf.set_item(item_id, p)

                if wbuf and wbuf.is_data_distributed:
                    """
                    When MCore Custom FSDP `optim_grads_params` is enabled,
                    it is necessary to save the tensor local shard. This local shard is
                    accessible through the  `fully_shard_param_local_shard`
                    attribute of the tensor.

                    This attribute contains the local shard of the fully
                    sharded parameter, which is essential for correctly
                    saving and loading the model state when using
                    `optim_grads_params` with FSDP.

                    Example:
                        >>> # Assuming `tensor` is a fully sharded parameter
                        >>> local_shard = tensor.fully_shard_param_local_shard
                        >>> # Save the local shard as needed
                    """
                    # Get local shard from the model weight buffer.
                    local_shard = wbuf.get_item(item_id, only_shard=True)
                    # Attach the parameter to the local shard, and vice versa.
                    local_shard.fsdp_shard_orig_param = p
                    p.fully_shard_param_local_shard = local_shard
                    # Get the index of the parameter relative to the global start of the shard
                    # if distributed, or the global start of the item in the bucket if not.
                    p.fully_shard_param_local_index = wbuf.locate_item_in_global_item(item_id)

                    def disable_shard_param_to_function(*unused):
                        """Prevents users from accessing the 'to' operation
                        on parameters after sharding.

                        This restriction helps maintain data integrity and
                        proper sharding behavior by disabling direct 'to'
                        device/dtype operations on sharded parameters.
                        """
                        raise RuntimeError(
                            "Your model is wrapped by MCore Custom FSDP. All "
                            "parameter dtypes and devices must be set before FSDP "
                            "wrapping. After FSDP wrapping, parameter storage "
                            "is sharded and you cannot modify parameter "
                            "dtypes or devices."
                        )

                    setattr(p, 'to', disable_shard_param_to_function)

                    def disable_shard_param_cpu_function(*unused):
                        """
                        Return an empty tensor when moving the FSDP-managed parameter to CPU.
                        """
                        warnings.warn(
                            "The parameters are sharded by custom fsdp, " "and no actual cpu operation is performed."
                        )
                        return torch.empty([], device='cpu')

                    setattr(p, 'cpu', disable_shard_param_cpu_function)

            if wbuf and wbuf.is_data_distributed:
                # Free the memory backing the temporarily-allocated bucket associated with this buffer.
                # The module parameters will still reference the (now empty) bucket Tensor.
                # Each rank of the data buffer will persistently store a shard of the module.
                # This reduces the memory footprint of the model in FSDP, such that the only
                # time the entire model's weights are allocated in memory is during initialization,
                # before forward activations and gradients are allocated in training.
                wbuf.free_bucket_storage()

        # Allocate the main_weight buffer and main_grad buffer data in one buffer.
        if self.buffer_all_in_one:
            self.buffer = {
                torch.float32: torch.empty(buffer_size[torch.float32], dtype=torch.float32, device=self.device),
                torch.float16: torch.empty(buffer_size[torch.float16], dtype=torch.float16, device=self.device),
                torch.bfloat16: torch.empty(buffer_size[torch.bfloat16], dtype=torch.bfloat16, device=self.device),
                "float8": torch.empty(buffer_size["float8"], dtype=torch.uint8, device=self.device),
            }
            offset = {torch.float32: 0, torch.float16: 0, torch.bfloat16: 0, "float8": 0}

        def _alloc(dtype, size):
            """
            If using a single buffer for main model weights and gradients,
            allocate memory per dtype buffer with size at the current offset.
            Return the allocated slice of the buffer data Tensor.

            If not using a single buffer, then return an empty Tensor on this device.
            """
            if self.buffer_all_in_one:
                if dtype == torch.uint8:
                    dtype = "float8"
                data = self.buffer[dtype][offset[dtype] : offset[dtype] + size]
                offset[dtype] += size
                return data
            return torch.empty(size, dtype=dtype, device=self.device)

        # Main Gradient Buffer Initialization
        for group in self.parameter_groups:
            gbuf = group.main_grad_buffer
            if not gbuf:
                # No gradient sharding.
                continue
            # Allocate the main grad buffer data, and attach it to the main grad buffer.
            gbuf.data = _alloc(gbuf.dtype, gbuf.data_size)
            gbuf.data.zero_()
            for item_id, p in enumerate(group.params):
                # Attach the main grad buffer data and metadata to the parameter.
                p.fsdp_managed_main_grad = gbuf.get_item(item_id)
                p._gbuf = gbuf
                p._item_id = item_id

                def main_grad_getter(p):
                    # Make sure main_grad memory is allocated when initially accessed.
                    bucket = p._gbuf.fetch_bucket()
                    gbuf = p._gbuf
                    item_id = p._item_id
                    # View it as p.shape so you can insert the param.grad into the bucket seamlessly.
                    return gbuf.get_item_from_bucket(bucket, item_id).view(p.shape)

                # Patch the parameter class to include a main_grad property.
                # Utilized in the gradient reduction pipeline to save computed
                # data-parallel gradients on every rank and reduce-scatter them.
                setattr(p.__class__, 'main_grad', property(main_grad_getter))

            if gbuf.is_data_distributed:
                # Free the memory backing the temporarily-allocated bucket associated with this buffer.
                # TODO(@cspades): Where did we allocate / fetch the bucket for GBuf?
                gbuf.free_bucket_storage()

        # Clean up deallocated memory.
        gc.collect()
        torch.cuda.empty_cache()

    def _reset_parameters(self, old_params, new_params):
        assert len(old_params) == len(new_params)
        param_map = {}
        for old_param, new_param in zip(old_params, new_params):
            param_map[old_param] = new_param
            self.param_to_name[new_param] = self.param_to_name[old_param]
            del self.param_to_name[old_param]

            self.param_to_param_group[new_param] = self.param_to_param_group[old_param]
            del self.param_to_param_group[old_param]

            self.param_to_direct_module[new_param] = self.param_to_direct_module[old_param]
            del self.param_to_direct_module[old_param]

        for item_id, p in enumerate(self.params):
            if p in param_map:
                new_p = param_map[p]
                self.params[item_id] = new_p

        for group in self.parameter_groups:
            for item_id, p in enumerate(group.params):
                if p not in param_map:
                    continue
                new_p = param_map[p]
                group.params[item_id] = new_p
                for buf in [
                    group.model_weight_buffer,
                    group.main_weight_buffer,
                    group.main_grad_buffer,
                ]:
                    if buf is None:
                        continue
                    buf.param_idx[new_p] = buf.param_idx[p]
                    del buf.param_idx[p]

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        for group in self.parameter_groups:
            if group.main_grad_buffer is None:
                continue
            group.main_grad_buffer.data *= scaling_factor
        self.update_main_grads()

    def zero_grad(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation
        for the next iteration of training.
        """
        for _, param in self.optimizer_named_parameters:
            if param.grad is not None and param.grad._base is None:
                # For tensors that are not referenced, trying to use storage
                # resize to make memory free immediately.
                _free_storage(param.grad)
            param.grad = None

        for group in self.parameter_groups:
            if group.main_grad_buffer is None:
                continue
            group.main_grad_buffer.data.zero_()

    def _init_optimizer_named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """
        Initialize the named parameters that require gradients for the optimizer.
        If sharded, parameters are stored in either an FP32 buffer or a buffer with
        precision equivalent to the original model weights. If not sharded, the
        parameters are just the original module parameters.
        """
        named_parameters = []
        for pg in self.parameter_groups:
            if pg.main_grad_buffer is None:
                # Parameter group does not require gradients.
                continue

            optimizer_state_is_shard = pg.main_grad_buffer.is_data_distributed or (
                pg.main_weight_buffer and pg.main_weight_buffer.is_data_distributed
            )
            for item_id, orig_param in enumerate(pg.params):

                # Get the parameters for the model from the buffer if sharded, otherwise default to the
                # original local parameters. Depending on the sharding strategy, the optimizer will
                # update the specified version of the model weights: high-precision, low-precision,
                # or the in-place unsharded module parameters.
                if pg.main_weight_buffer:
                    # High-precision model weights with sharding.
                    param = pg.main_weight_buffer.get_item(item_id, only_shard=optimizer_state_is_shard)
                elif pg.model_weight_buffer:
                    # Low-precision model weights with sharding.
                    param = pg.model_weight_buffer.get_item(item_id, only_shard=optimizer_state_is_shard)
                else:
                    # Not buffered, use the original parameter.
                    param = orig_param

                def set_param_attribute_closure(param, orig_param):
                    def set_param_attribute():
                        for attr_name in [
                            'requires_grad',
                            'sequence_parallel',
                            'shared',
                            'tensor_model_parallel',
                            'partition_dim',
                            'partition_stride',
                            'is_embedding_or_output_parameter',
                        ]:
                            if hasattr(orig_param, attr_name):
                                setattr(param, attr_name, getattr(orig_param, attr_name))

                    return set_param_attribute

                # Save parameter attributes in the parameter, and restore those attributes.
                setattr(param, 'reset_attribute', set_param_attribute_closure(param, orig_param))
                setattr(param, 'orig_param', orig_param)
                param.reset_attribute()

                # Add the selected parameter to the optimizer named parameter list for optimization.
                named_parameters.append((self.param_to_name[orig_param], param))

        # Return named parameters for the optimizer.
        return named_parameters

    def update_main_grads(self):
        """Update the main gradients for preparing the optimizer step."""
        for _, param in self.optimizer_named_parameters:
            param.reset_attribute()
            orig_param = param.orig_param
            group = self.parameter_groups[self.param_to_param_group[orig_param]]
            item_id = group.main_grad_buffer.param_idx[orig_param]
            optimizer_grad = group.main_grad_buffer.get_item(
                item_id, only_shard=group.main_weight_buffer.is_data_distributed
            )
            setattr(
                param,
                'grad',
                optimizer_grad.to(param.dtype) if optimizer_grad.numel() > 0 else None,
            )

    @property
    def num_buckets(self):
        """Return the number of buckets."""
        return len(self.parameter_groups)

    @torch.no_grad()
    def copy_main_weights_to_model_weights(self):
        """Update the model weights from the main weights."""
        for pg in self.parameter_groups:
            mbuf = pg.main_weight_buffer
            wbuf = pg.model_weight_buffer
            if mbuf is None:
                continue

            fp8_params = []
            shard_fp32_from_fp8 = []
            shard_offsets_in_fp8 = []
            shard_model_params = []

            for param in pg.params:
                item_id = mbuf.param_idx[param]
                if wbuf:
                    if wbuf.is_data_distributed or mbuf.is_data_distributed:
                        model_param = wbuf.get_item(item_id, only_shard=True)
                        main_weight = mbuf.get_item(item_id, only_shard=True)
                    else:
                        model_param = wbuf.get_item(item_id)
                        main_weight = mbuf.get_item(item_id)
                else:
                    assert not mbuf.is_data_distributed
                    model_param = param
                    main_weight = pg.main_weight_buffer.get_item(item_id)

                if is_float8tensor(param):
                    fp8_params.append(param)
                    if model_param.numel() == 0:
                        shard_fp32_from_fp8.append(None)
                        shard_offsets_in_fp8.append(None)
                        shard_model_params.append(None)
                    else:
                        shard_fp32_from_fp8.append(main_weight)
                        shard_offsets_in_fp8.append(wbuf.locate_item_in_global_item(item_id)[0])
                        shard_model_params.append(model_param)
                    continue

                if model_param.numel() > 0:
                    model_param.data.copy_(main_weight.view(model_param.shape))

            quantize_param_shard(
                fp8_params,
                shard_fp32_from_fp8,
                shard_offsets_in_fp8,
                wbuf.data_parallel_group,
                shard_model_params,
            )

    @torch.no_grad()
    def copy_model_weights_to_main_weights(self):
        """Copy the model weights to the main weights."""
        for group in self.parameter_groups:
            mbuf = group.main_weight_buffer
            if mbuf is None:
                continue
            wbuf = group.model_weight_buffer
            if mbuf.is_data_distributed:
                copyin_data = wbuf.get_shard_from_local_buffer()
            else:
                copyin_data = wbuf.data
            assert mbuf.data.numel() == copyin_data.numel(), (
                f"Master weight buffer size {mbuf.data.numel()} does not match "
                f"model weight buffer size {copyin_data.numel()}"
            )
            mbuf.data.copy_(copyin_data.data)

    def all_gather_parameters(self, async_op: bool = True):
        """All gather the parameters.
        Args:
            async_op (bool, optional): Whether to do the all-reduce
                asynchronously. Defaults to False.
        """
        assert all(
            [not g.model_weight_buffer.is_data_distributed for g in self.parameter_groups]
        ), 'all_gather_parameters() should only be called when parameters are not sharded.'

        all_gather_ops = []
        for g in self.parameter_groups:
            shard = g.model_weight_buffer.get_shard_from_local_buffer()
            all_gather_handler = torch.distributed.all_gather_into_tensor(
                output_tensor=g.model_weight_buffer.data,
                input_tensor=shard,
                group=g.model_weight_buffer.data_parallel_group,
                async_op=async_op,
            )
            if async_op:
                all_gather_ops.append(all_gather_handler)

        for op in all_gather_ops:
            op.wait()

    def reduce_scatter_gradients(self, async_op: bool = True):
        """Reduce scatter the gradients.
        Args:
            async_op (bool, optional): Whether to do the all-reduce
                asynchronously. Defaults to False.
        """
        assert all(
            [not g.main_grad_buffer.is_data_distributed for g in self.parameter_groups]
        ), 'reduce_scatter_gradients() should only be called when gradients are not sharded.'

        reduce_scatter_ops = []
        for g in self.parameter_groups:
            gbuf = g.main_grad_buffer
            if gbuf is not None:
                continue
            scaling_factor = gbuf.gradient_scaling_factor
            reduce_op = gradient_reduce_preprocessing(gbuf.data, scaling_factor, self.ddp_config)
            reduce_scatter_handler = torch.distributed.reduce_scatter_tensor(
                output=gbuf.get_shard_from_local_buffer(),
                input=gbuf.data,
                op=reduce_op,
                group=g.main_grad_buffer.data_parallel_group,
                async_op=async_op,
            )

            if async_op:
                reduce_scatter_ops.append(reduce_scatter_handler)

        for op in reduce_scatter_ops:
            op.wait()

    def all_reduce_gradients(self, async_op: bool = False):
        """All reduce the gradients.
        Args:
            async_op (bool, optional): Whether to do the all-reduce
                asynchronously. Defaults to False.
        """
        assert all(
            [not g.main_grad_buffer.is_data_distributed for g in self.parameter_groups if g.main_grad_buffer]
        ), 'all_reduce_gradients() should only be called when gradients are not sharded.'

        all_reduce_ops = []
        for g in self.parameter_groups:
            gbuf = g.main_grad_buffer
            if gbuf is not None:
                continue
            scaling_factor = gbuf.gradient_scaling_factor
            reduce_op = gradient_reduce_preprocessing(gbuf.data, scaling_factor, self.ddp_config)
            all_reduce_handler = torch.distributed.all_reduce(
                gbuf.data, op=reduce_op, group=gbuf.data_parallel_group, async_op=async_op
            )
            if async_op:
                all_reduce_ops.append(all_reduce_handler)

        for op in all_reduce_ops:
            op.wait()


class BucketStatus(Enum):
    """
    An enumeration of possible statuses for a data-parallel communication bucket.

    Attributes:
        EMPTY (int): The bucket is empty and not in use.
        COMMUNICATING (int): The bucket is currently being used for communication.
        READY_TO_USE (int): The bucket is filled with data and ready for use.
    """

    EMPTY = 1
    COMMUNICATING = 2
    READY_TO_USE = 3


class GradReducePipeline:
    """
    Pipeline for reducing gradients.
    """

    def __init__(
        self,
        param_and_grad_buffer: ParamAndGradBuffer,
        cuda_stream: Optional[torch.cuda.Stream] = None,
        check_nans: bool = False,
    ) -> None:
        self.buffer = param_and_grad_buffer
        # Track the status of ongoing gradient reduce-scatter operations before optimizer step.
        self.grad_reduce_queue = []
        self.bucket_status = {
            # All buckets are initially deallocated / empty after initialization of ParamAndGradBuffer.
            i: BucketStatus.EMPTY
            for i in range(self.buffer.num_buckets)
            if self.buffer.parameter_groups[i].main_grad_buffer
        }
        # Track the number of parameters in each bucket that are ready for gradient reduce-scatter.
        self.bucket_grad_ready_params = [set() for _ in range(self.buffer.num_buckets)]
        self.cuda_stream = cuda_stream
        self.check_nans = check_nans

    @property
    def num_buckets(self):
        """Return the number of buckets."""
        return self.buffer.num_buckets

    def reset(self):
        """Handle the processing tasks and reset the pipeline."""
        self.wait_for_previous_grad_reduce(0)
        for bucket_id, grad_ready_params in enumerate(self.bucket_grad_ready_params):
            param_list = self.buffer.parameter_groups[bucket_id].params
            n_params = len(param_list)
            param_to_name = self.buffer.param_to_name
            assert len(grad_ready_params) == 0, (
                f"Found {len(grad_ready_params)} out of {n_params} parameters that are ready for "
                f"reduce-scatter/all-reduce, but the pipeline is being reset. "
                f"grad_ready_params: {[param_to_name[p] for p in grad_ready_params]} "
                f"param_list: {[param_to_name[p] for p in param_list]}"
            )

        for bucket_id, _ in self.bucket_status.items():
            gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
            gbuf.free_bucket_storage()
            self.bucket_status[bucket_id] = BucketStatus.EMPTY

    def reduce_gradients(self, params: List[torch.Tensor], suggested_queue_capacity: Optional[int] = None):
        """Reduce the gradients for the given parameters.
        Args:
            params (List[torch.Tensor]): The parameters.
            suggested_queue_capacity (int, optional): The suggested queue capacity.
                Defaults to None.
        """
        for param in params:
            bucket_id = self.buffer.param_to_param_group[param]
            param_group = self.buffer.parameter_groups[bucket_id]
            if not param.requires_grad:
                assert param_group.requires_grad is False, (
                    f"Param {self.buffer.param_to_name[param]} has requires_grad=False, "
                    f"but it is in a parameter group with requires_grad=True."
                )
                continue
            assert param_group.requires_grad, (
                f"Param {self.buffer.param_to_name[param]} has requires_grad=True, "
                f"but it is in a parameter group with requires_grad=False."
            )

            # Mark grad as ready for reduce-scatter/all-reduce.
            self.bucket_grad_ready_params[bucket_id].add(param)
            if len(self.bucket_grad_ready_params[bucket_id]) == len(param_group.params):
                self.wait_for_previous_grad_reduce(suggested_queue_capacity=suggested_queue_capacity)
                self.mark_bucket_ready(bucket_id, async_rs=True)

    def wait_for_previous_grad_reduce(
        self, suggested_queue_size: int = 1, suggested_queue_capacity: Optional[int] = None
    ):
        """
        Wait for the previous reduce-scatter/all-reduce to finish.
        Args:
            suggested_queue_size (int, optional): The recommended queue size in buckets. Defaults to 1.
            suggested_queue_capacity (Optional[int], optional): The recommended queue capacity
                in number of parameters in all buckets in the reduction queue. Defaults to None.
        """
        if suggested_queue_capacity is not None:
            queue_space = sum(
                [
                    self.buffer.parameter_groups[bucket_id].main_grad_buffer.bucket_index.size
                    for _, _, bucket_id in self.grad_reduce_queue
                ]
            )
            while queue_space > suggested_queue_capacity:
                grad_reduce_event, free_up_grad_bucket, bucket_id = self.grad_reduce_queue.pop(0)
                grad_reduce_event.wait()
                free_up_grad_bucket()
                queue_space -= self.buffer.parameter_groups[bucket_id].main_grad_buffer.bucket_index.size
        else:
            suggested_queue_size = max(0, min(suggested_queue_size, self.buffer.num_buckets - 1))
            while len(self.grad_reduce_queue) > suggested_queue_size:
                grad_reduce_event, free_up_grad_bucket, _ = self.grad_reduce_queue.pop(0)
                grad_reduce_event.wait()
                free_up_grad_bucket()

    def mark_bucket_ready(self, bucket_id: int, async_rs: bool = False) -> bool:
        """Mark the bucket ready for reduce-scatter/all-reduce, if all bucket in
        the bucket group are ready, then do the reduce-scatter/all-reduce.
        Args:
            bucket_id (int): The bucket to be marked.
            async_rs (bool, optional): Whether to do the reduce-scatter/all-reduce
                asynchronously. Defaults to False.
        Returns:
            bool: True if the bucket is go for reduce-scatter/all-reduce.
        """
        # Prepare the bucket group for gradient reduce. Note that the
        # some bucket parameters do not require grad, so we need to
        # remove them from the bucket group.
        bucket_group = self.buffer.bucket_group_of_bucket[bucket_id]
        bucket_group = [i for i in bucket_group if self.buffer.parameter_groups[i].main_grad_buffer]
        # If any bucket in the bucket group is not ready, skip the gradient reduce
        # waiting for the bucket group to be all ready before executing.
        for bucket_id in bucket_group:
            param_group = self.buffer.parameter_groups[bucket_id]
            if len(self.bucket_grad_ready_params[bucket_id]) != len(param_group.params):
                return False

        # Wait for all CUDA operations to complete in the current stream before reduce-scatter.
        current_stream = torch.cuda.current_stream()
        reduce_scatter_stream = self.cuda_stream if self.cuda_stream is not None else torch.cuda.current_stream()
        reduce_scatter_stream.wait_stream(current_stream)

        dp_group = self.buffer.parameter_groups[bucket_id].main_grad_buffer.data_parallel_group
        with torch.cuda.stream(reduce_scatter_stream):
            with _coalescing_manager(dp_group, async_ops=async_rs) as coalescing_event:
                grad_shards = {}
                for bucket_id in bucket_group:
                    # Fetch pre-allocated main gradient bucket.
                    gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
                    bucket = gbuf.fetch_bucket()
                    scaling_factor = gbuf.gradient_scaling_factor
                    reduce_op = gradient_reduce_preprocessing(gbuf.data, scaling_factor, gbuf.ddp_config)
                    if gbuf.ddp_config.data_parallel_sharding_strategy == 'no_shard':
                        # All-reduce the gradients on every rank. No scattering / sharding necessary.
                        torch.distributed.all_reduce(bucket.data, op=reduce_op, group=gbuf.data_parallel_group)
                    else:
                        # Get the shard of the gradient from the pre-allocated bucket.
                        # The reduced gradient will be scattered into this shard of the
                        # bucket managed by the sharded buffer on this rank.
                        grad_shard = gbuf.get_shard_from_bucket(bucket)
                        # pylint: disable=C0301
                        # The `grad_shard`` is part of `bucket.data`` and the following
                        # new empty is important for memory safety, when using
                        # TORCH_NCCL_AVOID_RECORD_STREAMS=1.
                        # For reference: https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486
                        grad_shard = torch.empty_like(grad_shard)
                        # Reduce-scatter the unsharded gradient into a gradient shard in the allocated bucket.
                        torch.distributed.reduce_scatter_tensor(
                            output=grad_shard,
                            input=bucket.data,
                            op=reduce_op,
                            group=gbuf.data_parallel_group,
                        )
                        grad_shards[bucket_id] = grad_shard
                    self.bucket_status[bucket_id] = BucketStatus.COMMUNICATING
            # Wait for all reduce-scatter operations to complete.
            coalescing_event.wait()
            for bucket_id in bucket_group:
                gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
                if gbuf.ddp_config.data_parallel_sharding_strategy != 'no_shard':
                    # Accumulate the reduced gradient shard into the sharded buffer.
                    local_buffer = gbuf.get_shard_from_local_buffer()
                    local_buffer += grad_shards[bucket_id]
            # Record a checkpoint for the event to synchronize against the reduce-scatter stream.
            reduce_scatter_view_out_event = reduce_scatter_stream.record_event()

        free_up_grad_bucket_func = {}
        for bucket_id in bucket_group:

            def get_closure(bucket_id):
                def free_up_grad_bucket():
                    # Empty the set of parameters that are ready for gradient reduction.
                    self.bucket_grad_ready_params[bucket_id] = set()
                    gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
                    if gbuf.is_data_distributed:
                        # Free the memory backing the temporarily-allocated bucket associated with this buffer.
                        gbuf.free_bucket_storage()
                    # Mark the bucket as deallocated / empty.
                    self.bucket_status[bucket_id] = BucketStatus.EMPTY

                return free_up_grad_bucket

            free_up_grad_bucket_func[bucket_id] = get_closure(bucket_id)

        if async_rs:
            for bucket_id, free_up_grad_bucket in free_up_grad_bucket_func.items():
                self.grad_reduce_queue.append((reduce_scatter_view_out_event, free_up_grad_bucket, bucket_id))
            return True

        reduce_scatter_view_out_event.wait()
        for free_up_grad_bucket in free_up_grad_bucket_func.values():
            free_up_grad_bucket()
        return True


class PrefetchOrder(Enum):
    """
    An enumeration of possible prefetch orders for data-parallel operations.

    Attributes:
        FORWARD_PASS_ORDER (int): Prefetch in the order of forward pass computation.
        BACKWARD_PASS_ORDER (int): Prefetch in the order of backward pass computation.
    """

    FORWARD_PASS_ORDER = 0
    BACKWARD_PASS_ORDER = 1


class AllGatherPipeline:
    """
    Pipeline for all-gathering parameters.
    """

    def __init__(self, param_and_grad_buffer: ParamAndGradBuffer) -> None:
        self.buffer = param_and_grad_buffer
        # Track the status of all-gather operations for each bucket.
        self.param_gather_event_map = {}
        # All buckets are initially deallocated / empty after initialization of ParamAndGradBuffer.
        self.bucket_status = {i: BucketStatus.EMPTY for i in range(self.buffer.num_buckets)}
        # Track whether each bucket can be deallocated.
        self.bucket_can_be_released = {i: False for i in range(self.buffer.num_buckets)}

        # Map each bucket to the bucket group it belongs to by enumerated ID.
        # Made to collect a subset of buckets in the same bucket group.
        self.bucket_to_bucket_group = {}
        group_id = 0
        for bucket_group in self.buffer.bucket_group_of_bucket.values():
            new_group = False
            for bucket_id in bucket_group:
                if bucket_id not in self.bucket_to_bucket_group:
                    new_group = True
                    break
            if new_group:
                group_id += 1
                for bucket_id in bucket_group:
                    self.bucket_to_bucket_group[bucket_id] = group_id

    @property
    def num_buckets(self):
        """Return the number of buckets."""
        return self.buffer.num_buckets

    def reset(self):
        """Reset the pipeline state."""
        if len(self.param_gather_event_map) > 0:
            warnings.warn(
                "There are still pending all-gather tasks, process them. " f"Bucket status: {self.bucket_status}.",
                UserWarning,
            )
            while len(self.param_gather_event_map) > 0:
                bucket_id = next(iter(self.param_gather_event_map))
                self.wait_bucket_ready(bucket_id)
        for bucket_id in self.bucket_can_be_released:
            self.bucket_can_be_released[bucket_id] = True
        self.recycle_unused_buckets()

        assert all([status is BucketStatus.EMPTY for status in self.bucket_status.values()]), (
            f"There are still working buckets, it is not safe to reset. " f"bucket_status: {self.bucket_status}."
        )
        assert all([not can_be_released for can_be_released in self.bucket_can_be_released.values()]), (
            f"The bucket can be released table is in an abnormal state, not safe to reset. "
            f"bucket_can_be_released: {self.bucket_can_be_released}."
        )

    def all_gather_params(
        self,
        params: List[torch.Tensor],
        prefetch: bool = False,
        prefetch_order: PrefetchOrder = PrefetchOrder.FORWARD_PASS_ORDER,
        suggested_AG_prefetch_size: Optional[int] = None,
    ):
        """All-gather the params. If prefetch is enabled, prefetch next buckets
        in the order of `prefetch_order`.

        Args:
            params (List[torch.Tensor]): The list of params to be all-gathered.
            prefetch (bool, optional): Whether to prefetch the next bucket. Defaults to False.
            prefetch_order (PrefetchOrder, optional): The order of prefetching.
                Defaults to PrefetchOrder.FORWARD_PASS_ORDER.
            suggested_AG_prefetch_size (Optional[int], optional):
                The suggested prefetch size for all-gathering. Defaults to None.
        """
        if len(params) == 0:
            return

        ag_buckets = [self.buffer.param_to_param_group[item] for item in params]
        ag_buckets = list(sorted(set(ag_buckets)))  # Sort in order of unique bucket ID.
        parameter_groups = self.buffer.parameter_groups

        # If prefetch is enabled, we will add prefetch buckets to ag_buckets.
        if prefetch:

            def next_bucket_id(ag_buckets):
                """
                Search for the next bucket ID that is not in the list of all-gather buckets.
                """
                if prefetch_order == PrefetchOrder.FORWARD_PASS_ORDER:
                    # Search from the initial bucket.
                    bucket_id = ag_buckets[0] + 1
                    for i in ag_buckets[1:]:
                        if i != bucket_id:
                            break
                        bucket_id += 1
                else:
                    # Search from the last bucket.
                    bucket_id = ag_buckets[-1] - 1
                    for i in reversed(ag_buckets[:-1]):
                        if i != bucket_id:
                            break
                        bucket_id -= 1
                if bucket_id < 0 or bucket_id >= self.buffer.num_buckets:
                    # Out of bounds, return None.
                    return None
                return bucket_id

            if suggested_AG_prefetch_size is not None:
                bucket_id = next_bucket_id(ag_buckets)
                while bucket_id is not None:
                    all_gather_size = sum(
                        [parameter_groups[i].model_weight_buffer.bucket_index.size for i in ag_buckets]
                    )
                    if all_gather_size >= suggested_AG_prefetch_size:
                        # Reached the prefetch limit.
                        break
                    # Extend the list of all-gather buckets with another group of buckets.
                    ag_buckets.extend(self.buffer.bucket_group_of_bucket[bucket_id])
                    # Re-sort and find the next bucket not in the list.
                    ag_buckets = list(sorted(set(ag_buckets)))
                    bucket_id = next_bucket_id(ag_buckets)
            else:
                bucket_id = next_bucket_id(ag_buckets)
                if bucket_id is not None:
                    ag_buckets.extend(self.buffer.bucket_group_of_bucket[bucket_id])
                    ag_buckets = list(sorted(set(ag_buckets)))

        # Only all-gather on buckets that have not been allocated yet.
        ag_buckets = [i for i in ag_buckets if self.bucket_status[i] == BucketStatus.EMPTY]
        if len(ag_buckets) == 0:
            return

        # Divide buckets into aggregate groups. We need to reconstruct the bucket groups
        # because the all-gather parameter groups may be a subset of the buckets.
        bucket_group_to_buckets = {}
        for bucket_id in ag_buckets:
            group_id = self.bucket_to_bucket_group[bucket_id]
            if group_id not in bucket_group_to_buckets:
                bucket_group_to_buckets[group_id] = []
            bucket_group_to_buckets[group_id].append(bucket_id)

        # Coalesce all-gather operations for all buckets in the same data-parallel-group
        for _, buckets in bucket_group_to_buckets.items():
            param_group = parameter_groups[buckets[0]]
            dp_group = param_group.model_weight_buffer.data_parallel_group
            # Coalesce the asynchronous NCCL operations in this context.
            with _coalescing_manager(dp_group, async_ops=True) as coalescing_event:
                for bucket_id in buckets:
                    # All-gather the module weights from each buffer shard into an allocated bucket.
                    self.all_gather_bucket_and_set_items(bucket_id, async_op=True)

                # Replace the parameter all-gather event with coalescing event.
                for bucket_id in buckets:
                    _, mark_bucket_ready_to_use = self.param_gather_event_map[bucket_id]
                    self.param_gather_event_map[bucket_id] = (
                        coalescing_event,
                        mark_bucket_ready_to_use,
                    )

    def wait_bucket_ready(self, bucket_id, empty_ok=False):
        """Wait for the bucket to be ready."""
        if self.bucket_status[bucket_id] == BucketStatus.READY_TO_USE:
            # Already ready to use.
            return
        if self.bucket_status[bucket_id] == BucketStatus.EMPTY:
            if empty_ok:
                return
            # Bucket shouldn't be empty, this implies that the bucket
            # was not allocated or NCCL operations are not complete.
            raise ValueError(f"Bucket {bucket_id} is empty.")

        # Wait for asynchronous / overlapped NCCL operations to complete.
        param_gather_event, mark_bucket_ready_to_use = self.param_gather_event_map.pop(bucket_id)
        param_gather_event.wait()
        mark_bucket_ready_to_use()

    @torch.no_grad()
    def release_bucket(self, bucket_id: int):
        """Release the bucket."""
        if self.bucket_status[bucket_id] == BucketStatus.EMPTY:
            return

        if self.bucket_status[bucket_id] == BucketStatus.COMMUNICATING:
            raise ValueError(f"Bucket {bucket_id} is communicating.")

        wbuf = self.buffer.parameter_groups[bucket_id].model_weight_buffer
        wbuf.free_bucket_storage()
        self.bucket_status[bucket_id] = BucketStatus.EMPTY

    def recycle_unused_buckets(self):
        """Recycle the unused buckets."""
        for bucket_id, can_be_released in self.bucket_can_be_released.items():
            if can_be_released:
                self.release_bucket(bucket_id)
                self.bucket_can_be_released[bucket_id] = False

    @torch.no_grad()
    def all_gather_bucket_and_set_items(self, bucket_id: int, async_op: bool = False) -> None:
        """All-gather the bucket and set the items."""
        self.bucket_can_be_released[bucket_id] = False
        if self.bucket_status[bucket_id] != BucketStatus.EMPTY:
            return

        self.bucket_status[bucket_id] = BucketStatus.COMMUNICATING
        wbuf = self.buffer.parameter_groups[bucket_id].model_weight_buffer

        # Lazy release the unused buckets.
        self.recycle_unused_buckets()
        # Allocate an empty bucket to store the module weights.
        bucket = wbuf.fetch_bucket(and_allocate_params_data=True)
        # All-gather the module weights in each buffer shard into the allocated bucket.
        # Now each rank will have a copy of this FSDP unit module's weights.
        param_gather_event = torch.distributed.all_gather_into_tensor(
            output_tensor=bucket.data,
            input_tensor=wbuf.get_shard_from_local_buffer(),
            group=wbuf.data_parallel_group,
            async_op=async_op,
        )

        def get_closure(bucket_id):
            @torch.no_grad()
            def mark_bucket_ready_to_use():
                # Mark the bucket as ready to use - all NCCL operations are complete.
                self.bucket_status[bucket_id] = BucketStatus.READY_TO_USE

            return mark_bucket_ready_to_use

        mark_bucket_ready_to_use = get_closure(bucket_id)

        if async_op:
            # Track the async all-gather operation for the bucket.
            self.param_gather_event_map[bucket_id] = (param_gather_event, mark_bucket_ready_to_use)
            return
        mark_bucket_ready_to_use()


@torch.no_grad()
def gradient_reduce_preprocessing(grad_data, scaling_factor, ddp_config):
    """
    Gradient reduce preprocessing for gradient averaging and gradient scaling.
    """

    if scaling_factor is None:
        reduce_op = torch.distributed.ReduceOp.SUM
    elif ddp_config.average_in_collective:
        reduce_op = torch.distributed.ReduceOp.AVG
    elif ddp_config.gradient_reduce_div_fusion and grad_data.dtype != torch.bfloat16:
        reduce_op = torch.distributed._make_nccl_premul_sum(scaling_factor)
    else:
        grad_data.mul_(scaling_factor)
        reduce_op = torch.distributed.ReduceOp.SUM

    return reduce_op


def check_gpu_memory(threshold=0.9):
    """
    Check if the GPU memory is over the threshold.
    Args:
        threshold (float, optional): The threshold to check if the GPU memory is over.
            Defaults to 0.9.
    Returns:
        bool: True if the GPU memory is over the threshold.
    """
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory

    allocated_ratio = allocated / total
    reserved_ratio = reserved / total

    near_full = allocated_ratio >= threshold or reserved_ratio >= threshold

    if near_full:
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            f"GPU Memory: Allocated: {allocated_ratio:.2%}, Reserved: {reserved_ratio:.2%}",
        )
    return near_full


class ResetParametersContext:
    """
    Context manager for resetting parameters for meta device initialization module.
    """

    def __init__(self, init_param_with_fp8=False, with_cuda_rng_tracker=False):
        self.init_param_with_fp8 = init_param_with_fp8
        self.with_cuda_rng_tracker = with_cuda_rng_tracker

    def __enter__(self):
        self.stack = ExitStack()
        if self.init_param_with_fp8:
            assert HAVE_TE
            args = {"enabled": True}
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                args["preserve_high_precision_init_val"] = True
            self.stack.enter_context(fp8_model_init(**args))

        if self.with_cuda_rng_tracker:
            self.stack.enter_context(get_cuda_rng_tracker().fork())

        return self

    def __exit__(self, *exc_details):
        self.stack.__exit__(*exc_details)
