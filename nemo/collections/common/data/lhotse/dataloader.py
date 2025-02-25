# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import bisect
import os
import random
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence, TypeVar, Union

import numpy as np
import torch
from lhotse import CutSet, RecordingSet
from lhotse.cut import Cut
from lhotse.cut.text import TextExample, TextPairExample
from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    DynamicCutSampler,
    IterableDatasetWrapper,
    ReverbWithImpulseResponse,
    make_worker_init_fn,
)
from lhotse.dataset.dataloading import resolve_seed
from lhotse.dataset.sampling.base import SamplingConstraint, TimeConstraint, TokenConstraint
from lhotse.dataset.sampling.dynamic_bucketing import FixedBucketBatchSizeConstraint
from lhotse.lazy import LazyFlattener
from lhotse.utils import fastcopy, fix_random_seed
from omegaconf import DictConfig, OmegaConf

from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset, read_cutset_from_config
from nemo.collections.common.prompts.fn import get_prompt_format_fn
from nemo.utils import logging


@dataclass
class LhotseDataLoadingConfig:
    """
    Structured config used for OmegaConf schema validation.
    It's also a single source of truth for reading default option values.
    The options not supported anymore but present, e.g., in old configs,
    will be emitted in a DeprecationWarning and ignored.
    """

    # 1. Data inputs.
    #   a. "Classic" NeMo input path fields.
    input_cfg: Any = None  # TODO(pzelasko): typing
    manifest_filepath: Any = None  # str | list[list[str | float]] | None = None
    tarred_audio_filepaths: Any = None  # str | list[list[str]] | None = None
    #   b. Lhotse CutSet manifest / Lhotse Shar tar dir paths.
    cuts_path: str | None = None
    shar_path: Any = None  # str | list[str | tuple[str, float | int]] | None = None
    #  Enable this to support dataloading from JSON manifests that reference subsets of audio tar files.
    tarred_random_access: bool = False
    # 2. Batch size.
    #   a. Existing NeMo options.
    batch_size: int | None = None
    #   b. Lhotse dynamic batch sizes.
    batch_duration: float | None = None
    quadratic_duration: float | None = None
    #   c. Lhotse bucketing.
    use_bucketing: bool = False
    bucket_batch_size: list[int] | None = None
    num_buckets: int = 30
    num_cuts_for_bins_estimate: int = 10000
    bucket_duration_bins: Any = None  # list[float] | list[list[float]] | None = None
    bucket_buffer_size: int = 10000
    concurrent_bucketing: bool = True  # fetches data in a background thread
    #   d. Other Lhotse sampling options.
    shuffle_buffer_size: int | None = 10000
    drop_last: bool = False
    shard_seed: int | str = "trng"
    max_open_streams: int | None = None
    cuda_expandable_segments: bool = True

    # 2.1 Multimodal sampling override options
    pretokenize: bool = True  # should we apply tokenizer before data sampling
    prompt_format: str | None = None  # when provided, we'll apply the prompt in addition to the tokenizer
    use_multimodal_sampling: bool = False
    token_equivalent_duration: float | None = None
    batch_tokens: int | None = None
    quadratic_factor: float | None = None

    # 3. Supported existing NeMo options.
    shuffle: bool = False
    sample_rate: int = 16000
    min_duration: float | None = -1
    max_duration: float | None = float("inf")
    seed: int | str = 0
    num_workers: int = 0
    pin_memory: bool = False
    channel_selector: int | str | None = None
    min_tps: int = -1  # allowed tokens per second
    max_tps: float = float("inf")

    # 4. Optional Lhotse data augmentation.
    #   a. On-the-fly noise/audio mixing.
    noise_path: Any | None = (
        None  # str | dict where dict can have any of keys: manifest_filepath, tarred_audio_filepaths, cuts_path, shar_path
    )
    noise_snr: tuple[float, float] = (10.0, 20.0)
    noise_mix_prob: float = 0.5
    #   b. On-the-fly 3-way speed perturbation.
    perturb_speed: bool = False
    #   c. Cut concatenation (glue together multiple utterances into a single one)
    concatenate_samples: bool = False
    concatenate_gap_seconds: float = 0.1
    concatenate_duration_factor: float = 1.0
    concatenate_merge_supervisions: bool = True
    db_norm: Optional[float] = -25.0  # from CodeSwitchingDataset
    #   d. On-the-fly cut truncation or window slicing
    #       I) truncate: select one chunk of a fixed duration for each cut
    truncate_duration: Optional[float] = None  # set this to enable
    truncate_offset_type: str = "random"  # "random" | "start" (fixed) | "end" (fixed, counted back)
    #       II) cut_into_windows: convert each cut to smaller cut using a sliding window (define hop for overlapping windows)
    cut_into_windows_duration: Optional[float] = None  # set this to enable
    cut_into_windows_hop: Optional[float] = None
    #       III) common options
    keep_excessive_supervisions: bool = (
        True  # when a cut is truncated in the middle of a supervision, should we keep them.
    )
    #   e. RIR augmentation (synthetic RIR if rir_path is None)
    #   at the moment supports only Lhotse recording manifests, e.g. https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/rir_noise.py
    rir_enabled: bool = False
    rir_path: str | None = None  # str, must point to a lhotse RecordingSet manifest
    rir_prob: float = 0.5

    # 5. Other Lhotse options.
    text_field: str = "text"  # key to read the transcript from
    lang_field: str = "lang"  # key to read the language tag from
    # Enables iteration of NeMo non-tarred manifests that don't have a "sampling_rate" key without performing any I/O.
    # Note that this will not allow actual dataloading; it's only for manifest iteration as Lhotse objects.
    metadata_only: bool = False
    # Forces the resulting CutSet to be finite, so that the iteration will end after a full single epoch.
    # Do not turn this on unless you're sure that you know what you're doing.
    # In most cases (such as regular multi-GPU training) it will result in a deadlock due to
    # a different number of steps on different DDP ranks.
    force_finite: bool = False


def get_lhotse_dataloader_from_config(
    config: DictConfig,
    global_rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
    tokenizer=None,
) -> torch.utils.data.DataLoader:
    """
    Set up a Lhotse training dataloder.

    Expects a typical NeMo dataset configuration format, with additional fields: "use_lhotse=True".
    Some fields in the original NeMo configuration may be ignored.

    The ``dataset`` parameter should be an instance of a Lhotse-compatible PyTorch Dataset class.
    It only needs to define the following method ``__getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]``.
    This dataset is not expected to hold a reference to any actual data; it may be interpreted as a function
    mapping a Lhotse CutSet into a mini-batch of tensors.

    For an example, see: :class:`nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`,
    which is constructed from just a tokenizer and essentially loads and collates audio and tokenizes the transcript.

    The ``tokenizer`` is used both for audio and text datasets for on-the-fly tokenization.
    This allows us to stratify the bucketing by the count of input/output tokens (depending on modality).
    If "prompt_format" is additionally provided in the config, we will also apply a prompt formatter.
    Note that ``tokenizer`` can be any tokenizer type (e.g. both SentencePiece and Aggregate tokenizers work).
    """
    logging.info("We will be using a Lhotse DataLoader.")

    config = make_structured_with_schema_warnings(config)

    maybe_set_cuda_expandable_segments(enabled=config.cuda_expandable_segments)

    # First, resolve the random seed in case a string value was provided.
    seed = resolve_seed(config.seed)
    fix_random_seed(seed)

    # 1. Load a manifest as a Lhotse CutSet.
    cuts, is_tarred = read_cutset_from_config(config)

    # Apply channel selector
    if config.channel_selector is not None:
        logging.info('Using channel selector %s.', config.channel_selector)
        cuts = cuts.map(partial(_select_channel, channel_selector=config.channel_selector))

    # Resample as a safeguard; it's a no-op when SR is already OK
    cuts = cuts.resample(config.sample_rate)

    # Expands cuts if multiple translations are provided.
    cuts = CutSet(LazyFlattener(cuts.map(_flatten_alt_text, apply_fn=None)))

    if tokenizer is not None and config.pretokenize:
        from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper

        if not is_tarred:
            logging.warning(
                "You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). "
                "This will cause the tokenization to happen in the main (GPU) process, possibly impacting the training speed "
                "if your tokenizer is very large. If the impact is noticable, set pretokenize=False in dataloader config. "
                "(note: that will disable token-per-second filtering and 2D bucketing features)"
            )

        if config.prompt_format is not None:
            cuts = cuts.map(
                partial(tokenize_with_prompt, tokenizer=tokenizer, prompt_format=config.prompt_format), apply_fn=None
            )
        else:
            if not isinstance(tokenizer, TokenizerWrapper):
                tokenizer = TokenizerWrapper(tokenizer)
            cuts = cuts.map(partial(tokenize, tokenizer=tokenizer), apply_fn=None)
        cuts = cuts.filter(TokenPerSecondFilter(config.min_tps, config.max_tps))

    # 2. Optional augmentations.
    # 2.a. Noise mixing.
    if config.noise_path is not None:
        noise = guess_parse_cutset(config.noise_path)
        cuts = cuts.mix(
            cuts=noise,
            snr=tuple(config.noise_snr),
            mix_prob=config.noise_mix_prob,
            seed=config.shard_seed,
            random_mix_offset=True,
        )

    # 2.b. On-the-fly speed perturbation.
    #    mux here ensures it's uniformly distributed throughout sampling,
    #    and applying it here (before sampler/dataset) ensures optimal
    #    bucket allocation.
    if config.perturb_speed:
        cuts = CutSet.mux(
            cuts,
            cuts.perturb_speed(0.9),
            cuts.perturb_speed(1.1),
        )

    # 2.d: truncation/slicing
    if config.truncate_duration is not None:
        cuts = cuts.truncate(
            max_duration=config.truncate_duration,
            offset_type=config.truncate_offset_type,
            keep_excessive_supervisions=config.keep_excessive_supervisions,
        )
    if config.cut_into_windows_duration is not None:
        cuts = cuts.cut_into_windows(
            duration=config.cut_into_windows_duration,
            hop=config.cut_into_windows_hop,
            keep_excessive_supervisions=config.keep_excessive_supervisions,
        )

    # Duration filtering, same as native NeMo dataloaders.
    # We can filter after the augmentations because they are applied only when calling load_audio().
    cuts = cuts.filter(DurationFilter(config.min_duration, config.max_duration))

    bucket_duration_bins = determine_bucket_duration_bins(config)
    if config.use_multimodal_sampling:
        if config.bucket_batch_size is not None:
            assert (
                bucket_duration_bins is not None
            ), "Cannot use bucket_batch_size option if bucket_duration_bins are not provided."
            constraint = MultimodalFixedBucketBatchSizeConstraint2D(
                max_seq_len_buckets=bucket_duration_bins,
                batch_sizes=config.bucket_batch_size,
                token_equivalent_duration=config.token_equivalent_duration,
            )
        else:
            constraint = MultimodalSamplingConstraint(
                token_equivalent_duration=config.token_equivalent_duration,
                batch_size=config.batch_size,
                batch_tokens=config.batch_tokens,
                quadratic_factor=config.quadratic_factor,
            )
    else:
        if config.bucket_batch_size is not None:
            assert (
                bucket_duration_bins is not None
            ), "Cannot use bucket_batch_size option if bucket_duration_bins are not provided."
            constraint = FixedBucketBatchSizeConstraint2D(
                max_seq_len_buckets=bucket_duration_bins,
                batch_sizes=config.bucket_batch_size,
            )
        else:
            constraint = TimeConstraint(
                max_cuts=config.batch_size,
                max_duration=config.batch_duration,
                quadratic_duration=config.quadratic_duration,
            )

    # 3. The sampler.
    if config.use_bucketing:
        # Bucketing. Some differences from NeMo's native bucketing:
        #    - we can tweak the number of buckets and bucket duration bins using the configuration
        #    - batch size is dynamic and configurable via a single param: max_duration (config: batch_duration)
        #    - quadratic_duration introduces a penalty to balance batch sizes for quadratic time complexity models
        logging.info(
            f"Creating a Lhotse DynamicBucketingSampler "
            f"(max_batch_duration={config.batch_duration} max_batch_size={config.batch_size})"
        )
        # Determine the bucket duration bins
        sampler = DynamicBucketingSampler(
            cuts,
            constraint=constraint,
            shuffle=config.shuffle,
            drop_last=config.drop_last,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=config.shard_seed,
            num_buckets=config.num_buckets,
            duration_bins=determine_bucket_duration_bins(config),
            num_cuts_for_bins_estimate=config.num_cuts_for_bins_estimate,
            buffer_size=config.bucket_buffer_size,
            rank=0 if is_tarred else global_rank,
            world_size=1 if is_tarred else world_size,
        )
    else:
        # Non-bucketing sampler, similar to original NeMo dataloading without bucketing,
        # but we also use batch_duration instead of batch_size here.
        # Recommended for dev/test.
        logging.info(
            f"Creating a Lhotse DynamicCutSampler (bucketing is disabled, "
            f"(max_batch_duration={config.batch_duration} max_batch_size={config.batch_size})"
        )
        sampler = DynamicCutSampler(
            cuts,
            constraint=constraint,
            shuffle=config.shuffle,
            drop_last=config.drop_last,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=config.shard_seed,
            rank=0 if is_tarred else global_rank,
            world_size=1 if is_tarred else world_size,
        )

    if config.concatenate_samples:
        # Cut concatenation will produce longer samples out of shorter samples
        # by gluing them together from the shortest to longest not to exceed a duration
        # of longest_cut * duration_factor (greedy knapsack algorithm for minimizing padding).
        # Useful e.g. for simulated code-switching in multilingual setups.
        # We follow concatenation by ``merge_supervisions`` which creates a single supervision
        # object with texts joined by a whitespace so that "regular" dataset classes don't
        # have to add a special support for multi-supervision cuts.
        sampler = sampler.map(
            CutConcatenate(
                gap=config.concatenate_gap_seconds,
                duration_factor=config.concatenate_duration_factor,
            )
        )
        if config.db_norm is not None:
            sampler = sampler.map(partial(_normalize_loudness, db_norm=config.db_norm))
        if config.concatenate_merge_supervisions:
            sampler = sampler.map(_merge_supervisions)

    if config.rir_enabled:
        sampler = sampler.map(
            ReverbWithImpulseResponse(
                rir_recordings=RecordingSet.from_file(config.rir_path) if config.rir_path is not None else None,
                p=config.rir_prob,
                randgen=random.Random(seed),
            )
        )

    # 4. Creating dataloader.
    if is_tarred and not config.tarred_random_access:
        # Wrapper here is necessary when using NeMo tarred data or Lhotse Shar data,
        # because then I/O happens upon sampler iteration. Normally, the sampler resides
        # in the training loop process, but when we use iterable dataset, we can move it to
        # the dataloading worker process.
        # We use lhotse's own worker_init_fn which leverages information such as rank, world_size,
        # worker_id, etc. to set a different random seed for each (node, worker) combination.
        # This together with infinite datasets removes the need to split data across nodes/workers.
        dloader_kwargs = dict(
            dataset=IterableDatasetWrapper(dataset=dataset, sampler=sampler),
            worker_init_fn=make_worker_init_fn(rank=global_rank, world_size=world_size, seed=seed),
            persistent_workers=config.num_workers > 0,  # helps Lhotse Shar maintain shuffling state
        )
    else:
        # For non-tarred data, the sampler resides in the training loop process and
        # reads only light-weight JSON objects; it samples mini-batches and passes
        # the meta-data to Dataset, which performs the actual I/O inside its __getitem__ method.
        dloader_kwargs = dict(dataset=dataset, sampler=sampler)
    dloader = torch.utils.data.DataLoader(
        **dloader_kwargs,
        batch_size=None,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return dloader


def determine_bucket_duration_bins(config):
    if config.bucket_duration_bins is not None:
        # Bucket duration bins are provided: just use them.
        ans = OmegaConf.to_container(config.bucket_duration_bins)
        if isinstance(ans[0], Sequence):
            # 2D bucketing. Ensure we're using tuples for correct behaviour of '<' operator
            # between the bucket bin tuples and the output of measure_length.
            ans = [tuple(item) for item in ans]
        return ans
    # Bucket duration bins are not set.
    if config.use_multimodal_sampling:
        # For multimodal sampling it's currently impossible to define a linspace over durations
        # because the buckets are counted in the number of tokens.
        # The bins will be auto-estimated by lhotse at the cost of a slight lag in the training start.
        return None
    elif config.max_duration is not None and config.max_duration < float("inf"):
        # If max duration is provided, we can use that to compute uniformly distant bucket bins.
        # This is not optimal but should be close enough for users who didn't want to estimate these up-front.
        begin = config.min_duration if config.min_duration is not None and config.min_duration > 0 else 0.0
        end = config.max_duration
        return np.linspace(begin, end, config.num_buckets + 1)[1:-1].tolist()
    else:
        # If we don't know max_duration, we can't guess a reasonable estimate of the upper bound of
        # durations.
        # The bins will be auto-estimated by lhotse at the cost of a slight lag in the training start.
        return None


def make_structured_with_schema_warnings(config: DictConfig) -> DictConfig:
    """
    Checks the schema and fills missing default option values.
    Warns the user if any of the fields are not supported by the current schema
    but does not raise exceptions.
    """
    default = OmegaConf.structured(LhotseDataLoadingConfig)

    # Remove unsupported keys and warn about them.
    supported_keys = set(OmegaConf.to_container(default).keys())
    received_keys = set(OmegaConf.to_container(config).keys())
    unsupported_keys = received_keys - supported_keys
    if unsupported_keys:
        warnings.warn(
            f"The following configuration keys are no longer supported " f"and ignored: {','.join(unsupported_keys)}",
            category=DeprecationWarning,
        )
    config = OmegaConf.masked_copy(config, list(supported_keys))

    return OmegaConf.merge(default, config)


@dataclass
class MultimodalSamplingConstraint(SamplingConstraint):
    # how many seconds of audio is a text token worth; balances audio to text ratio in a mini-batch
    token_equivalent_duration: float

    # defines maximum batch size (may be lower than that if batch_length is also specified)
    batch_size: int | None = None

    # defines the total number of tokens in a mini-batch
    # setting this enables dynamic batch sizes
    # we will use ``token_equivalent_duration`` to convert audio examples to token sizes
    batch_tokens: int | None = None

    # when specified, this value is inversely proportional to the penalty we assign
    # to longer examples when measuring their length/duration;
    # i.e. large quadratic factor is a small penalty, small quadratic factor is a large penalty
    # tweaking this helps equalize the GPU memory usage for dynamic batch sizes when using bucketing
    quadratic_factor: float | None = None

    _internal = None

    def __post_init__(self):
        self._internal = TokenConstraint(
            max_tokens=self.batch_tokens,
            max_examples=self.batch_size,
            quadratic_length=self.quadratic_factor,
        )

    def add(self, example: Any) -> None:
        if isinstance(example, Cut):
            num_tokens = self.measure_length(example)
            example.num_tokens = num_tokens
        self._internal.add(example)

    def exceeded(self) -> bool:
        return self._internal.exceeded()

    def close_to_exceeding(self) -> bool:
        return self._internal.close_to_exceeding()

    def reset(self) -> None:
        self._internal.reset()

    def measure_length(self, example: Any) -> float:
        if isinstance(example, Cut):
            return example.duration / self.token_equivalent_duration
        if isinstance(example, (TextExample, TextPairExample)):
            return example.num_tokens
        raise RuntimeError(f"Unsupported example type: {type(example)}")


@dataclass
class FixedBucketBatchSizeConstraint2D(FixedBucketBatchSizeConstraint):
    @property
    def bucketing_2d_enabled(self) -> bool:
        return isinstance(self.max_seq_len_buckets[0], Sequence) and len(self.max_seq_len_buckets[0]) == 2

    def measure_length(self, example: Any) -> tuple[float, float]:
        if self.bucketing_2d_enabled:
            return example.duration, _measure_tokens(example)
        else:
            return example.duration

    def select_bucket(self, buckets: Any, example: Any = None, example_len: Any = None) -> int:
        if not self.bucketing_2d_enabled:
            return super().select_bucket(buckets=buckets, example=example, example_len=example_len)
        if example_len is None:
            example_len = self.measure_length(example)
        bucket_idx = bisect.bisect_right(buckets, example_len)
        # For 2D bucketing we have to refine the initially found bucket_idx, as bisect
        # looks primarily at the first index of a tuple (i.e. duration).
        # For example, with buckets [(1, 1), (1, 2), (2, 2), (2, 4)] and example (1.5, 3)
        # bisect would allocate it to bucket_idx=2 instead of bucket_idx=3.
        # To refine, we'll try to push the example to as many buckets to the right as possible,
        # as long as they have the same dim0 length (e.g. audio duration) and the example's dim1
        # is smaller than the bin's dim1 (e.g., output token sequence length).
        bin_dim0, bin_dim1 = self.max_seq_len_buckets[bucket_idx]
        num_buckets = len(self.max_seq_len_buckets)
        while (
            (next_idx := bucket_idx + 1) < num_buckets  # There is a next bucket
            and (bin := self.max_seq_len_buckets[next_idx])[0] == bin_dim0  # The next bucket has the same 1st dim.
            # The example's 2nd dim is between that of the current and the next bucket; or,
            # the next bucket's 2nd dim is still smaller than example.
            and (bin_dim1 < example_len[1] <= bin[1] or bin[1] < example_len[1])
        ):
            bucket_idx = next_idx
            bin_dim0, bin_dim1 = self.max_seq_len_buckets[bucket_idx]
        return bucket_idx


@dataclass
class MultimodalFixedBucketBatchSizeConstraint2D(FixedBucketBatchSizeConstraint2D):
    token_equivalent_duration: float | None = None

    def measure_length(self, example: Any) -> float:
        assert not self.bucketing_2d_enabled, "2D bucketing for multimodal sampling is not yet supported."
        if hasattr(example, "num_tokens"):
            return example.num_tokens
        if isinstance(example, Cut):
            assert (
                self.token_equivalent_duration is not None
            ), "Cannot use MultimodalFixedBucketBatchSizeConstraint with speech data when token_equivalent_duration was not specified."
            return example.duration / self.token_equivalent_duration
        raise RuntimeError(f"Unsupported example type: {type(example)}")


def is_text(example) -> bool:
    return isinstance(example, (TextExample, TextPairExample))


Example = TypeVar("Example", bound=Union[Cut, TextExample, TextPairExample])


def tokenize(example: Example, tokenizer) -> Example:
    if isinstance(example, Cut):
        for s in example.supervisions:
            if s.text is not None:
                s.tokens = np.asarray(tokenizer(s.text, s.language))
    elif isinstance(example, TextExample):
        example.tokens = np.asarray(tokenizer(example.text, example.language))
    elif isinstance(example, TextPairExample):
        example.source.tokens = np.asarray(tokenizer(example.source.text, example.source.language))
        example.target.tokens = np.asarray(tokenizer(example.source.text, example.target.language))
    else:
        raise RuntimeError(f"Unsupported type of example: {type(example)}")
    return example


def tokenize_with_prompt(example: Example, tokenizer, prompt_format: str) -> Example:
    # TODO(pzelasko): This mechanism makes it possible to measure the actual output sequence length
    #   for prompted models such as AED MultiTask (Canary), which includes the transcript and the prompt.
    #   We intend to extend it for text modality in follow-up work.
    if isinstance(example, Cut):
        prompt_format_fn = get_prompt_format_fn(prompt_format)
        (tokenized_prompted_transcript,), (tokenized_prompt,), (tokenized_transcript,) = prompt_format_fn(
            CutSet([example]), tokenizer
        )
        example.tokenized_prompted_transcript = tokenized_prompted_transcript
        example.tokenized_prompt = tokenized_prompt
        example.tokenized_transcript = tokenized_transcript
    else:
        raise RuntimeError(f"Currently we only support tokenization + prompting during sampling for audio modality.")
    return example


# The helper callables below exist to avoid passing lambdas into lhotse CutSet map/filter methods.
# Lambdas are not serializable across processes by pickle.
# Note: lhotse offers LHOTSE_DILL_ENABLED=1 and ``lhotse.lazy.set_dill_enabled(True)``
# to support pickling lambdas if its ever truly necessary.


class DurationFilter:
    """Callable, returns ``True`` if a cut's duration is in range [d_min, d_max] and ``False`` otherwise."""

    def __init__(self, d_min: float, d_max: float) -> None:
        self.d_min = d_min
        self.d_max = d_max

    def __call__(self, example) -> bool:
        if isinstance(example, Cut):
            return self.d_min <= example.duration <= self.d_max
        else:
            return True  # does not apply to text etc.


class TokenPerSecondFilter:
    """
    Callable, returns ``True`` if a cut's num_tokens (sum of len(tokens) for each supervision)
    is in range [tps_min, tps_max] and ``False`` otherwise.
    """

    def __init__(self, tps_min: float, tps_max: float) -> None:
        assert tps_min <= tps_max
        self.tps_min = tps_min
        self.tps_max = tps_max
        self.enabled = tps_min > 0 or tps_max < float("inf")

    def __call__(self, example) -> bool:
        if not isinstance(example, Cut) or not self.enabled:
            return True  # pass-through for non-audio examples.
        tps = _measure_tps(example)
        return self.tps_min <= tps <= self.tps_max


def _measure_tokens(cut: Cut) -> int:
    if hasattr(cut, "tokenized_prompted_transcript"):
        return len(cut.tokenized_prompted_transcript)  # tokenized with prompt formatter
    supervisions_with_tokens = [s for s in cut.supervisions if hasattr(s, "tokens")]
    assert len(supervisions_with_tokens) > 0, (
        "Cannot measure tokens-per-second with untokenized supervisions. "
        "Did you forget to provide the tokenizer argument to get_lhotse_dataloader_from_config() method?"
    )
    return sum(len(s.tokens) for s in supervisions_with_tokens)


def _measure_tps(cut: Cut) -> float:
    num_tokens = _measure_tokens(cut)
    return num_tokens / cut.duration


def _normalize_loudness(cuts: CutSet, db_norm: float) -> CutSet:
    return cuts.normalize_loudness(target=db_norm, mix_first=False)


def _merge_supervisions(cuts: CutSet) -> CutSet:
    return cuts.merge_supervisions()


def _flatten_alt_text(cut) -> list:
    ans = [cut]
    if not isinstance(cut, Cut) or cut.custom is None or cut.custom.get("alt_text") is None:
        return ans
    cut = cut.move_to_memory(audio_format="wav")  # performs I/O once and holds audio in memory from now on
    # Popping to ease eyesight on debug.
    paired_text = cut.custom.pop("alt_text")
    for data in paired_text.values():
        # Copy to avoid lazy dataloading issues
        data = data.copy()
        text_instance = cut.map_supervisions(lambda s: fastcopy(s, text=data["text"], language=data["lang"]))
        text_instance.custom = {"text": data.pop("text"), "lang": data.pop("lang"), **data}
        ans.append(text_instance)
    return ans


def maybe_set_cuda_expandable_segments(enabled: bool):
    """
    Configures PyTorch memory allocator to expand existing allocated segments
    instead of re-allocating them when tensor shape grows.
    This can help speed up the training when sequence length and/or batch size change often,
    and makes GPU more robust towards OOM.

    See here for more details:
    https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
    """
    if enabled and torch.cuda.is_available():
        if (
            (value := os.environ.get("PYTORCH_CUDA_ALLOC_CONF")) is not None
            and len(value) > 0
            and "expandable_segments:True" not in value
        ):
            warnings.warn(
                "You have set PYTORCH_CUDA_ALLOC_CONF without expandable_segments:True option. We're setting that option anyway. To disable it, set cuda_expandable_segments=False in NeMo dataloader configuration."
            )

        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        except RuntimeError:
            logging.info(
                "Failed to set expandable_segments:True for PyTorch CUDA allocator. You may get training speed improvements if you enable this"
            )


def _select_channel(cut, channel_selector: int | str) -> list:
    if isinstance(channel_selector, int):
        channel_idx = channel_selector
    elif isinstance(channel_selector, str):
        if channel_selector in cut.custom:
            channel_idx = cut.custom[channel_selector]
        else:
            raise ValueError(f"Channel selector {channel_selector} not found in cut.custom")

    if channel_idx >= cut.num_channels:
        raise ValueError(
            f"Channel index {channel_idx} is larger than the actual number of channels {cut.num_channels}"
        )

    if cut.num_channels == 1:
        # one channel available and channel_idx==0
        return cut
    else:
        # with_channels only defined on MultiCut
        return cut.with_channels(channel_idx)
