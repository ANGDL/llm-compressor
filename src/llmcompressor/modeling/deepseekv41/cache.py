"""Request-owned V4.1 state, stored on CPU between attention calls."""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache


class CompressedState:
    """Opaque handle: Accelerate must not recursively move all source history."""

    def __init__(self, state):
        self.state = state


class DeepseekV41Cache(Cache):
    """SWA tails, source-owned CSA2 state and bounded Engram token history.

    Only source layers own compressed KV/index K. Consumer layers receive their
    source through the current forward's shared tuple, without cache copies.
    The initial implementation supports unpadded batch=1, append-only requests.
    """

    def __init__(self, config):
        super().__init__(layers=[])
        self.config = config
        self.states: list[dict[str, torch.Tensor]] = [
            {} for _ in range(config.n_layers)
        ]
        self.sources = {
            index: CompressedState(self.states[index])
            for index in config.kv_source_layer_ids
        }
        self.seen_tokens = 0
        self.history_ids = None
        self.history_valid = None
        self._streaming_failed = False
        self._active = False

    def get_seq_length(self, layer_idx=0):
        return self.seen_tokens

    def get_max_length(self, layer_idx=0):
        return self.config.max_seq_len

    @property
    def is_croppable(self):
        return False

    @property
    def batch_size(self):
        return 1

    def begin(self, input_ids, mask):
        if self._streaming_failed or self._active:
            raise RuntimeError("V4.1 cache is incomplete; reset and prefill again")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("V4.1 cached inference requires nonempty batch=1 inputs")
        if mask.shape != input_ids.shape or not mask.bool().all():
            raise ValueError("V4.1 cached inference currently requires unpadded inputs")
        if self.seen_tokens + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("V4.1 cache exceeds max_seq_len")
        self._active = True

    def commit(self, input_ids, valid):
        keep = max(0, self.config.engram_max_ngram_size - 1)
        ids, mask = input_ids.detach().cpu(), valid.detach().cpu()
        if self.history_ids is not None:
            ids = torch.cat((self.history_ids, ids), dim=1)
            mask = torch.cat((self.history_valid, mask), dim=1)
        self.history_ids = ids[:, -keep:].clone() if keep else None
        self.history_valid = mask[:, -keep:].clone() if keep else None
        self.seen_tokens += input_ids.shape[1]
        self._active = False

    def reset(self):
        self.states = [{} for _ in self.states]
        self.sources = {
            index: CompressedState(self.states[index])
            for index in self.config.kv_source_layer_ids
        }
        self.seen_tokens = 0
        self.history_ids = self.history_valid = None
        self._active = self._streaming_failed = False

    def reorder_cache(self, beam_idx):
        if beam_idx.numel() != 1 or int(beam_idx[0]) != 0:
            raise ValueError("V4.1 cached inference does not yet support beam search")

    def batch_repeat_interleave(self, repeats):
        if repeats != 1:
            raise ValueError("V4.1 cached inference supports batch=1 only")

    def crop(self, max_length):
        if max_length == 0:
            self.reset()
        else:
            raise ValueError("V4.1 cache rollback is not yet supported")

    @staticmethod
    def append(state, name, value):
        value = value.detach().to("cpu")
        previous = state.get(name)
        state[name] = (
            value.clone() if previous is None else torch.cat((previous, value), dim=1)
        )
        return state[name]
