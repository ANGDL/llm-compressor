import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Empty, Queue
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.offload.dist_utils import as_broadcastable, is_distributed
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_args import ActivationOrdering
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    getattr_chain,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr
from torch import distributed as dist

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
    quantize_weight,
)
from llmcompressor.modifiers.quantization.calibration import update_weight_global_scale
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils import greedy_bin_packing, wait_for_comms
from llmcompressor.utils.metric_logging import CompressionLogger

__all__ = ["GPTQModifier"]

_GPTQ_Q_PARAMS = ["weight", "weight_scale", "weight_zero_point", "weight_g_idx"]


class GPTQModifier(Modifier, QuantizationMixin):
    """
    Implements the GPTQ algorithm from https://arxiv.org/abs/2210.17323. This modifier
    uses activations to calibrate a hessian matrix, which is then used to determine
    optimal quantization values and orderings for the model weights.

    Sample yaml:

    ```yaml
    test_stage:
      obcq_modifiers:
        GPTQModifier:
          block_size: 128
          dampening_frac: 0.001
          offload_hessians: False
          actorder: static
          config_groups:
            group_0:
              targets:
                - "Linear"
              input_activations: null
              output_activations: null
              weights:
                num_bits: 8
                type: "int"
                symmetric: true
                strategy: group
                group_size: 128
    ```

    Lifecycle:

    - on_initialize
        - apply config to model
    - on_start
        - add activation calibration hooks
        - add gptq weight calibration hooks
    - on_sequential_epoch_end
        - quantize_weight
    - on_finalize
        - remove_hooks()
        - model.apply(freeze_module_quantization)

    :param sequential_targets: list of layer names to compress during GPTQ, or
        '__ALL__' to compress every layer in the model
    :param block_size: Used to determine number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param actorder: order in which weight columns are quantized. Defaults to "static"
        activation ordering, which achieves best accuracy recovery with no runtime cost.
        For more information, see https://github.com/vllm-project/vllm/pull/8135
    :param offload_hessians: Set to True for decreased memory usage but increased
        runtime.

    :param config_groups: dictionary specifying quantization schemes to apply to target
        modules. Modules not matching a scheme target will NOT be quantized.
    :param targets: list of layer names to quantize if a scheme is provided. Defaults
        to Linear layers
    :param ignore: optional list of module class names or submodule names to not
        quantize even if they match a target in config_groups. Defaults to empty list.
    :param scheme: a single quantization scheme to apply to the model. This is a
        dictionary that supports all keys from QuantizationScheme except targets, which
        will be set to the targets parameter set at the modifier level. Can also be set
        to a dictionary of the format `preset_scheme_name: targets` for example:
        `W8A8: ['Linear']` for weight and activation 8-bit.
    :param kv_cache_scheme: optional QuantizationArgs, that specify the
        quantization of the kv cache. If None, kv cache is not quantized.
        When applying kv cache quantization to transformer AutoModelForCausalLM,
        the kv_cache_scheme gets converted into a QuantizationScheme that:
            - targets the `q_proj` and `k_proj` modules of the model. The outputs
              of those modules are the keys and values that might be cached
            - quantizes the outputs of the aforementioned layers, so that
              keys and values are compressed before storing them in the cache
        There is an explicit assumption that the model contains modules with
        `k_proj` and `v_proj` in their names. If this is not the case
        and kv_cache_scheme != None, the quantization of kv cache will fail
    :param muti_gpu_compression: Whether to use multiple GPUs to compress the model in parallel. 
        Only has an effect if more than 1 GPU is available.
    """

    # gptq modifier arguments
    sequential_targets: Union[str, List[str], None] = None
    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    # TODO: this does not serialize / will be incorrectly written
    actorder: Optional[Union[ActivationOrdering, Sentinel]] = Sentinel("static")
    offload_hessians: bool = False
    muti_gpu_compression: bool = False

    # private variables
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(
        default_factory=dict
    )

    def resolve_quantization_config(self) -> QuantizationConfig:
        config = super().resolve_quantization_config()

        def resolve_actorder(existing):
            # sentinel default only overrides if existing is None
            if self.actorder == Sentinel("static"):
                return ActivationOrdering.STATIC if existing is None else existing

            # user-provided value always attempts to override
            if existing is None or self.actorder == existing:
                return self.actorder

            # if existing provided and conflicts
            raise ValueError(
                "Cannot resolve activation ordering when both "
                "`GPTQModifier.actorder` and `QuantizationScheme.actorder` "
                f"are provided and differ ({self.actorder}, {existing}). "
                "Either unset `GPTQModifier.actorder` or "
                "remove `actorder` from config groups."
            )

        for scheme in config.config_groups.values():
            assert isinstance(scheme, QuantizationScheme)
            if (
                getattr_chain(scheme, "weights.strategy", None)
                == QuantizationStrategy.GROUP
            ):
                scheme.weights.actorder = resolve_actorder(scheme.weights.actorder)
        return config

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the GPTQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        # apply config to model and prepare calibration hooks
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        # prepare module names
        self._module_names = {
            m: name
            for name, m in match_named_modules(
                state.model, self.resolved_targets, self.ignore
            )
        }

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # register quantization calibration hooks
        # assume quantization has been initialized by this modifier or one before it
        QuantizationMixin.start_calibration(self, state.model)

        # register gptq hooks
        added_hook = False

        named_modules = list(
            match_named_modules(state.model, self.resolved_targets, self.ignore)
        )

        for _, module in named_modules:
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                # HACK: previously, embeddings were not quantized because they were not
                # accessible by the layer compressor. For now, we manually ignore it,
                # but in the FUTURE this should be ignored by the user
                if not isinstance(module, torch.nn.Embedding):
                    self.register_hook(module, self.calibrate_module, "forward")
                    added_hook = True

        # Optionally generate global scales if using TENSOR_GROUP quantization
        for _, module in named_modules:
            update_weight_global_scale(module)

        for module in state.model.modules():
            update_fused_layer_weight_global_scales(module)

        if not added_hook:
            raise ValueError(
                "GPTQModifier requires a weight quantization config be specified by "
                "this modifier or a modifier preceding it"
            )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self.compress_modules()

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self.compress_modules()

            if not self.ended_:
                self.on_end(state, None)

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        """
        Calibration hook used to accumulate the hessian of the input to the module

        :param module: module being calibrated
        :param args: inputs to the module, the first element of which is the
            canonical input
        :param _output: uncompressed module output, unused
        """
        # Assume that first argument is the input
        inp = args[0]

        # Initialize hessian if not present
        if module not in self._num_samples:
            init_device = (
                "cpu" if self.offload_hessians else get_execution_device(module)
            )
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = torch.zeros(
                tuple(), device=get_execution_device(module)
            )

        # Accumulate hessian with input with optional offloading
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

    def compress_modules(self):
        """
        Quantize modules which have been calibrated
        """
        ### Not Distributed
        if not is_distributed():
            module_list = list(self._num_samples.keys())
            if self.muti_gpu_compression and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                self.compress_module_list_muti_gpu(module_list)
            else:
                self.compress_module_list(module_list)
            return

        ### Distributed
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Assign modules to ranks
        module_list, rank_to_modules, module_to_rank = greedy_bin_packing(
            list(self._hessians.keys()),
            world_size,
            item_weight_fn=lambda mod: self._hessians[mod].shape[0],
        )

        # send hessians to assigned ranks
        self._reduce_hessian_to_target_rank(module_list, module_to_rank)

        self.compress_module_list(rank_to_modules[rank])

        # broadcast compressed modules to each rank
        self._broadcast_quantized_params(module_list, module_to_rank)

    def compress_module_list_muti_gpu(self, module_list):
        n_gpus = torch.cuda.device_count()

        if n_gpus <= 1 or len(module_list) <= 1:
            self.compress_module_list(module_list)
            return

        module_payloads: List[Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]] = []
        for module in module_list:
            module_payloads.append(
                (
                    module,
                    self._hessians.pop(module),
                    self._num_samples.pop(module),
                )
            )

        # Prioritize larger modules to reduce straggler effects across workers.
        module_payloads.sort(key=lambda payload: payload[1].shape[0], reverse=True)

        active_devices = min(n_gpus, len(module_payloads))
        devices = [torch.device(f"cuda:{gpu_idx}") for gpu_idx in range(active_devices)]

        logger.info(
            "Running GPTQ compression in parallel across "
            f"{active_devices} devices with async task consumption"
        )

        self._warmup_cholesky_on_devices(devices)

        # Per-device queues preserve affinity while still allowing work stealing.
        per_device_queues: Dict[torch.device, Queue] = {
            device: Queue() for device in devices
        }
        assigned_work: Dict[torch.device, int] = {device: 0 for device in devices}

        for payload in module_payloads:
            module, hessian, _ = payload
            module_work = int(hessian.shape[0])
            original_device = get_execution_device(module)
            preferred_device: Optional[torch.device] = None

            # Prefer the module's original CUDA device to minimize weight movement.
            if (
                isinstance(original_device, torch.device)
                and original_device.type == "cuda"
                and original_device.index is not None
            ):
                candidate_device = torch.device(f"cuda:{original_device.index}")
                if candidate_device in per_device_queues:
                    preferred_device = candidate_device

            # Fallback to least-assigned worker by estimated hessian work.
            if preferred_device is None:
                preferred_device = min(devices, key=lambda device: assigned_work[device])

            per_device_queues[preferred_device].put(payload)
            assigned_work[preferred_device] += module_work

        def _pop_next_payload(device: torch.device, allow_steal: bool = True):
            try:
                # Fast path: consume local queue first to keep affinity.
                payload = per_device_queues[device].get_nowait()
                return payload, per_device_queues[device]
            except Empty:
                pass

            if not allow_steal:
                return None, None

            # Slow path: steal from peers to avoid idle workers.
            for steal_device in devices:
                if steal_device == device:
                    continue
                try:
                    payload = per_device_queues[steal_device].get_nowait()
                    return payload, per_device_queues[steal_device]
                except Empty:
                    continue

            return None, None

        def _schedule_payload_for_device(payload_and_queue, device, prefetch_stream):
            payload, source_queue = payload_and_queue
            module, hessian, num_samples = payload
            module_work = int(hessian.shape[0])
            original_device = get_execution_device(module)
            ready_event = None

            if device.type == "cuda" and prefetch_stream is not None:
                # Prefetch tensors on a separate stream so copy can overlap compute.
                with torch.cuda.stream(prefetch_stream):
                    hessian = hessian.to(device=device, non_blocking=True)
                    num_samples = num_samples.to(device=device, non_blocking=True)
                    # Record readiness for this specific job.
                    ready_event = torch.cuda.Event()
                    ready_event.record(prefetch_stream)
            else:
                hessian = hessian.to(device=device)
                num_samples = num_samples.to(device=device)

            return {
                "module": module,
                "hessian": hessian,
                "num_samples": num_samples,
                "module_work": module_work,
                "original_device": original_device,
                "source_queue": source_queue,
                "ready_event": ready_event,
            }

        def _compress_device_modules(device: torch.device):
            if device.type == "cuda":
                torch.cuda.set_device(device)

            # Keep prefetch and compute streams separate for overlap.
            prefetch_stream = (
                torch.cuda.Stream(device=device) if device.type == "cuda" else None
            )
            compute_stream = (
                torch.cuda.Stream(device=device) if device.type == "cuda" else None
            )

            worker_start = perf_counter()
            processed_modules = 0
            processed_work = 0

            first_payload, first_queue = _pop_next_payload(device, allow_steal=True)
            if first_payload is None:
                elapsed = perf_counter() - worker_start
                logger.info(
                    f"[GPTQ multi-gpu] {device} processed {processed_modules} modules "
                    f"(work={processed_work}) in {elapsed:.2f}s"
                )
                return processed_modules, processed_work, elapsed

            # Prime the pipeline with the first prefetched job.
            current_job = _schedule_payload_for_device(
                (first_payload, first_queue), device, prefetch_stream
            )

            while current_job is not None:
                # Double-buffering prefetch only from local queue.
                # This avoids early global reservation that can starve faster workers.
                next_payload, next_queue = _pop_next_payload(device, allow_steal=False)
                next_job = None
                if next_payload is not None:
                    next_job = _schedule_payload_for_device(
                        (next_payload, next_queue), device, prefetch_stream
                    )

                try:
                    module = current_job["module"]
                    hessian = current_job["hessian"]
                    num_samples = current_job["num_samples"]
                    module_work = current_job["module_work"]
                    original_device = current_job["original_device"]
                    source_queue = current_job["source_queue"]
                    ready_event = current_job["ready_event"]
                    name = self._module_names[module]
                    quant_args = getattr_chain(module, "quantization_scheme.weights")

                    logger.info(
                        f"Quantizing {name} on {device} using {num_samples} samples"
                    )

                    # Wait only for this job's copy, not the whole prefetch stream.
                    if compute_stream is not None and ready_event is not None:
                        compute_stream.wait_event(ready_event)

                    # Run compute in dedicated stream to avoid contention on default stream
                    # in multi-threaded multi-GPU scenarios
                    with torch.cuda.stream(compute_stream) if compute_stream else contextlib.nullcontext():
                        with torch.no_grad():
                            if original_device != device:
                                module.to(device=device)

                            with CompressionLogger(module) as comp_logger:
                                loss, q_param_dict = quantize_weight(
                                    module=module,
                                    quant_args=quant_args,
                                    hessian=hessian / num_samples,
                                    blocksize=self.block_size,
                                    percdamp=self.dampening_frac,
                                )
                                comp_logger.set_results(name="GPTQ", loss=loss)

                            if original_device != device:
                                module.to(device=original_device)

                    for attr, val in q_param_dict.items():
                        update_offload_parameter(module, attr, val)
                    processed_modules += 1
                    processed_work += module_work
                finally:
                    # Mark completion on the queue this job originally came from.
                    source_queue = current_job["source_queue"]
                    if source_queue is not None:
                        source_queue.task_done()

                current_job = next_job

                # If local pipeline is empty, attempt to steal now.
                # Stealing at this point keeps overlap benefits without hoarding.
                if current_job is None:
                    stolen_payload, stolen_queue = _pop_next_payload(
                        device, allow_steal=True
                    )
                    if stolen_payload is not None:
                        current_job = _schedule_payload_for_device(
                            (stolen_payload, stolen_queue), device, prefetch_stream
                        )

            elapsed = perf_counter() - worker_start
            logger.info(
                f"[GPTQ multi-gpu] {device} processed {processed_modules} modules "
                f"(work={processed_work}) in {elapsed:.2f}s"
            )
            return processed_modules, processed_work, elapsed

        start_time = perf_counter()
        with ThreadPoolExecutor(max_workers=active_devices) as executor:
            futures = [executor.submit(_compress_device_modules, device) for device in devices]
            worker_stats = [future.result() for future in as_completed(futures)]

        # Summarize per-worker stats into wall-time and aggregate work numbers.
        total_elapsed = perf_counter() - start_time
        total_modules = sum(stat[0] for stat in worker_stats)
        total_work = sum(stat[1] for stat in worker_stats)
        sum_worker_time = sum(stat[2] for stat in worker_stats)

        logger.info(
            "[GPTQ multi-gpu] finished "
            f"modules={total_modules}, work={total_work}, wall={total_elapsed:.2f}s, "
            f"accumulated_worker_time={sum_worker_time:.2f}s"
        )

    def _warmup_cholesky_on_devices(self, devices: List[torch.device]) -> None:
        # Warm up CUDA linalg kernels per device before worker threads start.
        # This avoids concurrent first-use lazy initialization in thread workers.
        for device in devices:
            if device.type != "cuda":
                continue

            try:
                with torch.cuda.device(device), torch.no_grad():
                    warmup_hessian = torch.eye(2, dtype=torch.float32, device=device)
                    torch.linalg.cholesky(warmup_hessian)
                torch.cuda.synchronize(device)
            except RuntimeError as exc:
                logger.warning(
                    "[GPTQ multi-gpu] failed to warm up cholesky on "
                    f"{device}: {exc}. Continuing without explicit warmup."
                )

    def compress_module_list(self, module_list):
        for module in module_list:
            name = self._module_names[module]
            num_samples = self._num_samples[module]
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            logger.info(f"Quantizing {name} using {int(num_samples)} samples")
            with (
                torch.no_grad(),
                align_module_device(module),
                self._maybe_onload_hessian(module),
                CompressionLogger(module) as comp_logger,
            ):
                loss, q_param_dict = quantize_weight(
                    module=module,
                    quant_args=quant_args,
                    hessian=self._hessians.pop(module) / self._num_samples.pop(module),
                    blocksize=self.block_size,
                    percdamp=self.dampening_frac,
                )
                comp_logger.set_results(name="GPTQ", loss=loss)

            for attr, val in q_param_dict.items():
                update_offload_parameter(module, attr, val)

    def _reduce_hessian_to_target_rank(self, module_list, module_to_rank):
        rank = dist.get_rank()
        pending_comms = []
        for module in module_list:
            target_rank = module_to_rank[module]
            with self._maybe_onload_hessian(module):
                pending_comms.append(
                    dist.reduce(
                        self._hessians[module],
                        op=dist.ReduceOp.SUM,
                        dst=target_rank,
                        async_op=True,
                    )
                )
                pending_comms.append(
                    dist.reduce(
                        self._num_samples[module],
                        op=dist.ReduceOp.SUM,
                        dst=target_rank,
                        async_op=True,
                    )
                )
                if rank != target_rank:
                    self._hessians.pop(module, None)
                    self._num_samples.pop(module, None)
        wait_for_comms(pending_comms)

    def _broadcast_quantized_params(self, module_list, module_to_rank):
        pending_comms = []
        for module in module_list:
            src_rank = module_to_rank[module]

            # Get parameters from module
            for attr in _GPTQ_Q_PARAMS:
                if getattr(module, attr, None) is not None:
                    pending_comms.append(
                        dist.broadcast(
                            as_broadcastable(getattr(module, attr)),
                            src=src_rank,
                            async_op=True,
                        )
                    )
        wait_for_comms(pending_comms)

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        self.ended_ = True
        QuantizationMixin.end_calibration(self, state.model)
        self.remove_hooks()  # remove gptq hooks

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: session state storing input model and calibration data
        """
        if not self.ended_:
            self.on_end(state, None)

        if len(self._num_samples) > 0:
            raise ValueError(f"Failed to compress {len(self._num_samples)} modules")

        self._hessians = dict()
        self._num_samples = dict()

        return True

    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module: torch.nn.Module):
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)

        yield

        if self.offload_hessians:
            if module in self._hessians:  # may have been deleted in context
                self._hessians[module] = self._hessians[module].to(device="cpu")
