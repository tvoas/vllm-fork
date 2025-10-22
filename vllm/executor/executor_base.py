# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import array
import asyncio
import time
from abc import ABC, abstractmethod
from typing import (Any, Awaitable, Callable, Dict, Hashable, List, Optional,
                    Set, Tuple, Union)

import torch.nn as nn
import threading
from typing_extensions import TypeVar

import vllm.platforms
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import make_async, sha256
from vllm.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class ExecutorBase(ABC):
    """Base class for all executors.

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor 
    that can execute the model on multiple devices.
    """

    uses_ray: bool  # whether the executor uses Ray for orchestration.

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self._init_executor()
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict[str, Any]] = None) -> List[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.

        Returns:
            A list containing the results from each worker.
        
        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        raise NotImplementedError

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        Normally, this should simply delegate to the underlying Worker. Some
        ExecutorBase may require modification of the result, e.g. to ensure the
        selected cache sizes are compatible with all workers.

        Returns a Tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        results = self.collective_rpc("determine_num_available_blocks")
        a = min([r[0] for r in results])
        b = min([r[1] for r in results])
        return a, b

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 workers.
        logger.info("# %s blocks: %d, # CPU blocks: %d",
                    vllm.platforms.current_platform.device_name,
                    num_gpu_blocks, num_cpu_blocks)
        max_concurrency = (num_gpu_blocks * self.cache_config.block_size /
                           self.model_config.max_model_len)
        logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                    self.model_config.max_model_len, max_concurrency)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self.collective_rpc("initialize_cache",
                            args=(num_gpu_blocks, num_cpu_blocks))

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        Run a function directly on the model inside each worker,
        returning the result for each of them.
        """

        def rpc_func(worker: WorkerBase) -> _R:
            return func(worker.get_model())

        return self.collective_rpc(rpc_func)

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[Union[SamplerOutput, PoolerOutput]]]:
        output = self.collective_rpc("execute_model",
                                     args=(execute_model_req, ))
        return output[0]

    def stop_remote_worker_execution_loop(self) -> None:
        """Releases parallel workers from model loop."""
        return

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("add_lora", args=(lora_request, )))

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("remove_lora", args=(lora_id, )))

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("pin_lora", args=(lora_id, )))

    def list_loras(self) -> Set[int]:
        sets = self.collective_rpc("list_loras")
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
        return sets[0]

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        assert prompt_adapter_request.prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return all(
            self.collective_rpc("add_prompt_adapter",
                                args=(prompt_adapter_request, )))

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return all(
            self.collective_rpc("remove_prompt_adapter",
                                args=(prompt_adapter_id, )))

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return all(
            self.collective_rpc("pin_prompt_adapter",
                                args=(prompt_adapter_id, )))

    def list_prompt_adapters(self) -> Set[int]:
        sets = self.collective_rpc("list_prompt_adapters")
        for s in sets:
            assert (s == sets[0]
                    ), "All workers should have the same prompt adapters."
        return sets[0]

    def start_profile(self) -> None:
        self.collective_rpc("start_profile")

    def stop_profile(self) -> None:
        self.collective_rpc("stop_profile")

    def sleep(self, level: int = 1):
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        logger.info("It took %.6f seconds to fall asleep.",
                    time_after_sleep - time_before_sleep)

    def wake_up(self, tags: Optional[list[str]] = None):
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning("Tag %s is not in sleeping tags %s", tag,
                                   self.sleeping_tags)
                    return
        time_before_wakeup = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        time_after_wakeup = time.perf_counter()
        logger.info("It took %.6f seconds to wake up tags %s.",
                    time_after_wakeup - time_before_wakeup,
                    tags if tags is not None else self.sleeping_tags)
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.collective_rpc("save_sharded_state",
                            kwargs=dict(path=path,
                                        pattern=pattern,
                                        max_size=max_size))

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        return

    def __del__(self):
        self.shutdown()

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        """Executes one model step on the given sequences."""
        output = await make_async(self.execute_model)(execute_model_req)
        return output

    async def stop_remote_worker_execution_loop_async(self) -> None:
        """Releases parallel workers from model loop."""
        return

    async def check_health_async(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        self.check_health()


class DistributedExecutorBase(ExecutorBase):
    """Abstract superclass of distributed executor implementations."""

    def __init__(self, *args, **kwargs):
        # This is non-None when the execute model loop is running
        # in the parallel workers. It's a coroutine in the AsyncLLMEngine case.
        self.parallel_worker_tasks: Optional[Union[Any, Awaitable[Any]]] = None
        self.cached_execute_model_reqs: Dict[int, ExecuteModelRequest] = {}
        self.execute_step_count = 0
        import threading
        self.step_count_lock = threading.Lock()
        # Tracks per-seq prefill chunk step (1-based).
        self._prefill_chunk_steps: Dict[Hashable, int] = {}
        # Remainders beyond current transmitted truncation:
        # {virtual_engine: {seq_key: {attr: list/array/tuple remainder}}}
        self._chunk_remainders: Dict[int, Dict[Hashable, Dict[str, Any]]] = {}
        # Background streaming state
        self._streaming_threads: Dict[int, threading.Thread] = {}
        self._streaming_active: Dict[int, bool] = {}
        self._master_cache_lock = threading.Lock()
        self._cache_lock: Dict[int, threading.Lock] = {}
        super().__init__(*args, **kwargs)

    def execute_model(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> Optional[List[SamplerOutput]]:
        # TODO: unify into collective_rpc
        if self.parallel_worker_tasks is None:
            self.parallel_worker_tasks = self._run_workers(
                "start_worker_execution_loop",
                async_run_tensor_parallel_workers_only=True)

        # Only the driver worker returns the sampling results.
        driver_outputs = self._driver_execute_model(execute_model_req)
        return driver_outputs

    def stop_remote_worker_execution_loop(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        self._driver_execute_model(execute_model_req=None)
        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        self._wait_for_tasks_completion(parallel_worker_tasks)

    @abstractmethod
    def _driver_execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest]
    ) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution loop
        running in each of the remote workers. In this case, this method
        returns None. Otherwise, this method returns the model output.
        """
        raise NotImplementedError

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        return self._run_workers(method, *args, **(kwargs or {}))

    @abstractmethod
    def _run_workers(
        self,
        method: Union[str, Callable],
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers.

        Args:
            async_run_tensor_parallel_workers_only: If True the method will be
                run only in the remote TP workers, not the driver worker.
                It will also be run asynchronously and return a list of futures
                rather than blocking on the results.
        
        # TODO: simplify and merge with collective_rpc
        """
        raise NotImplementedError

    @abstractmethod
    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        raise NotImplementedError

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if self.parallel_worker_tasks is None:
            # Start model execution loop running in the parallel workers
            self.parallel_worker_tasks = asyncio.create_task(
                self._start_worker_execution_loop())

        # Only the driver worker returns the sampling results.
        return await self._driver_execute_model_async(execute_model_req)

    async def stop_remote_worker_execution_loop_async(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        await self._driver_execute_model_async()
        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        await parallel_worker_tasks

    @abstractmethod
    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> List[SamplerOutput]:
        """Execute the model asynchronously in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        raise NotImplementedError

    @abstractmethod
    async def _start_worker_execution_loop(self):
        """Run execution loop on all workers. It guarantees all workers run
        the loop or None of them is running the loop. Loop can be stopped by
        `stop_remote_worker_execution_loop`.
        The API is idempotent (guarantee only 1 loop run at any moment)."""
        raise NotImplementedError
    
    def _start_background_streaming(self, ve: int, original_prompt_sizes: Dict[int, Tuple[int,int,int]]):
        # Spawn only once per VE.
        if self._streaming_active.get(ve, False):
            return
        self._streaming_active[ve] = True
        def _worker():
            all_remainders = {}
            for seq_key in original_prompt_sizes:
                remainders = self._chunk_remainders.get(seq_key, {})
                all_remainders[seq_key] = remainders
            with self._master_cache_lock:
                if ve not in self._cache_lock:
                    self._cache_lock[ve] = threading.Lock()
            logger.info(f"[EXEC] attempt-lock _cache_lock ve={ve} func=_start_background_streaming seq_keys={list(all_remainders.keys())}")
            with self._cache_lock[ve]:
                logger.info(f"[EXEC] acquired-lock _cache_lock ve={ve} func=_start_background_streaming seq_keys={list(all_remainders.keys())}")
                entry = self.cached_execute_model_reqs.get(ve)
                base_hash = ""
                cache = {}
                if entry:
                    base_hash, cache = entry
                    for seq_key, attr_map in all_remainders.items():
                        if seq_key not in cache:
                            continue
                        for attr, remainder in attr_map.items():
                            if attr not in cache[seq_key]:
                                continue
                            target = cache[seq_key][attr]
                            if isinstance(target, (list, array.array)):
                                target.extend(remainder)
                            elif isinstance(target, tuple):
                                cache[seq_key][attr] = target + tuple(remainder)
                # Send to workers (they will ignore sequences not yet cached)
                try:
                    self.collective_rpc("stream_prefill_chunk", args=(ve, all_remainders))
                except Exception as e:
                    logger.warning(f"[EXEC] stream_prefill_chunk rpc failed ve={ve}: {e}")
                if entry:
                    self.cached_execute_model_reqs[ve] = (base_hash, cache)
                logger.info(f"[EXEC] release-lock _cache_lock ve={ve} func=_start_background_streaming seq_keys={list(all_remainders.keys())}")
            self._streaming_active[ve] = False
        t = threading.Thread(target=_worker, name=f"prefill-stream-ve{ve}", daemon=True)
        self._streaming_threads[ve] = t
        t.start()

    
    def _chunk_execute_model_req(
        self,
        execute_model_req: Any,
        original_prompt_sizes: Dict[int, Tuple[int, int, int]],
        chunkable_attrs: List[str],
    ):
        for seq_group in execute_model_req.seq_group_metadata_list:
            for seq_key, seq_data in seq_group.seq_data.items():
                self._chunk_remainders[seq_key] = {}
                chunk_info = original_prompt_sizes[seq_key]
                cutoff_len = chunk_info[2]
                if cutoff_len == 0:
                    continue
                for attr in chunkable_attrs:
                    val = getattr(seq_data, attr)
                    total_len = len(val)
                    if cutoff_len >= total_len:
                        continue
                    remainder = val[cutoff_len:total_len]
                    truncated = val[:cutoff_len]
                    self._chunk_remainders[seq_key][attr] = remainder
                    setattr(seq_data, attr, truncated)


    def restore_chunked_execute_model_req(
        self,
        execute_model_req: Any
    ):
        if execute_model_req is None:
            return
        def _as_array_l(val):
            if isinstance(val, array.array):
                return array.array(val.typecode, val)
            return array.array("l", val)
        chunkable_attrs = [
            "_cached_all_token_ids",
            "_prompt_token_ids",
            "_prompt_token_ids_tuple",
        ]
        with self._master_cache_lock:
            if execute_model_req.virtual_engine not in self._cache_lock:
                self._cache_lock[execute_model_req.virtual_engine] = threading.Lock()
        logger.info(f"[EXEC] attempt-lock _cache_lock ve={execute_model_req.virtual_engine} func=restore_chunked_execute_model_req")
        with self._cache_lock[execute_model_req.virtual_engine]:
            logger.info(f"[EXEC] acquire-lock _cache_lock ve={execute_model_req.virtual_engine} func=restore_chunked_execute_model_req")
            for seq_group in execute_model_req.seq_group_metadata_list:
                for seq_key, seq_data in seq_group.seq_data.items():
                    if seq_key not in self._chunk_remainders:
                        continue
                    remainder_map = self._chunk_remainders[seq_key]
                    for attr in chunkable_attrs:
                        if attr not in remainder_map:
                            continue
                        patch_val = remainder_map[attr]
                        cur = getattr(seq_data, attr)
                        if isinstance(cur, array.array):
                            if patch_val is not None:
                                if isinstance(patch_val, array.array):
                                    cur.extend(patch_val)
                                else:
                                    cur.extend(_as_array_l(patch_val))
                            setattr(seq_data, attr,
                                array.array("l", cur))  # avoid aliasing
                        elif isinstance(cur, list):
                            if patch_val:
                                cur.extend(patch_val)
                            setattr(seq_data, attr,
                                list(cur))  # avoid aliasing
                        elif isinstance(cur, tuple):
                            if patch_val:
                                cur = cur + (tuple(patch_val) if not isinstance(patch_val, tuple) else patch_val)
                            setattr(seq_data, attr, cur)
                    del self._chunk_remainders[seq_key]
            logger.info(f"[EXEC] release-lock _cache_lock ve={execute_model_req.virtual_engine} func=restore_chunked_execute_model_req")


    def _compute_execute_model_req_patch(
        self,
        prev_cache: Dict[Hashable, Dict[str, Any]],
        execute_model_req: Any,
        tracked_attrs: List[str],
    ) -> Dict[Hashable, Dict[str, Any]]:
        """Build a minimal patch describing changes since the previous cache.

        Compares the current execute_model_req data with the prev_cache and
        returns only the differences per sequence key:
        - For list/array/tuple attrs: returns the newly appended slice.
        - For other attrs: returns the new value if it changed.
        If a sequence key is new, returns the full set of tracked attributes and
        attaches the group's sampling_params.

        Args:
            prev_cache: Per-sequence cached values from the previous request.
            execute_model_req: Current execute-model request carrying
                               sequence data.
            tracked_attrs: Attribute names to compare/track incrementally.

        Returns:
            A dict keyed by sequence key with only the incremental changes.
        """
        chunkable_attrs = [
            "_cached_all_token_ids",
            "_prompt_token_ids",
            "_prompt_token_ids_tuple",
        ]
        patch_by_key: Dict[Hashable, Dict[str, Any]] = {}
        for seq_group in execute_model_req.seq_group_metadata_list:
            sampling_params = seq_group.sampling_params
            for seq_key, seq_data in seq_group.seq_data.items():
                if seq_key in prev_cache:
                    prev_entry = prev_cache[seq_key]
                    patch: Dict[str, Any] = {}
                    for attr in tracked_attrs:
                        curr_val = getattr(seq_data, attr)
                        prev_val = prev_entry[attr]
                        if isinstance(curr_val, (list, array.array, tuple)):
                            if len(curr_val) > len(prev_val):
                                patch[attr] = curr_val[len(prev_val):]
                        else:
                            if curr_val != prev_val:
                                patch[attr] = curr_val
                    patch_by_key[seq_key] = patch
                else:
                    copied = {}
                    for attr in tracked_attrs:
                        v = getattr(seq_data, attr)
                        if isinstance(v, array.array):
                            copied[attr] = array.array(v.typecode, v)
                        elif isinstance(v, list):
                            copied[attr] = list(v)
                        elif isinstance(v, tuple):
                            copied[attr] = tuple(v)
                        elif isinstance(v, dict):
                            copied[attr] = dict(v)
                        elif isinstance(v, set):
                            copied[attr] = set(v)
                        else:
                            copied[attr] = v  # or try v.copy() if available
                    patch_by_key[seq_key] = copied
                    patch_by_key[seq_key]["sampling_params"] = sampling_params

        return patch_by_key

    def _update_cache_with_new_tokens(
        self,
        cache: Optional[Dict[Hashable, Dict[str, Any]]],
        execute_model_req: Any,
        tracked_attrs: List[str],
    ) -> Dict[Hashable, Dict[str, Any]]:
        """Merge current per-sequence data into the cache in-place.

        For each tracked attribute:
        - list/array: append only the new tail
        - tuple: append the new tail as a tuple
        - other: overwrite with the current value

        Args:
            cache: Mutable cache from previous calls; created if None.
            execute_model_req: Current execute-model request carrying
                               sequence data.
            tracked_attrs: Attribute names to append/overwrite.

        Returns:
            The updated cache mapping by sequence key.
        """
        if cache is None:
            cache = {}

        for seq_group in execute_model_req.seq_group_metadata_list:
            for seq_key, seq_data in seq_group.seq_data.items():
                if seq_key not in cache:
                    cache[seq_key] = {
                        attr: (list(getattr(seq_data, attr)) if isinstance(
                            getattr(seq_data, attr),
                            (list, array.array)) else getattr(seq_data, attr))
                        for attr in tracked_attrs
                    }
                else:
                    cached_entry = cache[seq_key]
                    for attr in tracked_attrs:
                        curr_val = getattr(seq_data, attr)
                        if isinstance(curr_val, (list, array.array)):
                            cached_entry[attr].extend(
                                curr_val[len(cached_entry[attr]):])
                        elif isinstance(curr_val, tuple):
                            cached_entry[attr] += tuple(
                                curr_val[len(cached_entry[attr]):])
                        else:
                            cached_entry[attr] = curr_val

        return cache

    def _strip_patch_data_from_execute_model_req(
            self, execute_model_req: Any) -> None:
        """Clear per-request mutable fields after patch extraction.

        Resets sequence- and group-level fields so the base request is lean
        and safe to cache/reuse without duplicating incremental data.
        """
        for seq_group in execute_model_req.seq_group_metadata_list:
            seq_group.sampling_params = None
            for seq_data in seq_group.seq_data.values():
                seq_data._cached_all_token_ids = []
                seq_data._new_appended_tokens = []
                seq_data._output_token_ids = array.array("l", [])
                seq_data._prompt_token_ids = array.array("l", [])
                seq_data._prompt_token_ids_tuple = ()
                seq_data._num_computed_tokens = 0
                seq_data._stage = seq_data._stage.value

    def _get_chunked_prefill_limits(
        self, execute_model_req: Any
    ) -> Dict[Hashable, int]:
        """
        Return per-sequence allowed prompt length window for chunked prefill.

        For each sequence:
        - If global chunked prefill disabled: {} (callers treat as no limits).
        - If sequence is in decode phase (_num_computed_tokens > 0): 0.
        - If in prompt/prefill phase: step * token_chunk_size, where step is
          read from self._prefill_chunk_steps (default 1). This function DOES
          NOT increment the step counter; callers handle advancement.
        """
        if execute_model_req is None:
            return {}
        cfg = getattr(self, "scheduler_config", None)
        if cfg is None or not getattr(cfg, "chunked_prefill_enabled", False):
            return {}

        limits: Dict[Hashable, int] = {}
        for seq_group in execute_model_req.seq_group_metadata_list:
            is_prompt = seq_group.is_prompt
            chunk_size = getattr(seq_group, "token_chunk_size", 0) or 0
            for seq_key, seq_data in seq_group.seq_data.items():
                computed = getattr(seq_data, "get_num_computed_tokens", None)
                if callable(computed):
                    try:
                        computed = computed()
                    except Exception:
                        pass
                if is_prompt and chunk_size > 0:
                    step = self._prefill_chunk_steps.get(seq_key, 0)
                    assert step > 0, "Prefill chunk step should be greater than zero."
                    next_target = computed + chunk_size + 1
                    max_target = len(seq_data.prompt_token_ids)
                    limits[seq_key] = (0 if step == 1 else computed + 1, min(next_target, max_target))
                else:
                    limits[seq_key] = (0, 0)
        return limits

    def prepare_execute_model_req_patch(
        self,
        execute_model_req: Optional[Any],
        execute_step_count: int = 0,
    ) -> Tuple[Any, Dict[Hashable, Dict[str, Any]], bool]:
        """Produce an incremental patch and optionally reuse a cached
        base request.

        Computes a patch containing only the changes since the last request,
        updates the internal cache with newly observed data, strips mutable
        fields from the current request, and decides if the previous base
        request can be reused (based on a content hash).

        Args:
            execute_model_req: Current execute-model request or None.

        Returns:
            A tuple of:
            - Either the virtual_engine (if reusing cached base) or
              the request.
            - The computed patch dict keyed by sequence key.
            - A boolean indicating whether the base request was reused.
        """
        tracked_attrs: List[str] = [
            "_cached_all_token_ids",
            "_new_appended_tokens",
            "_num_computed_tokens",
            "_output_token_ids",
            "_prompt_token_ids",
            "_prompt_token_ids_tuple",
            "_cumulative_logprob",
        ]
        chunkable_attrs: List[str] = [
            "_cached_all_token_ids",
            "_prompt_token_ids",
            "_prompt_token_ids_tuple",
        ]
        original_prompt_sizes = {}
        execute_model_req_patch: Dict[Hashable, Dict[str, Any]] = {}
        use_cached_base_req = False
        if execute_model_req is not None:
            virtual_engine = execute_model_req.virtual_engine
            if virtual_engine in self.cached_execute_model_reqs:
                prev_execute_model_req_hash, cached_execute_model_req = (
                    self.cached_execute_model_reqs[virtual_engine])
            else:
                prev_execute_model_req_hash, cached_execute_model_req = "", {}

            for seq_group in execute_model_req.seq_group_metadata_list:
                for seq_key, seq_data in seq_group.seq_data.items():
                    # If sequence already decoding, clear any stale counter.
                    if not seq_group.is_prompt:
                        self._prefill_chunk_steps.pop(seq_key, None)
                        continue
                    # Initialize step if new.
                    step = self._prefill_chunk_steps.get(seq_key, 0)
                    self._prefill_chunk_steps[seq_key] = step + 1
                    original_prompt_sizes[seq_key] = len(seq_data._prompt_token_ids)

            chunk_sizes = self._get_chunked_prefill_limits(execute_model_req)
            for seq_group in execute_model_req.seq_group_metadata_list:
                for seq_key, seq_data in seq_group.seq_data.items():
                    chunk_size = chunk_sizes.get(seq_key, (0, 0))
                    original_prompt_sizes[seq_key] = (original_prompt_sizes.get(seq_key, 0), chunk_size[0], chunk_size[1])

            self._chunk_execute_model_req(execute_model_req, original_prompt_sizes, chunkable_attrs)
            with self._master_cache_lock:
                if execute_model_req.virtual_engine not in self._cache_lock:
                    self._cache_lock[execute_model_req.virtual_engine] = threading.Lock()
            logger.info(f"[EXEC] attempt-lock _cache_lock ve={execute_model_req.virtual_engine} func=prepare_execute_model_req_patch")
            with self._cache_lock[virtual_engine]:
                logger.info(f"[EXEC] acquire-lock _cache_lock ve={execute_model_req.virtual_engine} func=prepare_execute_model_req_patch")
                execute_model_req_patch = self._compute_execute_model_req_patch(
                    cached_execute_model_req, execute_model_req, tracked_attrs)
                self.cached_execute_model_reqs[
                    virtual_engine] = self._update_cache_with_new_tokens(
                        cached_execute_model_req, execute_model_req, tracked_attrs)
                self._strip_patch_data_from_execute_model_req(execute_model_req)

                new_execute_model_req_hash = sha256(execute_model_req)
                self.cached_execute_model_reqs[virtual_engine] = (
                    new_execute_model_req_hash,
                    self.cached_execute_model_reqs[virtual_engine],
                )
                if prev_execute_model_req_hash == new_execute_model_req_hash:
                    use_cached_base_req = True
                logger.info(f"[EXEC] release-lock _cache_lock ve={execute_model_req.virtual_engine} func=prepare_execute_model_req_patch")

            self._start_background_streaming(virtual_engine, original_prompt_sizes)
        return (
            virtual_engine if use_cached_base_req else execute_model_req,
            execute_model_req_patch,
            use_cached_base_req,
            original_prompt_sizes,
            execute_step_count,
        )
