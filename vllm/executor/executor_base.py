# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
import os
from abc import ABC, abstractmethod
from typing import (Any, Awaitable, Callable, Dict, Hashable, List, Optional,
                    Set, Tuple, Union)

import torch.nn as nn
import threading
from typing_extensions import TypeVar

import vllm.envs as envs
import vllm.platforms
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest, PoolerOutput, SequenceStage
from vllm.utils import make_async
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
        self.seq_id_state_machines: Dict[int, Dict[int, int]] = {}
        self.lock_update_req: Dict[int, threading.Lock] = {}
        self.prefill_steps_remaining = {}
        self.extended_critical: Dict[int, bool] = {}
        self._prefill_progress = {}
        # Loop index tracking per virtual engine (only while in extended critical).
        self._current_loop_idx: Dict[int, int] = {}
        # Decode gating: loop idx allowed to run decode per virtual engine.
        self._decode_valid_loop_idx: Dict[int, Dict[int, List[int]]] = {}
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
    
    async def _wait_for_sg_locks(self, virtual_engine: int):
        logger.info(f"DistributedExecutorBase._wait_for_sg_locks start for VE{virtual_engine}")
        wait_ms = 10
        log_spins = 200
        if virtual_engine not in self.seq_id_state_machines:
            logger.info(f"DistributedExecutorBase._wait_for_sg_locks has no VE{virtual_engine} in seq_id_state_machines. Creating")
            self.seq_id_state_machines[virtual_engine] = {}
        spins = 0
        start_t = time.perf_counter()
        # Wait while any seq locked OR extended critical section active.
        while any(state == 1 for state in self.seq_id_state_machines[virtual_engine].values()) \
              or self.extended_critical.get(virtual_engine, False):
            spins += 1
            if spins % log_spins == 0:
                elapsed = (time.perf_counter() - start_t) * 1000.0
                logger.info("VE%s still locked after %d spins (%.1f ms)", virtual_engine, spins, elapsed)
            await asyncio.sleep(wait_ms / 1000.0)
        elapsed = (time.perf_counter() - start_t) * 1000.0
        logger.info("VE%s free after %d spins (%.1f ms)", virtual_engine, spins, elapsed)
        if virtual_engine not in self.extended_critical:
            logger.info(f"DistributedExecutorBase._wait_for_sg_locks has no VE{virtual_engine} in extended_critical. Creating")
        self.extended_critical[virtual_engine] = True

    def _set_current_loop_idx(self, virtual_engine: int, loop_idx: int) -> None:
        assert self.extended_critical[virtual_engine], f"VE{virtual_engine} must be in extended critical to set loop idx."
        self._current_loop_idx[virtual_engine] = loop_idx

    def _unset_current_loop_idx(self, virtual_engine: int, loop_idx: int) -> None:
        if self.extended_critical[virtual_engine]:
            if self._current_loop_idx[virtual_engine] == loop_idx:
                del self._current_loop_idx[virtual_engine]
                self.extended_critical[virtual_engine] = False

    def _get_valid_decode_id(self, virtual_engine: int) -> Optional[int]:
        assert self.extended_critical[virtual_engine], f"VE{virtual_engine} must be in extended critical to get valid decode loop idx."
        if virtual_engine not in self._decode_valid_loop_idx:
            logger.info(f"DistributedExecutorBase._get_valid_decode_id has no VE{virtual_engine} in _decode_valid_loop_idx. Creating")
            self._decode_valid_loop_idx[virtual_engine] = {}
        if self._current_loop_idx[virtual_engine] not in self._decode_valid_loop_idx[virtual_engine]:
            logger.info(f"DistributedExecutorBase._get_valid_decode_id has no VE{virtual_engine}.loop_idx{self._current_loop_idx[virtual_engine]} in _decode_valid_loop_idx. Creating")
            self._decode_valid_loop_idx[virtual_engine][self._current_loop_idx[virtual_engine]] = []
        return self._decode_valid_loop_idx[virtual_engine][self._current_loop_idx[virtual_engine]]
    
    def compute_remaining_prefill_steps(self, execute_model_req):
        r = {}
        for sg in execute_model_req.seq_group_metadata_list:
            c = sg.token_chunk_size
            for seq_id, seq in sg.seq_data.items():
                if not sg.is_prompt:
                    r[seq_id] = 0
                    continue
                total = len(seq.prompt_token_ids)
                comp = seq.get_num_computed_tokens()
                rem = total - comp
                if rem <= 0 or c <= 0:
                    r[seq_id] = 0
                else:
                    steps = rem // c
                    if rem % c != 0:
                        steps += 1
                    if steps < 0:
                        steps = 0
                    r[seq_id] = steps
        return r

    def update_prefill_steps(self, execute_model_req):
        ve = execute_model_req.virtual_engine
        if ve in self.seq_id_state_machines:
            for seq_id, state in self.seq_id_state_machines[ve].items():
                assert state != 1, f"Prefill update reached with locked seq_id {seq_id} on VE{ve}"
        remaining = self.compute_remaining_prefill_steps(execute_model_req)
        for sg in execute_model_req.seq_group_metadata_list:
            for seq_id, seq in sg.seq_data.items():
                if seq_id not in self.prefill_steps_remaining:
                    self.prefill_steps_remaining[seq_id] = remaining[seq_id] - 1
                    logger.info(f"DistributedExecutorBase.update_prefill_steps has no SID{seq_id} in prefill_steps_remaining. Creating")
                else:
                    self.prefill_steps_remaining[seq_id] -= 1
                if self.prefill_steps_remaining[seq_id] == 0:
                    loop_idx = self._current_loop_idx[ve]
                    # Gate decode on this loop idx from now on.
                    self._decode_valid_loop_idx[ve][loop_idx].append(seq_id)
                    logger.info(f"Set decode valid for seq_id={seq_id} to loop idx {loop_idx} of VE{ve}")
        return self.prefill_steps_remaining
    
    def lock_seq_id_state_machines(self, execute_model_req):
        ve = execute_model_req.virtual_engine
        if ve not in self.seq_id_state_machines:
            self.seq_id_state_machines[ve] = {}
            logger.info(f"DistributedExecutorBase.lock_seq_id_state_machines has no VE{ve} in seq_id_state_machines. Creating")
        ve_map = self.seq_id_state_machines[ve]
        for sg in execute_model_req.seq_group_metadata_list:
            for seq_id in sg.seq_data.keys():
                if seq_id not in ve_map:
                    logger.info(f"DistributedExecutorBase.lock_seq_id_state_machines has no VE{ve}.SID{seq_id} in seq_id_state_machines. Creating")
                ve_map[seq_id] = 1

    def _fixup_execute_model_req_prefill(self, execute_model_req: Any) -> None:
        if execute_model_req is None:
            return
        cfg = getattr(self, "scheduler_config", None)
        max_num_batched_tokens = getattr(cfg, "max_num_batched_tokens", 0)
        orig_chunks = {}
        orig_computed = {}
        new_chunks = {}
        new_computed = {}
        for sg in execute_model_req.seq_group_metadata_list:
            if not sg.is_prompt:
                continue
            is_first_seq = True
            orig_token_chunk_size = sg.token_chunk_size
            for seq_id, seq_data in sg.seq_data.items():
                full_prompt = len(getattr(seq_data, "prompt_token_ids", ()))
                # Initialize tracking if missing
                if seq_id not in self._prefill_progress:
                    self._prefill_progress[seq_id] = (0, 0)
                    logger.info(f"DistributedExecutorBase._fixup_execute_model_req_prefill has no VE{execute_model_req.virtual_engine} in _prefill_progress. Creating")
                _, progress = self._prefill_progress[seq_id]
                # Determine base chunk size (first seen)
                chunk_size = max(1, min(max_num_batched_tokens, full_prompt - progress))
                orig_chunks[seq_id] = orig_token_chunk_size
                orig_computed[seq_id] = seq_data._num_computed_tokens
                if is_first_seq:
                    sg.token_chunk_size = chunk_size
                else:
                    sg.token_chunk_size = min(chunk_size, sg.token_chunk_size)
                # Reflect progress into sequence data object
                seq_data._num_computed_tokens = progress
                new_chunks[seq_id] = sg.token_chunk_size
                new_computed[seq_id] = seq_data._num_computed_tokens
                self._prefill_progress[seq_id] = (sg.token_chunk_size, seq_data._num_computed_tokens)
            for seq_id, seq_data in sg.seq_data.items():
                if seq_data._num_computed_tokens + sg.token_chunk_size >= len(seq_data.prompt_token_ids):
                    sg.do_sample = True
        logger.info(f"FixupExecuteModelReqPrefill: VE{execute_model_req.virtual_engine} orig_chunks={orig_chunks}, new_chunks={new_chunks}, orig_computed={orig_computed}, new_computed={new_computed}")

    def _advance_prefill_progress(self, execute_model_req: Any) -> None:
        """
        Update stored progress after a prefill chunk execution.
        Must be called AFTER the chunk has run.
        """
        for sg in execute_model_req.seq_group_metadata_list:
            if not sg.is_prompt:
                continue
            for seq_id, seq_data in sg.seq_data.items():
                full_prompt = len(getattr(seq_data, "prompt_token_ids", ()))
                chunk_size, prev = self._prefill_progress.get(seq_id, (0, 0))
                new_progress = min(full_prompt, prev + chunk_size)
                self._prefill_progress[seq_id] = (0, new_progress)

                # If full prompt consumed and prefill tracker says no steps remain, switch to decode
                remaining_steps = self.prefill_steps_remaining.get(seq_id)
                if remaining_steps == 0:
                    assert new_progress == full_prompt, f"Inconsistent prefill state for seq_id {seq_id}: remaining_steps=0 but new_progress={new_progress} != full_prompt={full_prompt}"
                    seq_data._num_computed_tokens = full_prompt
                    seq_data._stage = SequenceStage.DECODE
                    #sg.is_prompt = False
                    #sg.do_sample = True
                    #sg.token_chunk_size = 1
        # TODO: Cleanup prefill_steps_remaining entries now in decode

    async def prepare_execute_model_req_patch(
        self,
        execute_model_req: Optional[Any],
        execution_counter: Optional[int] = None,
    ) -> Tuple[Any, Dict[Hashable, Dict[str, Any]], bool]:
        loop_idx = None
        if execute_model_req is not None:
            virtual_engine = execute_model_req.virtual_engine
            if self.scheduler_config.chunked_prefill_enabled:
                if virtual_engine not in self.lock_update_req:
                    self.lock_update_req[virtual_engine] = threading.Lock()
                logger.info(f"DistributedExecutorBase.prepare_execute_model_req_patch has no VE{virtual_engine} in lock_update_req. Creating")
                with self.lock_update_req[virtual_engine]:
                    loop_idx = self._current_loop_idx[virtual_engine]
                    self._fixup_execute_model_req_prefill(execute_model_req)
                    self.update_prefill_steps(execute_model_req)
                    self.lock_seq_id_state_machines(execute_model_req)
            logger.info(f"Preparing execute_model_req patch for VE{virtual_engine} with remaining prefills {self.prefill_steps_remaining} and states {self.seq_id_state_machines[virtual_engine]}")

        return execute_model_req, loop_idx
