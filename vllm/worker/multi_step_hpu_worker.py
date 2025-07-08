# SPDX-License-Identifier: Apache-2.0
import dataclasses
from typing import Dict, Optional, Tuple

import torch

from vllm.distributed import broadcast_tensor_dict
from vllm.sequence import ExecuteModelRequest
from vllm.worker.hpu_model_runner import ModelInputForHPU
from vllm.worker.hpu_worker import HPUWorker
from vllm.worker.worker_base import WorkerInput

from vllm.distributed.parallel_state import (get_dp_group, get_tp_group,
                                             get_pp_group, get_world_group)
from vllm.logger import init_logger
logger = init_logger(__name__)
def logfn(input):
    logger.info(f"[TP{get_tp_group().rank_in_group}][PP{get_pp_group().rank_in_group}][DP{get_dp_group().rank_in_group}][WORLD{get_world_group().rank_in_group}] {input}")


class MultiStepHPUWorker(HPUWorker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_id_cached_model_input: Dict[int, ModelInputForHPU] = {}
        self.cached_model_input: Optional[ModelInputForHPU] = None
        self.seq_id_cached_num_steps: Dict[int, int] = {}
        self.cached_num_steps: Optional[int] = None

    def _get_driver_input_and_broadcast(
        self, execute_model_req: ExecuteModelRequest
    ) -> Tuple[ModelInputForHPU, WorkerInput, Dict[str, torch.Tensor]]:
        """
        Get the driver input and broadcast it to other workers.
        """
        assert self.is_driver_worker, f"This method should only be called on the driver worker"

        is_first_multi_step = execute_model_req.is_first_multi_step
        is_last_step = execute_model_req.is_last_step

        if is_first_multi_step:
            # on first step we prepare the worker input and model input normally
            worker_input: WorkerInput = self.prepare_worker_input(
                execute_model_req=execute_model_req)
            worker_input = dataclasses.replace(
                worker_input,
                num_steps=execute_model_req.num_lookahead_slots + 1)
            model_input: ModelInputForHPU = (
                self.model_runner.prepare_model_input(
                    execute_model_req.seq_group_metadata_list,
                    execute_model_req.virtual_engine,
                    execute_model_req.finished_requests_ids))

            if execute_model_req.async_callback:
                model_input = dataclasses.replace(
                    model_input,
                    async_callback=execute_model_req.async_callback)
        else:
            # on subsequent steps we reuse the worker input and model input
            seq_id = list(execute_model_req.seq_group_metadata_list[0].seq_data.keys())[0]
            assert seq_id in self.seq_id_cached_model_input, f"seq_id {seq_id} not found in seq_id_cached_model_input"
            model_input = self.seq_id_cached_model_input[seq_id]
            worker_input = WorkerInput()
            #worker_input = dataclasses.replace(
            #    worker_input,
            #    num_steps=self.seq_id_cached_num_steps[seq_id])

        model_input = dataclasses.replace(
            model_input,
            is_first_multi_step=is_first_multi_step,
            is_last_step=is_last_step)

        if self.do_metadata_broadcast:
            if is_first_multi_step:
                broadcast_data = worker_input.as_broadcastable_tensor_dict()
                broadcast_data.update(
                    model_input.as_broadcastable_tensor_dict())
                #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) pre_broadcast_1")
                broadcast_tensor_dict(broadcast_data, src=0)
                #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) post_broadcast_1")
                #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) val_broadcast_1: broadcast_data={broadcast_data}")
            else:
                broadcast_data = {
                    "is_first_multi_step": is_first_multi_step,
                    "is_last_step": is_last_step,
                    "virtual_engine": execute_model_req.virtual_engine,
                }
                #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) pre_broadcast_2")
                broadcast_tensor_dict(broadcast_data, src=0)
                #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) post_broadcast_2")
                #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) val_broadcast_2: broadcast_data={broadcast_data}")

        # Returning empty dict here to keep this compatible with
        # `LocalOrDistributedWorkerBase._get_driver_input_and_broadcast`
        return model_input, worker_input, {}

    def prepare_input(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[Tuple[ModelInputForHPU, WorkerInput, Dict[str,
                                                            torch.Tensor]]]:
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) pre_broadcast_3")
                    broadcast_tensor_dict({}, src=0)
                    #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) post_broadcast_3")
                    #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) val_broadcast_3")
                return None
            model_input, worker_input, _ = self._get_driver_input_and_broadcast(
                execute_model_req)
            if model_input.is_first_multi_step:
                for sid in model_input.sampling_metadata.seq_groups[0].seq_ids:
                    self.seq_id_cached_model_input[sid] = model_input
                    self.seq_id_cached_num_steps[sid] = worker_input.num_steps
            return model_input, worker_input, {}
        else:
            #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) pre_broadcast_4")
            broadcast_data = broadcast_tensor_dict(src=0)
            #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) post_broadcast_4")
            #logfn(f"MultiStepHPUWorker.prepare_input({self.execution_counter}) val_broadcast_4: broadcast_data={broadcast_data}")
            if not broadcast_data:
                return None

            if len(broadcast_data) == 3:                    
                assert self.cached_model_input is not None
                self.cached_model_input = dataclasses.replace(
                    self.cached_model_input,
                    is_first_multi_step=broadcast_data["is_first_multi_step"],
                    is_last_step=broadcast_data["is_last_step"])
                empty_worker_input = WorkerInput()
                #empty_worker_input = dataclasses.replace(
                #    empty_worker_input,
                #    num_steps=self.cached_num_steps)
                return self.cached_model_input, empty_worker_input, {}
            worker_input = WorkerInput.from_broadcasted_tensor_dict(
                broadcast_data)
            model_input = (
                self.model_runner.
                make_model_input_from_broadcasted_tensor_dict(broadcast_data))
            self.cached_model_input = model_input
            self.cached_num_steps = worker_input.num_steps
            return model_input, worker_input, {}
