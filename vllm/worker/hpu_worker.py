# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import array
import contextlib
import gc
import gzip
import json
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import habana_frameworks.torch as htorch  # noqa:F401
import torch
import torch.distributed
from vllm_hpu_extension.profiler import HabanaMemoryProfiler, format_bytes

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized, get_pp_group,
                              get_tp_group, get_world_group, init_distributed_environment)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SequenceStage, CompletionSequenceGroupOutput)
from vllm.utils import (bind_kv_cache, hpu_backend_string, hpu_device_string,
                        is_fake_hpu)
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.hpu_enc_dec_model_runner import HPUEncoderDecoderModelRunner
from vllm.worker.hpu_model_runner import HPUModelRunner, HPUModelRunnerBase
from vllm.worker.hpu_pooling_model_runner import HPUPoolingModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)

logger = init_logger(__name__)


class HPUWorker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a HPU.

    Each worker is associated with a single HPU. The worker is responsible for
    maintaining the KV cache and executing the model on the HPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[HPUModelRunner]] = None,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.hf_config.model_type \
                == model_config.hf_config.model_type) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator", "eagle", "deepseek_mtp"]) \
                    else {"return_hidden_states": True}

        is_encoder_decoder_model = self._is_encoder_decoder_model()
        ModelRunnerClass: Type[HPUModelRunnerBase] = HPUModelRunner
        is_causal = True
        if self.model_config.runner_type == "pooling":
            ModelRunnerClass = HPUPoolingModelRunner
        elif is_encoder_decoder_model:
            ModelRunnerClass = HPUEncoderDecoderModelRunner
        self.model_runner: HPUModelRunnerBase = ModelRunnerClass(
            vllm_config=vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
            is_causal=is_causal,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[HPUCacheEngine]
        # Initialize gpu_cache as pooling models don't initialize kv_caches
        self.hpu_cache: Optional[List[List[torch.Tensor]]] = None
        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)

            if os.getenv('VLLM_PROFILER_ENABLED') == 'full':
                fn = self.full_trace_handler
                with_stack = False
            else:
                fn = torch.profiler.tensorboard_trace_handler
                with_stack = False

            prof_activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.HPU,
            ]

            self.profiler = torch.profiler.profile(
                activities=prof_activities,
                with_stack=with_stack,
                on_trace_ready=fn(torch_profiler_trace_dir, use_gzip=True))

        else:
            self.profiler = None

        self.on_demand_profiler_mode = None
        if envs.VLLM_ON_DEMAND_TORCH_PROFILER:
            self.on_demand_profiler_step_start = 0
            self.on_demand_profiler_step_stop = 1
            self.on_demand_profiler_step_counter = 0

            self.update_on_demand_profiler_cfg()

        self.all_cached_seq_data: Dict[int, dict] = {}
        self.cached_execute_model_req: Dict[int, ExecuteModelRequest] = {}
        self.lock = threading.Lock()

    def update_on_demand_profiler_cfg(self):
        assert self.on_demand_profiler_step_counter==0

        # read config from the file set with VLLM_ON_DEMAND_TORCH_PROFILER
        on_demand_profiler_cfg = None
        if os.path.isfile(envs.VLLM_ON_DEMAND_TORCH_PROFILER):
            with open(envs.VLLM_ON_DEMAND_TORCH_PROFILER, 'r') as f:
                on_demand_profiler_cfg = f.read().strip() # mode,step_start,step_stop
        else:
            logger.warning(f"VLLM_ON_DEMAND_TORCH_PROFILER file {envs.VLLM_ON_DEMAND_TORCH_PROFILER} not found, skipping")

        if on_demand_profiler_cfg is not None:
            on_demand_profiler_cfg_list = [option.strip() for option in on_demand_profiler_cfg.split(',')]
            if len(on_demand_profiler_cfg_list) > 0:
                self.on_demand_profiler_mode = on_demand_profiler_cfg_list[0]
            if len(on_demand_profiler_cfg_list) > 1:
                self.on_demand_profiler_step_start = int(on_demand_profiler_cfg_list[1])
            if len(on_demand_profiler_cfg_list) > 2:
                self.on_demand_profiler_step_stop = int(on_demand_profiler_cfg_list[2])

            self.on_demand_profiler_step_counter = 0

            logger.info(f"on_demand_profiler config: ({self.on_demand_profiler_mode}, {self.on_demand_profiler_step_start},"
                         f" {self.on_demand_profiler_step_stop}),"
                         f" {self.on_demand_profiler_step_counter}")
        else:
            logger.warning(f"Invalid on_demand_profiler config in {envs.VLLM_ON_DEMAND_TORCH_PROFILER}, skipping")

    def full_trace_handler(self, dir_name, use_gzip=False):

        def handler_fn(prof) -> None:
            if not os.path.isdir(dir_name):
                try:
                    os.makedirs(dir_name, exist_ok=True)
                except Exception as e:
                    raise RuntimeError("Can't create directory: " +
                                       dir_name) from e
            file_name = f"vllm.{time.time_ns()}.pt.trace.json"
            file_path = os.path.join(dir_name, file_name)
            prof.export_chrome_trace(file_path)
            with open(file_path) as f:
                pytorch_trace = json.load(f)
            os.remove(file_path)
            base = pytorch_trace['baseTimeNanoseconds'] / 1000
            events = self.model_runner.profiler.profiling_trace_events
            while True:
                try:
                    event_str = events.get_nowait()
                    event = json.loads(event_str[:-1])
                    event['ts'] = event['ts'] - base
                    pytorch_trace['traceEvents'].append(event)
                except queue.Empty:
                    break

            pytorch_trace['traceEvents'].append({
                "args": {
                    "name": "vLLM"
                },
                "name": "process_name",
                "ph": "M",
                "pid": 1,
                "tid": 0,
                "ts": 0.0
            })
            if use_gzip:
                file_path = file_path + ".gz"
                with gzip.open(file_path, 'wt', encoding="ascii") as zipfile:
                    json.dump(pytorch_trace, zipfile)
            else:
                with open(file_path, "w") as outfile:
                    outfile.write(json.dumps(pytorch_trace))
            logger.info("Saved full profiling to %s", file_path)

        return handler_fn

    def _is_encoder_decoder_model(self):
        return self.model_config.is_encoder_decoder

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")

        self.do_start_profile()

    def do_start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")

        high_level_profiler = self.model_runner.profiler
        with high_level_profiler.record_event('internal', 'start_profiler'):
            # Clean up the queue
            while True:
                try:
                    high_level_profiler.profiling_trace_events.get_nowait()
                except queue.Empty:
                    break
            self.profiler.start()
        logger.info(f"Started profiler")

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")

        self.do_stop_profile()

    def do_stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")

        self.profiler.stop()
        logger.info(f"Stopped profiler")

    def do_step_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")

        self.profiler.step()
        logger.info(f"Stepped profiler")

    def _set_env_vars(self):
        local_rank = self.local_rank
        if self.parallel_config.world_size == 1:
            local_rank = -1
        import os
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["ID"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(self.parallel_config.world_size)
        os.environ["RANK"] = str(self.rank)

    def init_device(self) -> None:
        if self.device_config.device.type == "hpu":
            self.device = torch.device("hpu")
            torch.hpu.set_device(self.local_rank)
        elif self.device_config.device_type == "cpu":
            self.device = torch.device("cpu")
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        if self.model_config.quantization == 'inc':
            self._set_env_vars()
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()
        if isinstance(self.model_runner, HPUPoolingModelRunner):
            # recipes we will use the extra memory for graphs/blocks
            free_hpu_memory = torch.hpu.mem_get_info()[0]
            hpu_memory_margin = free_hpu_memory * (
                1 - self.cache_config.gpu_memory_utilization)
            self.model_runner.mem_margin = hpu_memory_margin
            self._warm_up_model()

    def _apply_patch_to_execute_model_req(
        self,
        new_execute_model_req: ExecuteModelRequest,
        cached_seq_data: dict,
        execute_model_req_patch: dict,
    ) -> dict:
        """
        Merge an incremental patch into the per-VE sequence cache and reflect
        it onto the in-flight ExecuteModelRequest.

        Args:
            new_execute_model_req: Request containing seq_group_metadata_list.
            cached_seq_data: Cache mapping sequence keys to cached values.
            execute_model_req_patch: Patch mapping sequences to updated fields.

        Returns:
            The updated VE-local cache (same object as cached_seq_data).
        """
        active_keys = set()  # keys seen in this request; used for pruning
        # Prune only during decode (is_prompt=False). On some ranks is_prompt
        # can be None; treat explicit False as decode.
        should_prune = any(
            getattr(seq_group, "is_prompt", None) is False
            for seq_group in new_execute_model_req.seq_group_metadata_list)

        def _as_array_l(val):
            if isinstance(val, array.array):
                return val
            return array.array("l", val)

        for seq_idx, seq_group in enumerate(
                new_execute_model_req.seq_group_metadata_list):
            for key, seq_data in seq_group.seq_data.items():
                active_keys.add(key)

                # Get or initialize cached data
                initial_data = {
                    '_cached_all_token_ids': [],
                    '_new_appended_tokens': [],
                    '_output_token_ids': array.array("l"),
                    '_prompt_token_ids': array.array("l"),
                    '_prompt_token_ids_tuple': (),
                    '_cumulative_logprob': None,
                    '_num_computed_tokens': 0,
                }
                cached_data = cached_seq_data.setdefault(key, initial_data)

                # Get patch data
                patch_data = execute_model_req_patch.get(key, {})

                for attr_key in initial_data:
                    cur = cached_data.get(attr_key)
                    patch_val = patch_data.get(attr_key, None)

                    if isinstance(cur, array.array):
                        if patch_val is not None:
                            cur.extend(_as_array_l(patch_val))
                        cached_data[attr_key] = cur
                        setattr(seq_data, attr_key,
                                array.array("l", cur))  # avoid aliasing
                    elif isinstance(cur, list):
                        if patch_val:
                            cur.extend(patch_val)
                        cached_data[attr_key] = cur
                        setattr(seq_data, attr_key,
                                list(cur))  # avoid aliasing
                    elif isinstance(cur, tuple):
                        if patch_val:
                            cur = cur + tuple(patch_val)
                        cached_data[attr_key] = cur
                        setattr(seq_data, attr_key, cur)
                    else:
                        # Scalars
                        if attr_key in patch_data:
                            cached_data[attr_key] = patch_val
                        setattr(seq_data, attr_key, cached_data.get(attr_key))

                # sampling_params lives on seq_group; cache it for continuity
                sp = patch_data.get('sampling_params')
                if sp is not None:
                    cached_data['sampling_params'] = sp
                if (seq_group.sampling_params is None
                        and 'sampling_params' in cached_data):
                    seq_group.sampling_params = cached_data['sampling_params']

                # Normalize stage enum.
                if isinstance(seq_data._stage, int):
                    seq_data._stage = SequenceStage(seq_data._stage)

        # Prune only during decode steps to prevent unbounded growth while
        # preserving prompt-batch data across prefill micro-batches.
        if should_prune:
            for key in list(cached_seq_data.keys()):
                if key not in active_keys:
                    cached_seq_data.pop(key, None)

        return cached_seq_data
    
    def _debug_log_model_input(
        self,
        model_input,
        execute_step_count: int,
        virtual_engine: Optional[int] = None,
    ) -> None:

        # Gate logs to world rank 0 (group 0)
        should_log = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_rank = get_world_group().rank_in_group
            should_log = (world_rank == 0)
        if not should_log:
            return

        # Lazy-init trackers on self
        if not hasattr(self, "_chunked_prefill_trackers"):
            # key: seq_id -> {
            #   'steps': int, 'last_seq_len': int, 'chunks': [int],
            #   'last_min_pos': int | None, 'last_max_pos': int | None,
            #   'last_virtual_engine': int | None
            # }
            self._chunked_prefill_trackers = {}

        attn = getattr(model_input, "attn_metadata", None)
        assert attn is not None, "attn_metadata missing"
        input_tokens = model_input.input_tokens
        input_positions = model_input.input_positions
        assert input_tokens.shape == input_positions.shape, "tokens/positions shape mismatch"

        bsz = getattr(model_input, "real_batch_size", input_tokens.size(0))
        if bsz is None:
            logger.info(f"[Rank={get_world_group().rank_in_group}] real_batch_size missing: model_input = {model_input}")
            return
        bsz = int(bsz)
        padded_len = int(input_tokens.size(1))
        is_prompt = bool(getattr(model_input, "is_prompt", False))
        if not is_prompt:
            return
        seq_lens = list(getattr(model_input, "seq_lens", []))
        query_lens = list(getattr(model_input, "query_lens", []))
        block_size = int(getattr(attn, "block_size", 0) or 0)

        # Slot mapping: non-zero means active token in this step
        slot_mapping = getattr(attn, "slot_mapping", None)
        assert slot_mapping is not None, "slot_mapping missing in attn_metadata"
        assert slot_mapping.shape == input_tokens.shape, "slot_mapping vs tokens shape mismatch"

        # Per-sample active token counts (no padding)
        non_pad_counts = (slot_mapping != 0).sum(dim=1).tolist()

        assert len(seq_lens) == bsz, f"seq_lens length {len(seq_lens)} != batch {bsz}"
        assert len(query_lens) == bsz or len(query_lens) == 0, "query_lens length invalid"

        # Pull seq ids and seq_data to cross-check lengths
        seq_groups = getattr(model_input, "sampling_metadata", None)
        assert seq_groups is not None, "sampling_metadata missing"
        seq_groups = getattr(seq_groups, "seq_groups", [])
        # Flatten seq_ids across groups preserving order
        flat_seq_ids = []
        flat_seq_datas = []
        for g in seq_groups:
            ids = list(getattr(g, "seq_ids", []))
            flat_seq_ids.extend(ids)
            data_map = dict(getattr(g, "seq_data", {}))
            for sid in ids:
                flat_seq_datas.append(data_map.get(sid))

        # Basic invariants per sample (assume one seq per sample for default vLLM batching)
        assert bsz == len(flat_seq_ids) or len(flat_seq_ids) == 0, "seq_ids vs batch size mismatch (multi-n seq_groups not supported here)"

        # Validate and track per-sample
        per_sample_logs = []
        for i in range(bsz):
            seq_len_i = int(seq_lens[i])
            q_len_i = int(query_lens[i]) if len(query_lens) == bsz else None
            active_i = int(non_pad_counts[i])

            # Positions range over active tokens for this step
            mask_i = slot_mapping[i] != 0
            if mask_i.any():
                pos_active = input_positions[i][mask_i]
                pos_min = int(pos_active.min().item())
                pos_max = int(pos_active.max().item())
            else:
                pos_min = None
                pos_max = None

            # Cross-check against seq_data (if available)
            seq_id = flat_seq_ids[i] if i < len(flat_seq_ids) else None
            seq_data = flat_seq_datas[i] if i < len(flat_seq_datas) else None
            if seq_data is not None and hasattr(seq_data, "prompt_token_ids"):
                try:
                    prompt_len = int(len(seq_data.prompt_token_ids))
                    # For prompts, seq_lens should equal total prompt length so far.
                    assert seq_len_i == prompt_len, f"seq_len ({seq_len_i}) != prompt_len ({prompt_len}) for seq_id={seq_id}"
                except Exception:
                    pass

            # Prefill-vs-decode checks
            num_prefills = int(getattr(attn, "num_prefills", 0))
            num_prefill_tokens = int(getattr(attn, "num_prefill_tokens", 0))
            num_decode_tokens = int(getattr(attn, "num_decode_tokens", 0))

            ## Active tokens this step should equal (prefill + decode) tokens reported
            #assert active_i == (num_prefill_tokens + num_decode_tokens), \
            #    f"active tokens {active_i} != prefill+decode {num_prefill_tokens + num_decode_tokens}"

            if is_prompt:
                # Chunked prefill step: expect q_len equals active tokens
                if q_len_i is not None:
                    assert q_len_i == active_i, f"query_len {q_len_i} != active tokens {active_i}"
                # Continuity: first chunk should start at 0; subsequent chunks continue
                tracker = self._chunked_prefill_trackers.get(seq_id, {
                    'steps': 0, 'last_seq_len': 0, 'chunks': [],
                    'last_min_pos': None, 'last_max_pos': None,
                    'last_virtual_engine': None,
                })
                if tracker['steps'] == 0:
                    if pos_min is not None:
                        assert pos_min == 0, f"first prefill pos_min {pos_min} != 0"
                else:
                    if pos_min is not None and tracker['last_max_pos'] is not None:
                        # Expect new chunk to start right after previous max
                        assert pos_min == (tracker['last_max_pos'] + 1), \
                            f"discontinuous prefill: pos_min {pos_min} vs last_max+1 {tracker['last_max_pos'] + 1}"

                # Monotonic growth of seq_len across steps
                assert seq_len_i >= tracker['last_seq_len'], \
                    f"seq_len decreased: {seq_len_i} < {tracker['last_seq_len']}"

                # Update tracker
                tracker['steps'] += 1
                tracker['last_seq_len'] = seq_len_i
                tracker['chunks'].append(active_i)
                tracker['last_min_pos'] = pos_min
                tracker['last_max_pos'] = pos_max
                tracker['last_virtual_engine'] = virtual_engine
                self._chunked_prefill_trackers[seq_id] = tracker

            # Extra sanity: padded length and block alignment (if applicable)
            assert padded_len >= active_i, "padded length smaller than active tokens"
            if block_size > 0:
                # Note: not all inputs are block aligned; log a warning if not aligned
                pass

            per_sample_logs.append({
                'seq_id': seq_id,
                'seq_len': seq_len_i,
                'query_len': q_len_i,
                'active_tokens': active_i,
                'pos_min': pos_min,
                'pos_max': pos_max,
                'num_prefills': num_prefills,
                'num_prefill_tokens': num_prefill_tokens,
                'num_decode_tokens': num_decode_tokens,
            })

        # Summary
        total_active = int((slot_mapping != 0).sum().item())
        total_token_slots = bsz * padded_len
        # Prefer attn counters for clarity
        total_prefill_tokens = int(getattr(attn, "num_prefill_tokens", 0))
        total_decode_tokens = int(getattr(attn, "num_decode_tokens", 0))

        # Build compact log lines
        header = (
            f"[MI] rank={world_rank} ve={virtual_engine} step={execute_step_count} "
            f"bsz={bsz} padded_len={padded_len} "
            f"active={total_active} prefill={total_prefill_tokens} decode={total_decode_tokens} "
            f"is_prompt={is_prompt} first_step={getattr(model_input, 'is_first_multi_step', False)} "
            f"last_step={getattr(model_input, 'is_last_step', False)}"
        )
        # Per-sample details
        details = []
        for i, d in enumerate(per_sample_logs):
            details.append(
                f"[rank={world_rank} ve={virtual_engine} step={execute_step_count} i={i} seq_id={d['seq_id']} seq_len={d['seq_len']} "
                f"q_len={d['query_len']} active={d['active_tokens']} "
                f"pos=[{d['pos_min']},{d['pos_max']}] "
                f"prefills={d['num_prefills']} pf_tokens={d['num_prefill_tokens']} "
                f"dec_tokens={d['num_decode_tokens']}]"
            )

        ## Optional: final assertion to ensure accounting matches
        #assert total_active == (total_prefill_tokens + total_decode_tokens), \
        #    "Total active tokens != prefill+decode tokens"

        # Emit logs
        if not should_log:
            return
        # Append same log line to per-rank file.
        try:
            #logger.info(header)
            with open(f"/workspace/world{world_rank}_model_input.txt", "a") as f:
                f.write(header + "\n")
                for line in details:
                    #logger.info(line)
                    f.write(line + "\n")
        except Exception:
            pass

        # Global aggregated stats across tracked sequence ids (per virtual engine)
        trackers = getattr(self, "_chunked_prefill_trackers", {})
        if trackers:
            prompt_lengths = [t['last_seq_len'] for t in trackers.values() if t.get('last_seq_len') is not None]
            chunk_counts = [len(t.get('chunks', [])) for t in trackers.values()]
            seq_count = len(prompt_lengths)
            if seq_count > 0:
                pl_min = min(prompt_lengths)
                pl_max = max(prompt_lengths)
                pl_mean = sum(prompt_lengths) / seq_count
                cc_min = min(chunk_counts)
                cc_max = max(chunk_counts)
                cc_mean = sum(chunk_counts) / seq_count
                global_line = (
                    f"[MI-GLOBAL] rank={world_rank} ve={virtual_engine} step={execute_step_count} "
                    f"seqs={seq_count} prompt_len_mean={pl_mean:.2f} prompt_len_min={pl_min} prompt_len_max={pl_max} "
                    f"chunks_mean={cc_mean:.2f} chunks_min={cc_min} chunks_max={cc_max}"
                )
            else:
                global_line = (
                    f"[MI-GLOBAL] rank={world_rank} ve={virtual_engine} step={execute_step_count} seqs=0"
                )
            # Append same log line to per-rank file.
            try:
                #logger.info(global_line)
                with open(f"/workspace/world{world_rank}_model_input.txt", "a") as f:
                    f.write(global_line + "\n")
            except Exception:
                pass

    def _last_prefill_or_decode(
        self,
        model_input,
        execute_model_req: ExecuteModelRequest,
    ) -> list[bool]:
        """
        Return per-sequence bool flags indicating whether current chunk
        completes the prompt (last prefill chunk).

        Criterion:
            curr_total_len (seq_lens_tensor[i]) == full prompt length.

        Assumes seq_lens_tensor order matches the flattened ordering of
        seq_ids across seq_group_metadata_list (same ordering used when
        building model_input tensors).
        """
        if not model_input.attn_metadata.is_prompt:
            # Decode step: sampling required for all sequences.
            num_seqs = 0
            if execute_model_req is not None:
                for sg in execute_model_req.seq_group_metadata_list:
                    num_seqs += len(sg.seq_data)
            return [True] * max(1, num_seqs)

        seq_lens = model_input.attn_metadata.seq_lens_tensor.tolist()

        # Reconstruct ordering
        ordered_seq_datas = []
        for sg_meta in execute_model_req.seq_group_metadata_list:
            for seq_id in sg_meta.seq_data:  # dict preserves insertion order
                ordered_seq_datas.append(sg_meta.seq_data[seq_id])

        assert len(ordered_seq_datas) == len(seq_lens), (
            "Sequence count mismatch vs seq_lens_tensor")

        flags = []
        for i, seq_data in enumerate(ordered_seq_datas):
            prompt_len = len(seq_data.prompt_token_ids)
            flags.append(seq_lens[i] == prompt_len)
        return flags
    
    def _debug_log_execute_model_req_patch(
        self,
        execute_step_count: int,
        execute_model_req: ExecuteModelRequest,
        execute_model_req_patch: dict,
        use_cached_base_req: bool,
        cached_seq_data: dict,
        phase: str,  # "before" or "after"
    ) -> None:
        """Rank-0 debug log for execute_model_req patching."""
        try:
            should_log = True
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_rank = get_world_group().rank_in_group
                should_log = (world_rank == 0)
            else:
                world_rank = 0
            if not should_log or execute_model_req is None:
                return

            ve = getattr(execute_model_req, "virtual_engine", None)
            seq_groups = getattr(execute_model_req, "seq_group_metadata_list", [])
            num_seq_groups = len(seq_groups)
            num_seqs = sum(len(sg.seq_data) for sg in seq_groups)
            is_patch = bool(execute_model_req_patch)

            header = (
                f"[EXEC-PATCH] rank={world_rank}, ve={ve}, step={execute_step_count}, phase={phase}, "
                f"use_cached_base_req={use_cached_base_req}, is_patch={is_patch}, "
                f"seq_groups={num_seq_groups}, seqs={num_seqs}"
            )

            # Patch summary
            patch_lines = []
            for key, pdata in execute_model_req_patch.items():
                parts = []
                for attr, val in pdata.items():
                    if attr == "sampling_params":
                        parts.append("sampling_params")
                        continue
                    if isinstance(val, (list, tuple, array.array)):
                        parts.append(f"{attr}+={len(val)}")
                    else:
                        parts.append(f"{attr}={val}")
                parts = ", ".join(parts)
                patch_lines.append("  seq_id={key}, patch={parts}")

            # Cache summary (lengths)
            cache_lines = []
            for key, cdata in cached_seq_data.items():
                def _len_safe(v):
                    if isinstance(v, (list, tuple, array.array)):
                        return len(v)
                    return v if isinstance(v, (int, float)) else (1 if v is not None else 0)
                cache_lines.append(
                    f"  cache seq_id={key}, lens("
                    f"cached={_len_safe(cdata.get('_cached_all_token_ids'))}, "
                    f"new={_len_safe(cdata.get('_new_appended_tokens'))}, "
                    f"out={_len_safe(cdata.get('_output_token_ids'))}, "
                    f"prompt={_len_safe(cdata.get('_prompt_token_ids'))}, "
                    f"prompt_tup={_len_safe(cdata.get('_prompt_token_ids_tuple'))}, "
                    f"num_comp={cdata.get('_num_computed_tokens', 0)})"
                )
            if not cache_lines:
                cache_lines.append("  (cache empty)")

            # Final seq_data snapshot
            final_lines = []
            for sg in seq_groups:
                for seq_id, sd in sg.seq_data.items():
                    def _len_attr(obj, name):
                        v = getattr(obj, name, None)
                        if isinstance(v, (list, tuple, array.array)):
                            return len(v)
                        return v
                    final_lines.append(
                        f"  final seq_id={seq_id}, prompt={_len_attr(sd,'prompt_token_ids')}, "
                        f"output={_len_attr(sd,'output_token_ids')}, "
                        f"cached_all={_len_attr(sd,'_cached_all_token_ids')}, "
                        f"new_appended={_len_attr(sd,'_new_appended_tokens')}, "
                        f"num_comp={getattr(sd,'_num_computed_tokens',None)}, "
                        f"stage={getattr(sd,'_stage',None)}"
                    )
            if not final_lines:
                final_lines.append("  (no seq_data)")

            with open(f"/workspace/world{world_rank}_exec_patch.txt", "a") as f:
                f.write(header + "\n")
                f.write("[          ] patch:\n")
                for line in patch_lines or ["  (no patch)"]:
                    f.write(line + "\n")
                f.write("[          ] cache:\n")
                for line in cache_lines:
                    f.write(line + "\n")
                f.write("[          ] seq_data:\n")
                for line in final_lines:
                    f.write(line + "\n")
        except Exception:
            pass
    
    def execute_model(
        self,
        execute_model_req: Optional[Union[ExecuteModelRequest, Tuple]] = None,
    ) -> Optional[List[SamplerOutput]]:
        # Timing: mark start & accumulate idle (global, not per VE)
        worker_execute_model_start = time.perf_counter()
        tracker = getattr(self, "_timing_tracker", None)
        if tracker is None:
            tracker = {}
            self._timing_tracker = tracker
        execute_step_count = 0
        if isinstance(execute_model_req, tuple):
            assert len(execute_model_req) == 4, (
                "execute_model_req must be a tuple of length 4, got "
                f"{len(execute_model_req)}")
            (execute_model_req, execute_model_req_patch,
             use_cached_base_req, execute_step_count) = execute_model_req

            if use_cached_base_req:
                ve = execute_model_req
                assert ve in self.cached_execute_model_req, (
                    f"Virtual engine {ve} not found in "
                    "cached_execute_model_req")
                execute_model_req = self.cached_execute_model_req[ve]

            if execute_model_req is not None:
                ve = execute_model_req.virtual_engine
                cached_seq_data = self.all_cached_seq_data.get(ve, {})
                self._debug_log_execute_model_req_patch(
                    execute_step_count,
                    execute_model_req,
                    execute_model_req_patch,
                    use_cached_base_req,
                    cached_seq_data,
                    phase="before",
                )
                self.all_cached_seq_data[ve] = (
                    self._apply_patch_to_execute_model_req(
                        execute_model_req,
                        cached_seq_data,
                        execute_model_req_patch,
                    ))
                self._debug_log_execute_model_req_patch(
                    execute_step_count,
                    execute_model_req,
                    execute_model_req_patch,
                    use_cached_base_req,
                    self.all_cached_seq_data[ve],
                    phase="after",
                )

        # VLLM_HPU_LOG_STEP_GRAPH_COMPILATION     - will log graph compilations per engine step, only when there was any - highly recommended to use alongside PT_HPU_METRICS_GC_DETAILS! # noqa:E501
        # VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL - will log graph compilations per engine step, always, even if there were none # noqa:E501
        # VLLM_HPU_LOG_STEP_CPU_FALLBACKS         - will log cpu fallbacks per engine step, only when there was any # noqa:E501
        # VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL     - will log cpu fallbacks per engine step, always, even if there were none # noqa:E501
        log_graph_compilation_all = os.environ.get(
            'VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL', '0') != '0'
        log_graph_compilation = os.environ.get(
            'VLLM_HPU_LOG_STEP_GRAPH_COMPILATION',
            '0') != '0' or log_graph_compilation_all
        log_cpu_fallbacks_all = os.environ.get(
            'VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL', '0') != '0'
        log_cpu_fallbacks = os.environ.get('VLLM_HPU_LOG_STEP_CPU_FALLBACKS',
                                           '0') != '0' or log_cpu_fallbacks_all
        if (log_graph_compilation or log_cpu_fallbacks) and \
            execute_model_req is not None:
            from habana_frameworks.torch.hpu.metrics import metric_localcontext
            seq_group_metadata_list = execute_model_req.seq_group_metadata_list
            is_prompt = any([
                seq_group_metadata.is_prompt
                for seq_group_metadata in seq_group_metadata_list
            ])
            # for dummy run in DP, we don't have real seq,
            # so use a dummy context_len here
            if len(seq_group_metadata_list) == 0:
                max_context_len = 128
            else:
                max_context_len = max([
                    max([
                        len(v.prompt_token_ids) + len(v.output_token_ids)
                        for v in seq_group_metadata.seq_data.values()
                    ]) for seq_group_metadata in seq_group_metadata_list
                ])  # whoa, that's some spicy stuff right here
            max_num_blocks = (
                (max_context_len - 1) // self.cache_config.block_size) + 1
            input_stats = (f'is_prompt: {is_prompt}, '
                           f'num_seqs: {len(seq_group_metadata_list)}, '
                           f'max_context_len: {max_context_len}, '
                           f'max_num_blocks {max_num_blocks}')
            gc_ctx = metric_localcontext(
                "graph_compilation"
            ) if log_graph_compilation else contextlib.nullcontext()
            cpu_fallback_ctx = metric_localcontext(
                "cpu_fallback"
            ) if log_cpu_fallbacks else contextlib.nullcontext()
            with gc_ctx as gc_local_metric, \
                cpu_fallback_ctx as cpu_fallback_local_metric:
                output, ve = self._execute_model(execute_model_req, execute_step_count)
            if (log_graph_compilation and gc_local_metric.stats()[0][1]
                    > 0) or log_graph_compilation_all:
                msg = ("VLLM_HPU_STEP_GRAPH_COMPILATION: "
                       f"{gc_local_metric.stats()}, {input_stats}")
                logger.warning(msg)
            if (log_cpu_fallbacks and cpu_fallback_local_metric.stats()[0][1]
                    > 0) or log_cpu_fallbacks_all:
                msg = ("VLLM_HPU_STEP_CPU_FALLBACK: "
                       f"{cpu_fallback_local_metric.stats()}, {input_stats}")
                logger.warning(msg)
        else:
            output, ve = self._execute_model(execute_model_req, execute_step_count)
        if execute_model_req is not None:
            self.cached_execute_model_req[
                execute_model_req.virtual_engine] = execute_model_req
            
        all_seq_ids = getattr(self, "_last_seq_ids", {})
        if ve in all_seq_ids:
            seq_ids = all_seq_ids[ve]
            for seq_id in seq_ids:
                if seq_id not in tracker:
                    tracker[seq_id] = {}
                if self.is_prompt not in tracker[seq_id]:
                    tracker[seq_id][self.is_prompt] = []

                entries = tracker[seq_id][self.is_prompt]

                data = {}
                step_gap = 0
                now = time.perf_counter()
                if len(tracker[seq_id][self.is_prompt]) > 0:
                    step_gap = worker_execute_model_start - tracker[seq_id][self.is_prompt][-1]['worker_execute_model_end']
                    data['worker_execute_model_dur'] = now - tracker[seq_id][self.is_prompt][-1]['worker_execute_model_end']
                else:
                    data['worker_execute_model_dur'] = now - worker_execute_model_start
                data['step_gap'] = step_gap
                data['worker_execute_model_start'] = worker_execute_model_start
                data['worker_execute_model_end'] = now
                
                # Bonus timings (may be missing on early steps; guard with get).
                bonus = self._bonus_info.get(ve, {})
                recv_done = bonus.get('recv_done')
                forward_send_done = bonus.get('forward_send_done')
                sample_done = bonus.get('sample_done')

                data['recv_done'] = recv_done
                data['forward_send_done'] = forward_send_done
                data['sample_done'] = sample_done

                # Derived durations (only compute when timestamps exist).
                recv_dur = None
                forward_send_dur = None
                sample_dur = None

                if recv_done is not None:
                    recv_dur = max(0.0, recv_done - data['worker_execute_model_start'])
                if recv_done is not None and forward_send_done is not None:
                    forward_send_dur = max(0.0, forward_send_done - recv_done)
                if forward_send_done is not None and sample_done is not None:
                    sample_dur = max(0.0, sample_done - forward_send_done)

                data['recv_dur'] = recv_dur
                data['forward_send_dur'] = forward_send_dur
                data['sample_dur'] = sample_dur

                entries.append(data)

                # Per-seq_id logging (avg + last) for current stage.
                stage_steps = len(entries)
                # Total steps across all stages for this seq_id.
                total_steps_all = sum(len(v) for v in tracker[seq_id].values())

                def avg(key):
                    vals = [e[key] for e in entries if e.get(key) is not None]
                    return (sum(vals) / len(vals)) if vals else None

                last = entries[-1]
                avg_dur = avg('worker_execute_model_dur')
                avg_gap = avg('step_gap')
                avg_recv = avg('recv_dur')
                avg_fwd_send = avg('forward_send_dur')
                avg_sample = avg('sample_dur')

                # Aggregate percentages (compute from summed raw times).
                sum_dur = sum(e['worker_execute_model_dur'] for e in entries)
                sum_gap = sum(e['step_gap'] for e in entries)
                sum_recv = sum(e['recv_dur'] for e in entries if e['recv_dur'] is not None)
                sum_fwd_send = sum(e['forward_send_dur'] for e in entries if e['forward_send_dur'] is not None)
                sum_sample = sum(e['sample_dur'] for e in entries if e['sample_dur'] is not None)
                pct_gap = (sum_gap / sum_dur * 100.0) if sum_dur > 0 and sum_gap > 0 else 0.0
                pct_recv = (sum_recv / sum_dur * 100.0) if sum_dur > 0 and sum_recv > 0 else 0.0
                pct_fwd_send = (sum_fwd_send / sum_dur * 100.0) if sum_dur > 0 and sum_fwd_send > 0 else 0.0
                pct_sample = (sum_sample / sum_dur * 100.0) if sum_dur > 0 and sum_sample > 0 else 0.0
                pct_unaccounted = max(0.0, 100.0 - (pct_gap + pct_recv + pct_fwd_send + pct_sample)) if sum_dur > 0 else 0.0

                stage_label = "PROMPT" if self.is_prompt is True else ("DECODE" if self.is_prompt is False else "NONE")
                world_rank = get_world_group().rank_in_group
                tp_rank = get_tp_group().rank_in_group
                pp_rank = get_pp_group().rank_in_group

                log_line = (
                    f"[SEQ_TIMING {stage_label}] world_rank={world_rank} tp_rank={tp_rank} pp_rank={pp_rank} "
                    f"ve={ve} seq_id={seq_id} stage_step={stage_steps} total_steps={total_steps_all} "
                    f"last_dur={last['worker_execute_model_dur']:.6f}s avg_dur={avg_dur:.6f}s "
                    f"last_gap={last['step_gap'] if last['step_gap'] is not None else 'None'} "
                    f"avg_gap={(avg_gap if avg_gap is not None else 'None')} "
                    f"last_recv={last['recv_dur'] if last['recv_dur'] is not None else 'None'} "
                    f"avg_recv={(avg_recv if avg_recv is not None else 'None')} "
                    f"last_forward_send={last['forward_send_dur'] if last['forward_send_dur'] is not None else 'None'} "
                    f"avg_forward_send={(avg_fwd_send if avg_fwd_send is not None else 'None')} "
                    f"last_sample={last['sample_dur'] if last['sample_dur'] is not None else 'None'} "
                    f"avg_sample={(avg_sample if avg_sample is not None else 'None')} "
                    f"pct_gap={pct_gap:.2f}% pct_recv={pct_recv:.2f}% pct_forward_send={pct_fwd_send:.2f}% pct_sample={pct_sample:.2f}% pct_unaccounted={pct_unaccounted:.2f}%"
                )

                #logger.info(
                #    log_line
                #)
                # Append same log line to per-rank file.
                try:
                    with open(f"/workspace/world{world_rank}_seq_timing.txt", "a") as f:
                        f.write(log_line + "\n")
                except Exception:
                    pass
        return output

    def _execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
        execute_step_count: int = 0,
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""

        # SAFE extraction of seq_ids (sampling_metadata may be None during idle / warmup loops).
        def _extract_seq_ids(mi):
            sm = getattr(mi, "sampling_metadata", None)
            if sm is None:
                return []
            seq_groups = getattr(sm, "seq_groups", None)
            if not seq_groups:
                return []
            flat = []
            for g in seq_groups:
                flat.extend(getattr(g, "seq_ids", []) or [])
            return flat

        with self.lock:
            if getattr(self, "_bonus_info", None) is None:
                self._bonus_info = {}
            if getattr(self, "_last_seq_ids", None) is None:
                self._last_seq_ids = {}
            inputs = self.prepare_input(execute_model_req)
            if inputs is None:
                return None, None

            model_input, worker_input, kwargs = inputs
            self.is_prompt = model_input.is_prompt
            self._debug_log_model_input(
                model_input,
                execute_step_count,
                getattr(model_input, "virtual_engine", None),
            )

            if envs.VLLM_ON_DEMAND_TORCH_PROFILER:
                if self.on_demand_profiler_mode == "p" and model_input.attn_metadata.is_prompt or \
                    self.on_demand_profiler_mode == "d" and not model_input.attn_metadata.is_prompt or \
                    self.on_demand_profiler_mode == "a":
                    self.on_demand_profiler_step_counter += 1
                    if self.on_demand_profiler_step_counter == self.on_demand_profiler_step_start:
                        self.do_start_profile()
                        self.on_demand_profiler_mode = "a"

            num_steps = worker_input.num_steps
            if (execute_model_req is not None
                    and execute_model_req.spec_step_idx):
                kwargs["spec_step_idx"] = execute_model_req.spec_step_idx

            self.execute_worker(worker_input)

            # If there is no input, we don't need to execute the model.
            if worker_input.num_seq_groups == 0:
                self.is_prompt = None
                return [], model_input.virtual_engine
            self._last_seq_ids[model_input.virtual_engine] = _extract_seq_ids(model_input)
            self._bonus_info[model_input.virtual_engine] = {}
            if not self._last_seq_ids[model_input.virtual_engine]:
                # Ensure no stale ids from previous step linger.
                self._last_seq_ids[model_input.virtual_engine] = []
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group(), deferred=True)
            self._bonus_info[model_input.virtual_engine]['recv_done'] = time.perf_counter()
            
            output = self.model_runner.execute_model(
                model_input=model_input,
                kv_caches=self.kv_cache[worker_input.virtual_engine]
                if self.kv_cache is not None else None,
                intermediate_tensors=intermediate_tensors,
                num_steps=num_steps,
                info=self._bonus_info[model_input.virtual_engine],
                execute_step_count=execute_step_count,
                **kwargs,
            )

            if not get_pp_group().is_last_rank:
                # output is IntermediateTensors
                assert isinstance(output, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    output.tensors, all_gather_group=get_tp_group())

                if envs.VLLM_ON_DEMAND_TORCH_PROFILER:
                    #self.do_step_profile()
                    if self.on_demand_profiler_step_counter == self.on_demand_profiler_step_stop:
                        self.do_stop_profile()

                self._bonus_info[model_input.virtual_engine]['forward_send_done'] = time.perf_counter()

                return [None], model_input.virtual_engine
        self._bonus_info[model_input.virtual_engine]['forward_send_done'] = time.perf_counter()

        if get_pp_group().is_last_rank and get_pp_group().world_size > 1:
            if model_input.needs_sampling:
                output = self.model_runner.execute_sample(
                    hidden_states=output,
                    model_input=model_input,
                    num_steps=num_steps,
                )
            elif get_tp_group().is_first_rank:
                # All sequences still in prefill: return dummy sampler outputs
                dummy_count = len(execute_model_req.seq_group_metadata_list)
                output = [
                    SamplerOutput(
                        outputs=[CompletionSequenceGroupOutput(samples=[], prompt_logprobs=None)],
                        sampled_token_probs=None,
                        sampled_token_ids=None,
                        spec_decode_worker_metrics=None,
                    )
                    for _ in range(dummy_count)
                ]
            else:
                output = [None]

        self._bonus_info[model_input.virtual_engine]['sample_done'] = time.perf_counter()

        if envs.VLLM_ON_DEMAND_TORCH_PROFILER:
            if self.on_demand_profiler_step_counter == self.on_demand_profiler_step_stop:
                self.do_stop_profile()

        # output is List[SamplerOutput]
        return output, model_input.virtual_engine

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        if is_fake_hpu():
            cache_block_size = self.get_cache_block_size_bytes()
            fake_hpu_cache_alloc = 4 * 2**30  # take 4 GiB flat on fake hpu
            num_fake_hpu_blocks = fake_hpu_cache_alloc // cache_block_size
            self.model_runner.bucketing_manager.num_hpu_blocks = \
                    num_fake_hpu_blocks
            return num_fake_hpu_blocks, 0
        with HabanaMemoryProfiler() as m:
            self.model_runner.profile_run()
            torch.hpu.synchronize()
        msg = ("Model profiling run "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        # At this point we should've allocated the maximum workspace for all
        # recipes we will use the extra memory for graphs/blocks
        free_hpu_memory = torch.hpu.mem_get_info()[0]

        cache_block_size = self.get_cache_block_size_bytes()
        graph_reserved_mem = (float(
            os.environ.get('VLLM_GRAPH_RESERVED_MEM', '0.1'))
                              if not self.model_config.enforce_eager else 0)
        graph_headroom = 1 - graph_reserved_mem
        available_hpu_memory = free_hpu_memory * \
            self.cache_config.gpu_memory_utilization
        hpu_memory_margin = free_hpu_memory * (
            1 - self.cache_config.gpu_memory_utilization)
        self.model_runner.mem_margin = hpu_memory_margin
        cache_size_bytes = available_hpu_memory * graph_headroom
        graph_headroom_bytes = available_hpu_memory * (1 - graph_headroom)
        msg = (
            f"Free device memory: {format_bytes(free_hpu_memory)}, "
            f"{format_bytes(available_hpu_memory)} usable "
            f"(gpu_memory_utilization={self.cache_config.gpu_memory_utilization}),"
            f" {format_bytes(graph_headroom_bytes)} reserved for HPUGraphs "
            f"(VLLM_GRAPH_RESERVED_MEM={graph_reserved_mem}), "
            f"{format_bytes(cache_size_bytes)} reserved for KV cache")
        logger.info(msg)
        num_hpu_blocks = int(cache_size_bytes // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_hpu_blocks = max(num_hpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()

        gc.collect()
        return num_hpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(
            num_gpu_blocks, self.cache_config.block_size,
            self.model_config.max_model_len,
            self.parallel_config.pipeline_parallel_size)
        target_gpu_blocks = int(
            num_gpu_blocks // (self.parallel_config.pipeline_parallel_size + envs.VLLM_PP_BONUS_VE))
        target_cpu_blocks = int(
            num_cpu_blocks // (self.parallel_config.pipeline_parallel_size + envs.VLLM_PP_BONUS_VE))
        self.cache_config.num_gpu_blocks = target_gpu_blocks * (self.parallel_config.pipeline_parallel_size + envs.VLLM_PP_BONUS_VE)
        self.cache_config.num_cpu_blocks = target_cpu_blocks * (self.parallel_config.pipeline_parallel_size + envs.VLLM_PP_BONUS_VE)

        self.model_runner.bucketing_manager.num_hpu_blocks = target_gpu_blocks
        self.model_runner.bucketing_manager.generate_prompt_buckets()
        if not self.model_runner.is_pooler:
            self.model_runner.bucketing_manager.generate_decode_buckets()

        with HabanaMemoryProfiler() as m:
            self._init_cache_engine()
            torch.hpu.synchronize()
        msg = ("Initializing cache engine "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            HPUCacheEngine(self.cache_config, self.model_config,
                           self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size + envs.VLLM_PP_BONUS_VE)
        ]
        self.hpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size + envs.VLLM_PP_BONUS_VE)
        ]
        bind_kv_cache(self.compilation_config.static_forward_context,
                      self.hpu_cache)

    def _warm_up_model(self) -> None:
        # NOTE(kzawora): We should use virtual engine index here
        # for pipeline parallelism. Using 0 for now.
        if not isinstance(self.model_runner, HPUPoolingModelRunner):
            assert self.hpu_cache is not None
            self.model_runner.warmup_model(self.hpu_cache[0])
        else:
            self.model_runner.warmup_model(None)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.hpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def list_prompt_adapters(self) -> Set[int]:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def shutdown(self):
        getattr(self.model_runner, 'shutdown_inc', lambda: None)()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return HPUCacheEngine.get_cache_block_size(self.cache_config,
                                                   self.model_config,
                                                   self.parallel_config)


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    backend = hpu_backend_string()
    init_distributed_environment(parallel_config.world_size,
                                 rank,
                                 distributed_init_method,
                                 local_rank,
                                 backend=backend)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    if parallel_config.pipeline_parallel_size > 1 and \
        not envs.VLLM_PP_USE_CPU_COMS:
        # torch-ccl hpu need a collective API warm up
        # before calling send/recv API
        get_pp_group().all_reduce(torch.zeros(1).to('hpu'))
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        expected_size = parallel_config.world_size *\
            parallel_config.data_parallel_size
        if torch_world_size != expected_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size * "
                "parallel_config.data_parallel_size "
                f"({torch_world_size} vs. {expected_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        backend = hpu_backend_string()
        torch.distributed.init_process_group(
            backend=backend,
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup & checking conformance.
    device = hpu_device_string()
    dummy_tensor_hpu = torch.ones(1).to(device)
    if not envs.VLLM_PP_USE_CPU_COMS:
        torch.distributed.all_reduce(dummy_tensor_hpu)
        assert dummy_tensor_hpu.item(
        ) == parallel_config.world_size * parallel_config.data_parallel_size
    else:
        get_tp_group().all_reduce(dummy_tensor_hpu)
        assert dummy_tensor_hpu.item() == parallel_config.tensor_parallel_size
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
    ensure_kv_transfer_initialized(vllm_config)


def raise_if_cache_size_invalid(num_gpu_blocks, block_size, max_model_len,
                                pipeline_parallel_size) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * (num_gpu_blocks // (pipeline_parallel_size + envs.VLLM_PP_BONUS_VE))
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")


class HPUCacheEngine(CacheEngine):

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        k_cache_shape = kv_cache_shape
        v_cache_shape = None if self.model_config.use_mla else kv_cache_shape
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        dtype = self.dtype
        if device != 'hpu' and not is_fake_hpu() \
          and self.dtype == torch.float8_e4m3fn:
            dtype = torch.uint8
        for _ in range(self.num_attention_layers):
            key_cache = torch.zeros(k_cache_shape, dtype=dtype, device=device)
            if v_cache_shape is not None:
                value_cache = torch.zeros(v_cache_shape,
                                          dtype=dtype,
                                          device=device)
            else:
                value_cache = None
            kv_layer = (key_cache, value_cache)
            kv_cache.append(kv_layer)
        return kv_cache
