# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group, get_world_group, get_pp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)

def world_broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict

    #return get_world_group().broadcast_tensor_dict(tensor_dict, src)

    # Only TP leaders participate in the PP chain.
    if get_tp_group().is_first_rank:
        if get_pp_group().is_first_rank and get_pp_group().is_last_rank:
            pass
        elif get_pp_group().is_last_rank:
            get_pp_group().send_object(tensor_dict, dst=get_pp_group().rank_in_group - 1)
        elif not get_pp_group().is_first_rank:
            tensor_dict = get_pp_group().recv_object(src=get_pp_group().rank_in_group + 1)
            get_pp_group().send_object(tensor_dict, dst=get_pp_group().rank_in_group - 1)
        else:
            tensor_dict = get_pp_group().recv_object(src=get_pp_group().rank_in_group + 1)

    # Barrier to ensure the PP chain is complete.
    get_pp_group().barrier()

    # Step 2: Within each PP group, FP leaders broadcast the tensor dict to the TP group.
    if get_tp_group().is_first_rank:
        broadcast_tensor_dict(tensor_dict, src=0)
    else:
        tensor_dict = broadcast_tensor_dict(src=0)

    # Final barrier to synchronize all processes.
    get_pp_group().barrier()
    return tensor_dict
