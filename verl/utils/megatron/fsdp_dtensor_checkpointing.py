# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, default_planner

logger = logging.getLogger(__file__)

try:
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        preprocess_state_dict_for_uneven_dtensor,
    )
    from megatron.core.transformer.fsdp_dtensor_checkpoint import (
        handle_experts_in_state_dict,
        handle_fp8_extra_state_case,
        handle_swiglu_in_state_dict,
        print_diff_in_state_dicts,
    )
    from megatron.core.utils import get_model_config

    HAVE_MEGATRON_FSDP_CHECKPOINTING = True
except ImportError:
    HAVE_MEGATRON_FSDP_CHECKPOINTING = False

try:
    from megatron.core.transformer.fsdp_dtensor_checkpoint import handle_gdn_in_state_dict
except ImportError:
    handle_gdn_in_state_dict = None


def _require_megatron_fsdp_checkpointing() -> None:
    if not HAVE_MEGATRON_FSDP_CHECKPOINTING:
        raise RuntimeError("Megatron-FSDP checkpointing helpers are not available in this Megatron-Core build.")


class _FsdpDtensorModelAdapter:
    """Adapt verl's FSDP->Float16Module->GPTModel nesting to MCore checkpoint helpers."""

    def __init__(self, model: Any):
        self._model = model

    def get_parameter(self, target: str):
        candidates = [target]
        if target.startswith("module."):
            candidates.append(f"module.module.{target[len('module.') :]}")
        else:
            candidates.append(f"module.{target}")
            candidates.append(f"module.module.{target}")

        last_error = None
        for candidate in candidates:
            try:
                return self._model.get_parameter(candidate)
            except AttributeError as error:
                last_error = error
        raise last_error

    def named_modules(self, *args, **kwargs):
        return self._model.named_modules(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


def preprocess_fsdp_dtensor_state_dict(transformer_config: Any, raw_state_dict: dict[str, Any], model: Any):
    """Preprocess Megatron-FSDP DTensor state before PyTorch DCP save/load."""
    _require_megatron_fsdp_checkpointing()

    state_dict = raw_state_dict.copy()
    if "model" not in state_dict:
        preprocess_state_dict_for_uneven_dtensor(state_dict)
        return state_dict

    model_state_dict = state_dict["model"]
    handle_fp8_extra_state_case(model_state_dict)

    checkpoint_model = _FsdpDtensorModelAdapter(model)

    try:
        model_config = get_model_config(checkpoint_model)
    except Exception:
        model_config = transformer_config

    is_swiglu = (
        getattr(model_config, "gated_linear_unit", False) and getattr(model_config, "activation_func", None) is F.silu
    )
    if is_swiglu:
        if "optimizer" in state_dict:
            model_state_dict, optimizer_state_dict = handle_swiglu_in_state_dict(
                checkpoint_model, state_dict["model"], state_dict["optimizer"]
            )
            state_dict["model"] = model_state_dict
            state_dict["optimizer"] = optimizer_state_dict
        else:
            model_state_dict, _ = handle_swiglu_in_state_dict(checkpoint_model, state_dict["model"], None)
            state_dict["model"] = model_state_dict

    if handle_gdn_in_state_dict is not None:
        if "optimizer" in state_dict:
            model_state_dict, optimizer_state_dict = handle_gdn_in_state_dict(
                checkpoint_model, state_dict["model"], state_dict["optimizer"]
            )
            state_dict["model"] = model_state_dict
            state_dict["optimizer"] = optimizer_state_dict
        else:
            model_state_dict, _ = handle_gdn_in_state_dict(checkpoint_model, state_dict["model"], None)
            state_dict["model"] = model_state_dict

    num_experts = getattr(model_config, "num_moe_experts", None)
    if num_experts:
        state_dict["model"] = handle_experts_in_state_dict(state_dict["model"], num_experts)

    preprocess_state_dict_for_uneven_dtensor(state_dict)
    return state_dict


def save_fsdp_dtensor_checkpointing(state_dict: dict[str, Any], ckpt_path: str):
    _require_megatron_fsdp_checkpointing()
    storage_writer = FileSystemWriter(ckpt_path)
    torch.distributed.checkpoint.save(state_dict=state_dict, storage_writer=storage_writer)


def load_fsdp_dtensor_checkpointing(state_dict: dict[str, Any], ckpt_path: str):
    _require_megatron_fsdp_checkpointing()
    storage_reader = FileSystemReader(ckpt_path)
    metadata = storage_reader.read_metadata().state_dict_metadata
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        logger.info("Loading Megatron-FSDP DTensor checkpoint with partial-load diagnostics enabled.")
    print_diff_in_state_dicts(metadata, state_dict)
    planner = default_planner.DefaultLoadPlanner(allow_partial_load=True)
    torch.distributed.checkpoint.load_state_dict(
        state_dict=state_dict,
        storage_reader=storage_reader,
        planner=planner,
    )
    return state_dict
