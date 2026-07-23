# Copyright 2025 The llm-d Authors.
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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.kv_offload.base import OffloadingSpec

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.spec import SharedStorageOffloadingSpec


@pytest.mark.no_cuda_required
def test_gpu_blocks_per_file_uses_gpu_block_granularity(monkeypatch):
    """File grouping must match vLLM's GPU-block-based scheduler factor."""
    extra_config = {"block_size": 256}

    def mock_offloading_spec_init(self, vllm_config, kv_cache_config):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.extra_config = extra_config
        self.gpu_block_size = (64,)
        self.hash_block_size = 16
        self.block_size_factor = 1

    monkeypatch.setattr(OffloadingSpec, "__init__", mock_offloading_spec_init)

    file_mapper = MagicMock()
    from_vllm_config = MagicMock(return_value=file_mapper)
    monkeypatch.setattr(FileMapper, "from_vllm_config", from_vllm_config)

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        world_size=1,
    )
    vllm_config = SimpleNamespace(parallel_config=parallel_config)

    spec = SharedStorageOffloadingSpec(vllm_config, MagicMock())

    assert spec.block_size_factor == 4
    assert spec.gpu_blocks_per_file == 4
    assert (
        from_vllm_config.call_args.kwargs["gpu_blocks_per_file"]
        == spec.block_size_factor
    )
