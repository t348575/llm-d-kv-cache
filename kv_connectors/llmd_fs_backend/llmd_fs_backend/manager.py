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

import os
from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
)

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec

logger = init_logger(__name__)


class SharedStorageOffloadingManager(OffloadingManager):
    """
    SharedStorageOffloadingManager manages KV offloading to a shared storage medium.
    """

    def __init__(self, file_mapper: FileMapper) -> None:
        self.file_mapper: FileMapper = file_mapper

    # ----------------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------------
    def lookup(self, keys: Iterable[OffloadKey]) -> int | None:
        """
        Return how many consecutive blocks from the start are already offloaded.
        """
        hit_count = 0
        for key in keys:
            file_path = self.file_mapper.get_file_name(key)
            if not os.path.exists(file_path):
                break
            hit_count += 1
        return hit_count

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------
    def prepare_load(self, keys: Iterable[OffloadKey]) -> LoadStoreSpec:
        """
        For shared storage, loading is stateless - return specs that point to files.
        """
        return SharedStorageLoadStoreSpec(keys)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        """
        Update access times if desired.
        Shared storage version does nothing here because updates are handled
        by the file thread for performance reasons.
        """
        pass

    def complete_load(self, keys: Iterable[OffloadKey]) -> None:
        """Stateless load - no post-load action needed."""
        pass

    # ----------------------------------------------------------------------
    # Store
    # ----------------------------------------------------------------------
    def prepare_store(self, keys: Iterable[OffloadKey]) -> PrepareStoreOutput | None:
        """
        Prepare storing new blocks.
        Shared storage always accepts new blocks. Eviction is not needed.
        If a file already exists, the file thread handles it.
        """
        keys_to_store = list(keys)

        # Set up store spec
        store_spec = SharedStorageLoadStoreSpec(keys_to_store)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=[],  # no eviction needed
        )

    def complete_store(self, keys: Iterable[OffloadKey], success: bool = True) -> None:
        """
        For shared storage, storing is stateless - no action needed.
        """
        pass

    def shutdown(self) -> None:
        return
