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
import time
from collections.abc import Iterable

from simple_profiler import profiler

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
        start_ns = time.perf_counter_ns()
        hit_count = 0
        checked = 0
        for key in keys:
            checked += 1
            file_name_start_ns = time.perf_counter_ns()
            file_path = self.file_mapper.get_file_name(key)
            if profiler._active:
                profiler.add_event(
                    "storage_manager.lookup_file_name",
                    "kv_offload",
                    file_name_start_ns,
                    time.perf_counter_ns() - file_name_start_ns,
                )
            exists_start_ns = time.perf_counter_ns()
            exists = os.path.exists(file_path)
            if profiler._active:
                profiler.add_event(
                    "storage_manager.lookup_exists_one",
                    "kv_offload",
                    exists_start_ns,
                    time.perf_counter_ns() - exists_start_ns,
                    args={"hit": exists},
                )
            if not exists:
                break
            hit_count += 1
        if profiler._active:
            profiler.add_event(
                "storage_manager.lookup_exists",
                "kv_offload",
                start_ns,
                time.perf_counter_ns() - start_ns,
                args={"checked_keys": checked, "hit_count": hit_count},
            )
        return hit_count

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------
    def prepare_load(self, keys: Iterable[OffloadKey]) -> LoadStoreSpec:
        """
        For shared storage, loading is stateless - return specs that point to files.
        """
        start_ns = time.perf_counter_ns()
        spec = SharedStorageLoadStoreSpec(keys)
        if profiler._active:
            profiler.add_event(
                "storage_manager.prepare_load_spec",
                "kv_offload",
                start_ns,
                time.perf_counter_ns() - start_ns,
                args={"num_keys": len(spec.keys)},
            )
        return spec

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
        Shared storage stores only blocks that do not already exist on disk.
        Eviction is not needed.
        """
        start_ns = time.perf_counter_ns()
        checked = 0
        keys_to_store = []
        for key in keys:
            checked += 1
            file_name_start_ns = time.perf_counter_ns()
            file_path = self.file_mapper.get_file_name(key)
            if profiler._active:
                profiler.add_event(
                    "storage_manager.prepare_store_file_name",
                    "kv_offload",
                    file_name_start_ns,
                    time.perf_counter_ns() - file_name_start_ns,
                )
            exists_start_ns = time.perf_counter_ns()
            exists = os.path.exists(file_path)
            if profiler._active:
                profiler.add_event(
                    "storage_manager.prepare_store_exists_one",
                    "kv_offload",
                    exists_start_ns,
                    time.perf_counter_ns() - exists_start_ns,
                    args={"exists": exists},
                )
            if not exists:
                keys_to_store.append(key)

        if not keys_to_store:
            if profiler._active:
                profiler.add_event(
                    "storage_manager.prepare_store_exists",
                    "kv_offload",
                    start_ns,
                    time.perf_counter_ns() - start_ns,
                    args={
                        "checked_keys": checked,
                        "keys_to_store": 0,
                        "result": "already_stored",
                    },
                )
            return None

        # Set up store spec
        store_spec = SharedStorageLoadStoreSpec(keys_to_store)

        output = PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=[],  # no eviction needed
        )
        if profiler._active:
            profiler.add_event(
                "storage_manager.prepare_store_exists",
                "kv_offload",
                start_ns,
                time.perf_counter_ns() - start_ns,
                args={
                    "checked_keys": checked,
                    "keys_to_store": len(keys_to_store),
                    "result": "prepared",
                },
            )
        return output

    def complete_store(self, keys: Iterable[OffloadKey], success: bool = True) -> None:
        """
        For shared storage, storing is stateless - no action needed.
        """
        pass

    def shutdown(self) -> None:
        return
