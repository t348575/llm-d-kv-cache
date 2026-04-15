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

import math
import os
import time
from collections.abc import Sequence

import storage_offload
import torch
from simple_profiler import profiler
from vllm.logger import init_logger
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCaches
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.stats import connector_stats

logger = init_logger(__name__)

# ----------------------------------------------------------------------
# Base Storage Offloading Handler
# ----------------------------------------------------------------------
DEFAULT_MAX_STAGING_MEMORY_GB = 150
DEFAULT_THREADS_PER_GPU = 64
DEFAULT_READ_PREFERRING_WORKERS_RATIO = 0.75


class BaseStorageOffloadingHandler(OffloadingHandler):
    """
    BaseStorageOffloadingHandler handles transfers for both directions,
    either GPU->Storage (PUT) or Storage->GPU (GET).
    """

    def __init__(
        self,
        gpu_blocks_per_file: int,
        file_mapper: FileMapper,
        engine: storage_offload.StorageOffloadEngine,
        transfer_jobs: dict,
    ):
        """
        Initialize a SingleStorageDirectionOffloadingHandler.

        Args:
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file.
            file_mapper: The FileMapper mapping blocks to files.
            engine: the storage engine.
            transfer_jobs: Shared dict mapping job_id -> (wall_start_ns, profile_tid, req_id, direction).
                           Must be shared between all handlers using the same engine so that
                           whichever handler drains engine.get_finished() first can emit profiling
                           for all job types.
        """
        self.file_mapper = file_mapper
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.engine = engine
        self._transfer_jobs = transfer_jobs

    def get_finished(self) -> list[TransferResult]:
        """
        Poll finished async transfers.

        Returns:
            List of completed transfer results.
        """

        def merged_duration_ns(samples: list[tuple[int, int, int]]) -> int:
            intervals = sorted(
                (start_ns, start_ns + duration_ns)
                for start_ns, duration_ns, _num_bytes in samples
                if duration_ns > 0
            )
            if not intervals:
                return 0

            merged_start, merged_end = intervals[0]
            merged_duration = 0
            for start_ns, end_ns in intervals[1:]:
                if start_ns <= merged_end:
                    merged_end = max(merged_end, end_ns)
                else:
                    merged_duration += merged_end - merged_start
                    merged_start, merged_end = start_ns, end_ns
            return merged_duration + (merged_end - merged_start)

        finished_tuples = self.engine.get_finished()
        results = []
        for (
            job_id,
            success,
            num_bytes,
            file_io_samples,
            cuda_copy_samples,
        ) in finished_tuples:
            end_ns = time.perf_counter_ns()
            job_meta = self._transfer_jobs.pop(job_id, None)
            if job_meta is not None:
                wall_start_ns, profile_tid, req_id, direction = job_meta
                is_store = direction == "gpu_to_storage"
                file_io_samples = sorted(file_io_samples)
                cuda_copy_samples = sorted(cuda_copy_samples)
                file_io_wall_ns = merged_duration_ns(file_io_samples)
                cuda_copy_wall_ns = merged_duration_ns(cuda_copy_samples)
                # Compute bandwidth from wall-clock span
                file_io_bw_gbps = (
                    (num_bytes / file_io_wall_ns) if file_io_wall_ns > 0 else 0.0
                )
                cuda_copy_bw_gbps = (
                    (num_bytes / cuda_copy_wall_ns) if cuda_copy_wall_ns > 0 else 0.0
                )
                job_args = {
                    "req_id": req_id,
                    "success": success,
                    "num_bytes": num_bytes,
                    "file_io_bw_GBps": round(file_io_bw_gbps, 3),
                    "cuda_copy_bw_GBps": round(cuda_copy_bw_gbps, 3),
                }
                profiler.add_event(
                    name=f"fs_transfer({direction}, job={job_id})",
                    category="fs_transfer",
                    start_ns=wall_start_ns,
                    duration_ns=end_ns - wall_start_ns,
                    tid=profile_tid,
                    args=job_args,
                )
                for sample_idx, (start_ns, duration_ns, sample_num_bytes) in enumerate(
                    cuda_copy_samples
                ):
                    profiler.add_event(
                        name=(
                            f"cuda_staging({'gpu_to_cpu' if is_store else 'cpu_to_gpu'}, "
                            f"job={job_id}, sample={sample_idx})"
                        ),
                        category="cuda",
                        start_ns=start_ns,
                        duration_ns=duration_ns,
                        tid=profile_tid,
                        args={
                            "req_id": req_id,
                            "num_bytes": sample_num_bytes,
                            "bw_GBps": round(
                                (sample_num_bytes / duration_ns)
                                if duration_ns > 0
                                else 0.0,
                                3,
                            ),
                        },
                    )
                for sample_idx, (start_ns, duration_ns, sample_num_bytes) in enumerate(
                    file_io_samples
                ):
                    profiler.add_event(
                        name=(
                            f"file_{'write' if is_store else 'read'}"
                            f"(job={job_id}, sample={sample_idx})"
                        ),
                        category="fs",
                        start_ns=start_ns,
                        duration_ns=duration_ns,
                        tid=profile_tid,
                        args={
                            "req_id": req_id,
                            "num_bytes": sample_num_bytes,
                            "bw_GBps": round(
                                (sample_num_bytes / duration_ns)
                                if duration_ns > 0
                                else 0.0,
                                3,
                            ),
                        },
                    )
            results.append(
                TransferResult(job_id=job_id, success=success, transfer_size=num_bytes)
            )
        return results

    def wait(self, job_ids: set[int]):
        """
        Block until the specified transfer jobs complete.

        Args:
            job_ids: Set of job IDs to wait for.
        """
        for job_id in job_ids:
            self.engine.wait_job(job_id)

    def shutdown(self) -> None:
        pending_job_ids = set(self._transfer_jobs)
        if pending_job_ids:
            self.wait(pending_job_ids)
            self._transfer_jobs.clear()

    def _build_file_block_mapping(
        self,
        keys: Sequence[bytes],
        block_ids: Sequence[int],
        group_sizes: Sequence[int],
        block_indices: Sequence[int] | None = None,
    ):
        """
        Build per-file block ID lists for grouped transfers.

        Returns:
            tuple[list[str], list[list[int]]]
                - file paths
                - per-file block ID lists
        """
        files: list[str] = []
        per_file_block_ids: list[Sequence[int]] = []
        key_idx = 0
        block_id_idx = 0

        for group_idx, group_size in enumerate(group_sizes):
            first_block_index = 0
            if block_indices is not None:
                first_block_index = block_indices[group_idx] % self.gpu_blocks_per_file

            num_group_keys = math.ceil(
                (group_size + first_block_index) / self.gpu_blocks_per_file
            )
            group_keys = keys[key_idx : key_idx + num_group_keys]
            group_block_ids = block_ids[block_id_idx : block_id_idx + group_size]

            start = 0
            chunk_size = self.gpu_blocks_per_file - first_block_index
            for key in group_keys:
                end = min(start + chunk_size, len(group_block_ids))
                files.append(self.file_mapper.get_file_name(key))
                per_file_block_ids.append(group_block_ids[start:end])
                start = end
                chunk_size = self.gpu_blocks_per_file

            key_idx += num_group_keys
            block_id_idx += group_size

        assert key_idx == len(keys)
        assert block_id_idx == len(block_ids)

        return files, per_file_block_ids


class GPUToStorageHandler(BaseStorageOffloadingHandler):
    """Handler for GPU -> Storage (PUT) transfers."""

    def transfer_async(
        self,
        job_id: int,
        spec: TransferSpec,
        profile_tid: str = "kv_store",
        req_id: str = "",
    ) -> bool:
        """
        Launch an asynchronous transfer GPU -> Storage.

        Args:
            job_id: Unique identifier for the transfer job.
            spec: Transfer specification describing source and destination
                block IDs and file hashes.

        Returns:
            True if the transfer was successfully submitted.
        """
        src_spec, dst_spec = spec
        assert isinstance(src_spec, GPULoadStoreSpec)
        assert isinstance(dst_spec, SharedStorageLoadStoreSpec)

        dst_files, per_file_block_ids = self._build_file_block_mapping(
            keys=dst_spec.keys,
            block_ids=src_spec.block_ids,
            group_sizes=src_spec.group_sizes,
        )

        wall_start_ns = time.perf_counter_ns()
        success = self.engine.async_store_gpu_blocks(
            job_id, dst_files, per_file_block_ids
        )
        connector_stats.record_publish_submit(
            file_count=len(dst_files),
            latency_ns=time.perf_counter_ns() - wall_start_ns,
            success=success,
        )
        if success:
            self._transfer_jobs[job_id] = (
                wall_start_ns,
                profile_tid,
                req_id,
                "gpu_to_storage",
            )
        return success


class StorageToGPUHandler(BaseStorageOffloadingHandler):
    """Handler for asynchronous transfers from storage to GPU."""

    def transfer_async(
        self,
        job_id: int,
        spec: TransferSpec,
        profile_tid: str = "kv_load",
        req_id: str = "",
    ) -> bool:
        """
        Launch an asynchronous transfer Storage -> GPU.

        Args:
            job_id: Unique identifier for the transfer job.
            spec: Transfer specification describing source and destination
                block IDs and file hashes.

        Returns:
            True if the transfer was successfully submitted.
        """
        src_spec, dst_spec = spec
        assert isinstance(src_spec, SharedStorageLoadStoreSpec)
        assert isinstance(dst_spec, GPULoadStoreSpec)

        src_files, per_file_block_ids = self._build_file_block_mapping(
            keys=src_spec.keys,
            block_ids=dst_spec.block_ids,
            group_sizes=dst_spec.group_sizes,
            block_indices=dst_spec.block_indices,
        )

        wall_start_ns = time.perf_counter_ns()
        success = self.engine.async_load_gpu_blocks(
            job_id, src_files, per_file_block_ids
        )
        connector_stats.record_load_submit(
            file_count=len(src_files),
            latency_ns=time.perf_counter_ns() - wall_start_ns,
            success=success,
        )
        if success:
            self._transfer_jobs[job_id] = (
                wall_start_ns,
                profile_tid,
                req_id,
                "storage_to_gpu",
            )
        return success


class StorageOffloadingHandlers:
    """Base handler with common helpers for Storage offloading."""

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        file_mapper: FileMapper,
        gpu_block_size: int,
        gpu_blocks_per_file: int,
        threads_per_gpu: int,
        max_staging_memory_gb: int = DEFAULT_MAX_STAGING_MEMORY_GB,
        read_preferring_ratio: float = DEFAULT_READ_PREFERRING_WORKERS_RATIO,
        use_odirect: bool = False,
    ):
        threads_per_gpu = min(threads_per_gpu, os.cpu_count() or 1)
        tensors = StorageOffloadingHandlers._get_tensors(kv_caches)
        assert tensors

        # Compute staging memory buffer size
        buffer_size_mb = self._compute_buffer_size_mb(tensors, gpu_blocks_per_file)

        # Adjust threads_per_gpu if exceeding max_staging_memory_gb
        if buffer_size_mb * threads_per_gpu > max_staging_memory_gb * 1024:
            threads_per_gpu = min(
                threads_per_gpu, int(max_staging_memory_gb * 1024 / buffer_size_mb)
            )
            logger.warning(
                f"Adjusted threads_per_gpu to {threads_per_gpu} due to "
                f"max_staging_memory_gb {max_staging_memory_gb} "
                f"limit (buffer_size_mb={buffer_size_mb})."
            )

        # Calculate number of read-preferring workers
        read_preferring_workers = max(1, int(threads_per_gpu * read_preferring_ratio))

        # Initialize storage offload resources for async transfers
        self.engine = storage_offload.StorageOffloadEngine(
            io_threads=threads_per_gpu,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tensors=tensors,
            read_preferring_workers=read_preferring_workers,
            use_odirect=use_odirect,
        )

        logger.info(
            f"StorageOffloadingHandlers: "
            f"threads_per_gpu={threads_per_gpu},"
            f"offloading block_size={gpu_blocks_per_file * gpu_block_size}, "
            f"staging_buffer_size_mb={buffer_size_mb}, "
            f"max_staging_memory_gb={max_staging_memory_gb}, "
            f"read_preferring_workers={read_preferring_workers}, "
            f"use_odirect={use_odirect}, "
        )

        # Shared transfer_jobs dict so whichever handler drains engine.get_finished()
        # first can emit profiling events for all job types (store and load).
        shared_transfer_jobs: dict[int, tuple] = {}

        self.gpu_to_storage_handler = GPUToStorageHandler(
            engine=self.engine,
            file_mapper=file_mapper,
            gpu_blocks_per_file=gpu_blocks_per_file,
            transfer_jobs=shared_transfer_jobs,
        )

        self.storage_to_gpu_handler = StorageToGPUHandler(
            engine=self.engine,
            file_mapper=file_mapper,
            gpu_blocks_per_file=gpu_blocks_per_file,
            transfer_jobs=shared_transfer_jobs,
        )

    def _compute_buffer_size_mb(
        self,
        tensors: list[torch.Tensor],
        gpu_blocks_per_file: int,
    ):
        """
        Estimate staging memory size in MB, applying min/max limits.

        Args:
            tensors: List of KV-cache tensors used to infer per-block memory usage.
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file.

        Returns:
            Estimated staging buffer size in megabytes.
        """
        per_block_size_in_bytes = sum(tensor.shape[1] for tensor in tensors)
        file_size_in_bytes = per_block_size_in_bytes * gpu_blocks_per_file
        file_size_mb = math.ceil(file_size_in_bytes / (1 << 20))
        return file_size_mb

    @staticmethod
    def _get_tensors(
        kv_caches: CanonicalKVCaches,
    ) -> list[torch.Tensor]:
        """
        Canonicalize the given KV caches to 2D int8 tensors with shape
        (num_blocks, page_size_bytes).

        Returns:
            list_of_kv_cache_tensors
        """
        tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            page_size_bytes = kv_cache_tensor.page_size_bytes
            tensor = kv_cache_tensor.tensor.view(torch.int8).view((-1, page_size_bytes))
            tensors.append(tensor)

        return tensors
