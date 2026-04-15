import os
import threading
import time

from vllm.logger import init_logger


logger = init_logger(__name__)


class ConnectorStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._interval_sec = float(os.getenv("LLMD_FS_STATS_INTERVAL_SEC", "5"))
        now_ns = time.monotonic_ns()
        self._next_emit_ns = now_ns + int(self._interval_sec * 1e9)
        self._last_emit_ns = now_ns

        self._lookup_txn = 0
        self._lookup_keys = 0
        self._lookup_hits = 0
        self._lookup_misses = 0
        self._lookup_latency_ns = 0

        self._prepare_store_txn = 0
        self._prepare_store_keys = 0

        self._publish_submit_txn = 0
        self._publish_submit_files = 0
        self._publish_submit_fail = 0
        self._publish_submit_latency_ns = 0

        self._load_submit_txn = 0
        self._load_submit_files = 0
        self._load_submit_fail = 0
        self._load_submit_latency_ns = 0

        self._last_lookup_txn = 0
        self._last_lookup_keys = 0
        self._last_lookup_hits = 0
        self._last_lookup_misses = 0
        self._last_prepare_store_txn = 0
        self._last_prepare_store_keys = 0
        self._last_publish_submit_txn = 0
        self._last_publish_submit_files = 0
        self._last_publish_submit_fail = 0
        self._last_load_submit_txn = 0
        self._last_load_submit_files = 0
        self._last_load_submit_fail = 0

    def record_lookup(
        self,
        checked_keys: int,
        hit_count: int,
        miss_count: int,
        latency_ns: int,
    ) -> None:
        with self._lock:
            self._lookup_txn += 1
            self._lookup_keys += checked_keys
            self._lookup_hits += hit_count
            self._lookup_misses += miss_count
            self._lookup_latency_ns += latency_ns
            self._maybe_emit_locked()

    def record_prepare_store(self, key_count: int) -> None:
        with self._lock:
            self._prepare_store_txn += 1
            self._prepare_store_keys += key_count
            self._maybe_emit_locked()

    def record_publish_submit(
        self, file_count: int, latency_ns: int, success: bool
    ) -> None:
        with self._lock:
            self._publish_submit_txn += 1
            self._publish_submit_files += file_count
            self._publish_submit_latency_ns += latency_ns
            if not success:
                self._publish_submit_fail += 1
            self._maybe_emit_locked()

    def record_load_submit(
        self, file_count: int, latency_ns: int, success: bool
    ) -> None:
        with self._lock:
            self._load_submit_txn += 1
            self._load_submit_files += file_count
            self._load_submit_latency_ns += latency_ns
            if not success:
                self._load_submit_fail += 1
            self._maybe_emit_locked()

    def _maybe_emit_locked(self) -> None:
        if self._interval_sec <= 0:
            return
        now_ns = time.monotonic_ns()
        if now_ns < self._next_emit_ns:
            return
        self._next_emit_ns = now_ns + int(self._interval_sec * 1e9)
        elapsed_sec = max((now_ns - self._last_emit_ns) / 1e9, 1e-9)
        self._last_emit_ns = now_ns

        lookup_txn_rate = (self._lookup_txn - self._last_lookup_txn) / elapsed_sec
        lookup_key_rate = (self._lookup_keys - self._last_lookup_keys) / elapsed_sec
        lookup_hit_rate = (self._lookup_hits - self._last_lookup_hits) / elapsed_sec
        lookup_miss_rate = (
            self._lookup_misses - self._last_lookup_misses
        ) / elapsed_sec
        prepare_store_txn_rate = (
            self._prepare_store_txn - self._last_prepare_store_txn
        ) / elapsed_sec
        prepare_store_key_rate = (
            self._prepare_store_keys - self._last_prepare_store_keys
        ) / elapsed_sec
        publish_submit_txn_rate = (
            self._publish_submit_txn - self._last_publish_submit_txn
        ) / elapsed_sec
        publish_submit_file_rate = (
            self._publish_submit_files - self._last_publish_submit_files
        ) / elapsed_sec
        publish_submit_fail_rate = (
            self._publish_submit_fail - self._last_publish_submit_fail
        ) / elapsed_sec
        load_submit_txn_rate = (
            self._load_submit_txn - self._last_load_submit_txn
        ) / elapsed_sec
        load_submit_file_rate = (
            self._load_submit_files - self._last_load_submit_files
        ) / elapsed_sec
        load_submit_fail_rate = (
            self._load_submit_fail - self._last_load_submit_fail
        ) / elapsed_sec

        self._last_lookup_txn = self._lookup_txn
        self._last_lookup_keys = self._lookup_keys
        self._last_lookup_hits = self._lookup_hits
        self._last_lookup_misses = self._lookup_misses
        self._last_prepare_store_txn = self._prepare_store_txn
        self._last_prepare_store_keys = self._prepare_store_keys
        self._last_publish_submit_txn = self._publish_submit_txn
        self._last_publish_submit_files = self._publish_submit_files
        self._last_publish_submit_fail = self._publish_submit_fail
        self._last_load_submit_txn = self._load_submit_txn
        self._last_load_submit_files = self._load_submit_files
        self._last_load_submit_fail = self._load_submit_fail

        lookup_avg_us = (
            self._lookup_latency_ns / self._lookup_txn / 1e3
            if self._lookup_txn
            else 0.0
        )
        publish_avg_us = (
            self._publish_submit_latency_ns / self._publish_submit_txn / 1e3
            if self._publish_submit_txn
            else 0.0
        )
        load_avg_us = (
            self._load_submit_latency_ns / self._load_submit_txn / 1e3
            if self._load_submit_txn
            else 0.0
        )

        logger.info(
            "llmd_fs_connector_stats "
            "lookup_txn=%d lookup_keys=%d lookup_hits=%d lookup_misses=%d lookup_avg_us=%.1f "
            "lookup_txn_s=%.1f lookup_keys_s=%.1f lookup_hits_s=%.1f lookup_misses_s=%.1f "
            "prepare_store_txn=%d prepare_store_keys=%d prepare_store_txn_s=%.1f prepare_store_keys_s=%.1f "
            "publish_submit_txn=%d publish_submit_files=%d publish_submit_fail=%d publish_submit_avg_us=%.1f "
            "publish_submit_txn_s=%.1f publish_submit_files_s=%.1f publish_submit_fail_s=%.1f "
            "load_submit_txn=%d load_submit_files=%d load_submit_fail=%d load_submit_avg_us=%.1f "
            "load_submit_txn_s=%.1f load_submit_files_s=%.1f load_submit_fail_s=%.1f",
            self._lookup_txn,
            self._lookup_keys,
            self._lookup_hits,
            self._lookup_misses,
            lookup_avg_us,
            lookup_txn_rate,
            lookup_key_rate,
            lookup_hit_rate,
            lookup_miss_rate,
            self._prepare_store_txn,
            self._prepare_store_keys,
            prepare_store_txn_rate,
            prepare_store_key_rate,
            self._publish_submit_txn,
            self._publish_submit_files,
            self._publish_submit_fail,
            publish_avg_us,
            publish_submit_txn_rate,
            publish_submit_file_rate,
            publish_submit_fail_rate,
            self._load_submit_txn,
            self._load_submit_files,
            self._load_submit_fail,
            load_avg_us,
            load_submit_txn_rate,
            load_submit_file_rate,
            load_submit_fail_rate,
        )


connector_stats = ConnectorStats()
