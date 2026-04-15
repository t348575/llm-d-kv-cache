/*
 * Copyright 2025 The llm-d Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <sys/stat.h>
#include <random>
#include <atomic>
#include <mutex>

#include "file_io.hpp"
#include "thread_pool.hpp"
#include "logger.hpp"

namespace fs = std::filesystem;

namespace {

struct FileOpStats {
  std::atomic<uint64_t> file_writes{0};
  std::atomic<uint64_t> file_reads{0};
  std::atomic<uint64_t> parent_dir_ensures{0};
  std::atomic<uint64_t> file_renames{0};
  std::atomic<uint64_t> exists_skips{0};
  std::atomic<uint64_t> atime_updates{0};
  std::atomic<uint64_t> bytes_written{0};
  std::atomic<uint64_t> bytes_read{0};
  std::atomic<int64_t> next_emit_ns{0};
  int64_t last_emit_ns{0};
  uint64_t last_file_writes{0};
  uint64_t last_file_reads{0};
  uint64_t last_parent_dir_ensures{0};
  uint64_t last_file_renames{0};
  uint64_t last_exists_skips{0};
  uint64_t last_atime_updates{0};
  uint64_t last_bytes_written{0};
  uint64_t last_bytes_read{0};
  std::mutex emit_mutex;

  void maybe_emit() {
    static const int64_t interval_ns = []() -> int64_t {
      const char* env = std::getenv("LLMD_FS_STATS_INTERVAL_SEC");
      if (!env) return 5LL * 1000000000LL;
      try {
        double seconds = std::stod(env);
        if (seconds <= 0.0) return 0;
        return static_cast<int64_t>(seconds * 1000000000.0);
      } catch (...) {
        return 5LL * 1000000000LL;
      }
    }();

    if (interval_ns <= 0) return;

    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
    int64_t current_next = next_emit_ns.load(std::memory_order_relaxed);
    if (current_next == 0) {
      next_emit_ns.store(now_ns + interval_ns, std::memory_order_relaxed);
      last_emit_ns = now_ns;
      return;
    }
    if (now_ns < current_next) return;

    std::lock_guard<std::mutex> lock(emit_mutex);
    current_next = next_emit_ns.load(std::memory_order_relaxed);
    if (now_ns < current_next) return;
    next_emit_ns.store(now_ns + interval_ns, std::memory_order_relaxed);

    const double elapsed_sec =
        std::max(static_cast<double>(now_ns - last_emit_ns) / 1e9, 1e-9);
    last_emit_ns = now_ns;

    const uint64_t file_writes_now = file_writes.load(std::memory_order_relaxed);
    const uint64_t file_reads_now = file_reads.load(std::memory_order_relaxed);
    const uint64_t parent_dir_ensures_now =
        parent_dir_ensures.load(std::memory_order_relaxed);
    const uint64_t file_renames_now = file_renames.load(std::memory_order_relaxed);
    const uint64_t exists_skips_now = exists_skips.load(std::memory_order_relaxed);
    const uint64_t atime_updates_now = atime_updates.load(std::memory_order_relaxed);
    const uint64_t bytes_written_now = bytes_written.load(std::memory_order_relaxed);
    const uint64_t bytes_read_now = bytes_read.load(std::memory_order_relaxed);

    FS_LOG_INFO(
        "llmd_fs_file_stats"
        << " file_writes=" << file_writes_now
        << " file_reads=" << file_reads_now
        << " parent_dir_ensures=" << parent_dir_ensures_now
        << " file_renames=" << file_renames_now
        << " exists_skips=" << exists_skips_now
        << " atime_updates=" << atime_updates_now
        << " bytes_written=" << bytes_written_now
        << " bytes_read=" << bytes_read_now
        << " file_writes_s=" << (file_writes_now - last_file_writes) / elapsed_sec
        << " file_reads_s=" << (file_reads_now - last_file_reads) / elapsed_sec
        << " parent_dir_ensures_s="
        << (parent_dir_ensures_now - last_parent_dir_ensures) / elapsed_sec
        << " file_renames_s=" << (file_renames_now - last_file_renames) / elapsed_sec
        << " exists_skips_s=" << (exists_skips_now - last_exists_skips) / elapsed_sec
        << " atime_updates_s=" << (atime_updates_now - last_atime_updates) / elapsed_sec
        << " bytes_written_s=" << (bytes_written_now - last_bytes_written) / elapsed_sec
        << " bytes_read_s=" << (bytes_read_now - last_bytes_read) / elapsed_sec);

    last_file_writes = file_writes_now;
    last_file_reads = file_reads_now;
    last_parent_dir_ensures = parent_dir_ensures_now;
    last_file_renames = file_renames_now;
    last_exists_skips = exists_skips_now;
    last_atime_updates = atime_updates_now;
    last_bytes_written = bytes_written_now;
    last_bytes_read = bytes_read_now;
  }
};

FileOpStats& file_op_stats() {
  static FileOpStats stats;
  return stats;
}

}  // namespace

// -------------------------------------------------------------------
// Constants and thread-local buffers
// -------------------------------------------------------------------
// Define a larger buffer (1MB) to reduce syscall overhead and speed up I/O
const size_t WRITE_BUFFER_SIZE = 1 * 1024 * 1024;  // 1MB buffer

// O_DIRECT alignment requirement (512 bytes covers all common block sizes)
const size_t DIRECT_IO_ALIGN = 512;

// Allocate custom I/O buffer for this thread (replaces small default buffer)
thread_local std::vector<char> thread_write_buffer(WRITE_BUFFER_SIZE);

// Thread-local unique suffix for temporary files
thread_local std::string tmp_file_suffix =
    "_" + std::to_string(std::random_device{}()) + ".tmp";

static inline size_t align_up(size_t n, size_t align) {
  return (n + align - 1) & ~(align - 1);
}

static bool ensure_parent_dir(const std::string& path) {
  fs::path parent_dir = fs::path(path).parent_path();
  try {
    fs::create_directories(parent_dir);
    record_parent_dir_ensure();
    return true;
  } catch (const fs::filesystem_error& e) {
    FS_LOG_ERROR("Failed to create directories: " << e.what());
    return false;
  }
}

// -------------------------------------------------------------------
// file-IO Functions
// -------------------------------------------------------------------
// Write a buffer to disk using a temporary file and atomic rename
bool write_buffer_to_file(const StagingBufferInfo& buf,
                          const std::string& target_path,
                          bool use_odirect) {
  if (!ensure_parent_dir(target_path)) return false;

  std::string tmp_path = target_path + tmp_file_suffix;

  if (use_odirect) {
    // O_DIRECT requires pointer and size to be aligned to DIRECT_IO_ALIGN
    size_t write_size = align_up(buf.size, DIRECT_IO_ALIGN);
    if (write_size > buf.size + DIRECT_IO_ALIGN) {
      // Alignment added more than one block – something is very wrong.
      FS_LOG_ERROR("O_DIRECT: buf.size " << buf.size << " requires unexpected padding");
      return false;
    }
    if (reinterpret_cast<uintptr_t>(buf.ptr) % DIRECT_IO_ALIGN != 0) {
      FS_LOG_ERROR("O_DIRECT: buf.ptr is not aligned to " << DIRECT_IO_ALIGN);
      return false;
    }

    int fd = open(tmp_path.c_str(),
                  O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT,
                  0644);
    if (fd < 0) {
      FS_LOG_ERROR("O_DIRECT open for write failed: " << tmp_path
                   << " - " << std::strerror(errno));
      return false;
    }

    ssize_t written = write(fd, buf.ptr, write_size);
    close(fd);

    if (written != static_cast<ssize_t>(write_size)) {
      FS_LOG_ERROR("O_DIRECT write failed: " << tmp_path
                   << " (wrote " << written << "/" << write_size << " bytes)"
                   << " - " << std::strerror(errno));
      std::remove(tmp_path.c_str());
      return false;
    }
  } else {
    std::ofstream ofs(tmp_path, std::ios::out | std::ios::binary);
    if (!ofs) {
      FS_LOG_ERROR("Failed to open temporary file for writing: "
                   << tmp_path << " - " << std::strerror(errno));
      return false;
    }

    ofs.rdbuf()->pubsetbuf(thread_write_buffer.data(), WRITE_BUFFER_SIZE);
    ofs.write(reinterpret_cast<const char*>(buf.ptr), buf.size);
    if (!ofs) {
      FS_LOG_ERROR("Failed to write to temporary file: " << tmp_path
                   << " - " << std::strerror(errno));
      std::remove(tmp_path.c_str());
      return false;
    }

    ofs.flush();
    if (!ofs) {
      FS_LOG_ERROR("Failed to flush data to temporary file: "
                   << tmp_path << " - " << std::strerror(errno));
      return false;
    }
  }

  record_file_write(buf.size);

  if (std::rename(tmp_path.c_str(), target_path.c_str()) != 0) {
    FS_LOG_ERROR("Failed to rename " << tmp_path << " to " << target_path
                 << " - " << std::strerror(errno));
    std::remove(tmp_path.c_str());
    return false;
  }

  record_file_rename();

  return true;
}

// Read a file into a thread-local staging buffer
bool read_buffer_from_file(const std::string& path, StagingBufferInfo& buf, bool use_odirect) {
  if (use_odirect) {
    // Determine file size via stat (avoids opening with buffered IO first)
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
      FS_LOG_ERROR("O_DIRECT: stat failed: " << path << " - " << std::strerror(errno));
      return false;
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    // O_DIRECT requires pointer and read size to be aligned to DIRECT_IO_ALIGN
    if (reinterpret_cast<uintptr_t>(buf.ptr) % DIRECT_IO_ALIGN != 0) {
      FS_LOG_ERROR("O_DIRECT: buf.ptr is not aligned to " << DIRECT_IO_ALIGN);
      return false;
    }
    size_t read_size = align_up(file_size, DIRECT_IO_ALIGN);
    if (!buf.ptr || buf.size < read_size) {
      FS_LOG_ERROR("Staging buffer too small for O_DIRECT read: "
                   << path << " (required=" << read_size
                   << " available=" << buf.size << " ptr=" << buf.ptr << ")");
      return false;
    }

    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
      FS_LOG_ERROR("O_DIRECT open for read failed: " << path
                   << " - " << std::strerror(errno));
      return false;
    }

    ssize_t bytes_read = read(fd, buf.ptr, read_size);
    close(fd);

    if (bytes_read < static_cast<ssize_t>(file_size)) {
      FS_LOG_ERROR("O_DIRECT read failed: " << path
                   << " (read " << bytes_read << "/" << file_size << " bytes)"
                   << " - " << std::strerror(errno));
      return false;
    }

    // Only file_size bytes are valid; caller must not rely on padding bytes
    buf.size = file_size;
    record_file_read(file_size);
    return true;
  }

  // Open files
  std::ifstream ifs(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!ifs) {
    FS_LOG_ERROR("Failed to open file: " << path);
    return false;
  }

  // Determine file size
  std::ifstream::pos_type end_pos = ifs.tellg();
  if (end_pos == std::streampos(-1)) {
    FS_LOG_ERROR("Failed to determine file size: " << path);
    return false;
  }
  size_t file_size = static_cast<size_t>(end_pos);
  ifs.seekg(0, std::ios::beg);  // Move read pointer to start for reading

  // Acquire staging buffer of the required size
  if (!buf.ptr || buf.size < file_size) {
    FS_LOG_ERROR("Staging buffer too small for file: "
                 << path << " (required=" << file_size
                 << " available=" << buf.size << " ptr=" << buf.ptr << ")");
    return false;
  }

  // Read file into Staging buffer
  ifs.read(reinterpret_cast<char*>(buf.ptr),
           static_cast<std::streamsize>(file_size));
  std::streamsize bytes_read = ifs.gcount();
  if (bytes_read != static_cast<std::streamsize>(file_size) || !ifs.good()) {
    FS_LOG_ERROR("Failed to read full file: " << path << " (read " << bytes_read
                                              << "/" << file_size << " bytes)");
    return false;
  }

  record_file_read(file_size);

  return true;
}

// update_atime update only the atime of a file without changing mtime
void update_atime(const std::string& path) {
  struct timespec times[2];
  times[1].tv_nsec = UTIME_NOW;   // update atime to now
  times[0].tv_nsec = UTIME_OMIT;  // keep mtime unchanged
  utimensat(AT_FDCWD, path.c_str(), times, 0);
  record_atime_update();
}

void record_file_write(size_t num_bytes) {
  auto& stats = file_op_stats();
  stats.file_writes.fetch_add(1, std::memory_order_relaxed);
  stats.bytes_written.fetch_add(num_bytes, std::memory_order_relaxed);
  stats.maybe_emit();
}

void record_file_read(size_t num_bytes) {
  auto& stats = file_op_stats();
  stats.file_reads.fetch_add(1, std::memory_order_relaxed);
  stats.bytes_read.fetch_add(num_bytes, std::memory_order_relaxed);
  stats.maybe_emit();
}

void record_parent_dir_ensure() {
  auto& stats = file_op_stats();
  stats.parent_dir_ensures.fetch_add(1, std::memory_order_relaxed);
  stats.maybe_emit();
}

void record_file_rename() {
  auto& stats = file_op_stats();
  stats.file_renames.fetch_add(1, std::memory_order_relaxed);
  stats.maybe_emit();
}

void record_exists_skip() {
  auto& stats = file_op_stats();
  stats.exists_skips.fetch_add(1, std::memory_order_relaxed);
  stats.maybe_emit();
}

void record_atime_update() {
  auto& stats = file_op_stats();
  stats.atime_updates.fetch_add(1, std::memory_order_relaxed);
  stats.maybe_emit();
}
