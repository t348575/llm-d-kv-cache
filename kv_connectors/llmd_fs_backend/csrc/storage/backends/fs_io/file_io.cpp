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
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <random>

#include "tensor_copier.hpp"
#include "file_io.hpp"
#include "thread_pool.hpp"
#include "logger.hpp"

namespace fs = std::filesystem;

// -------------------------------------------------------------------
// Constants and thread-local buffers
// -------------------------------------------------------------------
// Define a larger buffer (1MB) to reduce syscall overhead and speed up I/O
const size_t WRITE_BUFFER_SIZE = 1 * 1024 * 1024;  // 1MB buffer

// O_DIRECT alignment requirement (512 bytes covers all common block sizes).
const size_t DIRECT_IO_ALIGN = 512;

// Allocate custom I/O buffer for this thread (replaces small default buffer)
thread_local std::vector<char> thread_write_buffer(WRITE_BUFFER_SIZE);

// Thread-local unique suffix for temporary files
thread_local std::string tmp_file_suffix =
    "_" + std::to_string(std::random_device{}()) + ".tmp";

// True when ptr, offset and size all satisfy O_DIRECT's sector alignment.
static inline bool odirect_aligned(const void* ptr, size_t offset, size_t size) {
  return (reinterpret_cast<uintptr_t>(ptr) % DIRECT_IO_ALIGN == 0) &&
         (offset % DIRECT_IO_ALIGN == 0) && (size % DIRECT_IO_ALIGN == 0);
}
// -------------------------------------------------------------------
// file-IO Functions
// -------------------------------------------------------------------
// Partial-write of a back-of-buffer slice via temp file + atomic rename.
bool FileIO::write_buffer_to_file(const StagingBufferInfo& buf,
                                  const std::string& target_path,
                                  size_t write_offset,
                                  size_t write_size,
                                  bool use_odirect) {
  if (!buf.ptr || write_offset + write_size > buf.size) {
    FS_LOG_ERROR("write_buffer_to_file: bad range for "
                 << target_path << " (offset=" << write_offset
                 << " size=" << write_size << " buf.size=" << buf.size << ")");
    return false;
  }
  // Create parent directory if needed
  fs::path file_path(target_path);
  fs::path parent_dir = file_path.parent_path();
  try {
    fs::create_directories(parent_dir);
  } catch (const fs::filesystem_error& e) {
    FS_LOG_ERROR("Failed to create directories: " << e.what());
    return false;
  }

  // Write to a temporary file to ensure atomic replace on rename
  // Include tmp_file_suffix so each thread uses a unique temporary file
  std::string tmp_path = target_path + tmp_file_suffix;

  const char* src = reinterpret_cast<const char*>(buf.ptr) + write_offset;

  // Fast path: O_DIRECT bypasses the page cache. Only taken when the slice is
  // sector-aligned; falls through to buffered I/O otherwise so correctness
  // never depends on the model's block byte size.
  if (use_odirect && odirect_aligned(src, write_offset, write_size)) {
    int fd =
        open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd >= 0) {
      ssize_t written = ::write(fd, src, write_size);
      ::close(fd);
      if (written != static_cast<ssize_t>(write_size)) {
        FS_LOG_ERROR("O_DIRECT write failed: "
                     << tmp_path << " (wrote " << written << "/" << write_size
                     << " bytes) - " << std::strerror(errno));
        std::remove(tmp_path.c_str());
        return false;
      }
      if (std::rename(tmp_path.c_str(), target_path.c_str()) != 0) {
        FS_LOG_ERROR("Failed to rename " << tmp_path << " to " << target_path
                                         << " - " << std::strerror(errno));
        std::remove(tmp_path.c_str());
        return false;
      }
      return true;
    }
    // O_DIRECT not supported by this filesystem — fall back to buffered.
    FS_LOG_WARN("O_DIRECT open failed for " << tmp_path << " ("
                << std::strerror(errno) << "), falling back to buffered write");
  }

  std::ofstream ofs(tmp_path, std::ios::out | std::ios::binary);
  if (!ofs) {
    FS_LOG_ERROR("Failed to open temporary file for writing: "
                 << tmp_path << " - " << std::strerror(errno));
    return false;
  }

  // Apply the custom buffer to the file stream
  ofs.rdbuf()->pubsetbuf(thread_write_buffer.data(), WRITE_BUFFER_SIZE);

  // Write only the actual data region of the staging buffer.
  ofs.write(src, write_size);
  if (!ofs) {
    FS_LOG_ERROR("Failed to write to temporary file: " << tmp_path << " - "
                                                       << std::strerror(errno));
    std::remove(tmp_path.c_str());  // Clean up temp file
    return false;
  }

  ofs.flush();
  if (!ofs) {
    FS_LOG_ERROR("Failed to flush data to temporary file: "
                 << tmp_path << " - " << std::strerror(errno));
    return false;
  }

  // Atomically rename temp file to final target name after a successful write
  if (std::rename(tmp_path.c_str(), target_path.c_str()) != 0) {
    FS_LOG_ERROR("Failed to rename " << tmp_path << " to " << target_path
                                     << " - " << std::strerror(errno));
    std::remove(tmp_path.c_str());
    return false;
  }

  return true;
}

// Partial-read into a back-of-buffer slice; seeks to file tail if needed.
bool FileIO::read_buffer_from_file(const std::string& path,
                                   StagingBufferInfo& buf,
                                   size_t buf_offset,
                                   size_t bytes_per_block,
                                   size_t blocks_in_file,
                                   bool use_odirect) {
  size_t read_size = blocks_in_file * bytes_per_block;
  char* dst = reinterpret_cast<char*>(buf.ptr) + buf_offset;

  // Bounds check destination buffer.
  if (!buf.ptr || buf.size < buf_offset + read_size) {
    FS_LOG_ERROR("Staging buffer too small for file: "
                 << path << " (buf_offset=" << buf_offset
                 << " required=" << read_size << " available=" << buf.size
                 << " ptr=" << buf.ptr << ")");
    return false;
  }

  // Fast path: O_DIRECT pread at the file tail. stat() gives the size without
  // a buffered open. Only taken when file size, offset and slice are all
  // sector-aligned; otherwise falls through to buffered I/O.
  if (use_odirect) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
      FS_LOG_ERROR("O_DIRECT: stat failed: " << path << " - "
                                             << std::strerror(errno));
      return false;
    }
    size_t file_size = static_cast<size_t>(st.st_size);
    if (file_size < read_size) {
      FS_LOG_ERROR("File too small: " << path << " (file_size=" << file_size
                                      << " required=" << read_size << ")");
      return false;
    }
    size_t file_offset = file_size - read_size;
    if (odirect_aligned(dst, buf_offset, read_size) &&
        (file_offset % DIRECT_IO_ALIGN == 0) &&
        (file_size % DIRECT_IO_ALIGN == 0)) {
      int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
      if (fd >= 0) {
        ssize_t bytes_read =
            pread(fd, dst, read_size, static_cast<off_t>(file_offset));
        ::close(fd);
        if (bytes_read != static_cast<ssize_t>(read_size)) {
          FS_LOG_ERROR("O_DIRECT read failed: "
                       << path << " (read " << bytes_read << "/" << read_size
                       << " bytes from offset " << file_offset << ") - "
                       << std::strerror(errno));
          return false;
        }
        return true;
      }
      // O_DIRECT not supported — fall back to buffered.
      FS_LOG_WARN("O_DIRECT open failed for " << path << " ("
                  << std::strerror(errno) << "), falling back to buffered read");
    }
  }

  // Open file and grab its size in one pass (ios::ate).
  std::ifstream ifs(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!ifs) {
    FS_LOG_ERROR("Failed to open file: " << path);
    return false;
  }
  std::ifstream::pos_type end_pos = ifs.tellg();
  if (end_pos == std::streampos(-1)) {
    FS_LOG_ERROR("Failed to determine file size: " << path);
    return false;
  }
  size_t file_size = static_cast<size_t>(end_pos);

  // File must hold at least the blocks the caller is asking for.
  if (file_size < read_size) {
    FS_LOG_ERROR("File too small: " << path << " (file_size=" << file_size
                                    << " required=" << read_size << ")");
    return false;
  }

  size_t file_offset = file_size - read_size;
  ifs.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);

  // Read file into Staging buffer
  ifs.read(dst, static_cast<std::streamsize>(read_size));
  std::streamsize bytes_read = ifs.gcount();
  if (bytes_read != static_cast<std::streamsize>(read_size) || !ifs.good()) {
    FS_LOG_ERROR("Failed to read file: "
                 << path << " (read " << bytes_read << "/" << read_size
                 << " bytes from offset " << file_offset << ")");
    return false;
  }

  return true;
}

// update_atime update only the atime of a file without changing mtime
void FileIO::update_atime(const std::string& path) {
  struct timespec times[2];
  times[0].tv_nsec = UTIME_NOW;   // atime → now
  times[1].tv_nsec = UTIME_OMIT;  // mtime → unchanged
  utimensat(AT_FDCWD, path.c_str(), times, 0);
}

// Write via CPU staging - wraps copy_blocks + write_buffer_to_file
bool FileIO::write_blocks_to_file(const std::string& dst_file,
                                  const std::vector<int64_t>& block_ids,
                                  int group_idx,
                                  int head_offset,
                                  cudaStream_t stream) {
  // Get thread-local staging buffer
  StagingBufferInfo& buf = ThreadPool::get_staging_buffer();
  auto* cpu_base = static_cast<uint8_t*>(buf.ptr);
  bool is_store = true;

  // Stage 1: copy tensors from GPU to staging CPU buffer at slot head_offset
  TIME_EXPR("write phase 1: copy_blocks ",
            m_tensor_copier.copy_blocks(cpu_base,
                                        block_ids,
                                        group_idx,
                                        head_offset,
                                        is_store),
            "file: ",
            dst_file);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    FS_LOG_ERROR("write_blocks_to_file: cudaStreamSynchronize failed: "
                 << cudaGetErrorString(err));
    return false;
  }

  // Stage 2: persist only the populated slice, starting at the head_offset
  // slot. Read path mirrors this offset to recover the position.
  size_t bytes_per_block = m_tensor_copier.bytes_per_block_for_group(group_idx);
  size_t blocks_in_file = block_ids.size();
  size_t write_offset = static_cast<size_t>(head_offset) * bytes_per_block;
  size_t write_size = blocks_in_file * bytes_per_block;
  bool success = TIME_EXPR(
      "write phase 2: write_buffer_to_file",
      write_buffer_to_file(buf, dst_file, write_offset, write_size, m_use_odirect),
      "file:",
      dst_file,
      " size:",
      write_size);

  if (!success) {
    FS_LOG_ERROR(
        "write_blocks_to_file: Store failed during file write: " << dst_file);
  }

  return success;
}

// Read via CPU staging - wraps read_buffer_from_file + copy_blocks
bool FileIO::read_blocks_from_file(const std::string& src_file,
                                   const std::vector<int64_t>& block_ids,
                                   int group_idx,
                                   int head_offset,
                                   cudaStream_t stream) {
  // Get thread-local staging buffer
  StagingBufferInfo& buf = ThreadPool::get_staging_buffer();

  // Stage 1: read blocks into the staging buffer at the head_offset slot.
  // read_buffer_from_file seeks to the file tail for suffix-of-full-file,
  // or reads from offset 0 for partial-write files.
  size_t bytes_per_block = m_tensor_copier.bytes_per_block_for_group(group_idx);
  size_t blocks_in_file = block_ids.size();
  size_t buf_offset = static_cast<size_t>(head_offset) * bytes_per_block;
  bool success = TIME_EXPR("read phase 1: read_buffer_from_file",
                           read_buffer_from_file(src_file,
                                                 buf,
                                                 buf_offset,
                                                 bytes_per_block,
                                                 blocks_in_file,
                                                 m_use_odirect),
                           "file:",
                           src_file);
  if (!success) {
    FS_LOG_ERROR("read_blocks_from_file: read_buffer_from_file failed for "
                 << src_file);
    return false;
  }

  // Stage 2: copy tensors from staging CPU buffer to GPU
  auto* cpu_base = static_cast<uint8_t*>(buf.ptr);
  bool is_store = false;

  success = TIME_EXPR("read phase 2: copy_cpu_tensor_to_gpu_tensors",
                      m_tensor_copier.copy_blocks(cpu_base,
                                                  block_ids,
                                                  group_idx,
                                                  head_offset,
                                                  is_store),
                      "file: ",
                      src_file);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    FS_LOG_ERROR("read_blocks_from_file: cudaStreamSynchronize failed: "
                 << cudaGetErrorString(err));
    return false;
  }

  return success;
}
