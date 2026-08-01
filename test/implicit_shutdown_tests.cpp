// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ort_genai.h"

namespace {

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
  if (_putenv_s(name, value) != 0) {
    throw std::runtime_error(std::string{"Failed to set "} + name);
  }
#else
  if (setenv(name, value, 1) != 0) {
    throw std::runtime_error(std::string{"Failed to set "} + name);
  }
#endif
}

void ConfigureIsolatedTelemetry(const std::filesystem::path& state_dir) {
  std::error_code ec;
  std::filesystem::remove_all(state_dir, ec);
  ec.clear();
  std::filesystem::create_directories(state_dir, ec);
  if (ec) {
    throw std::runtime_error("Failed to create telemetry test state directory: " + ec.message());
  }

  // Keep ORT telemetry suppressed while the test-only GenAI hook runs 1DS with no collector URL.
  SetEnv("ORT_RUNNING_UNIT_TESTS", "1");
  SetEnv("ORT_DISABLE_TELEMETRY", "1");
  SetEnv("ORTGENAI_INTERNAL_TELEMETRY_TEST_MODE", "1");
#ifdef _WIN32
  SetEnv("LOCALAPPDATA", state_dir.string().c_str());
#elif defined(__APPLE__)
  SetEnv("HOME", state_dir.string().c_str());
#else
  SetEnv("XDG_CACHE_HOME", state_dir.string().c_str());
#endif
}

std::filesystem::path TelemetryStorageDir(const std::filesystem::path& state_dir) {
#ifdef _WIN32
  return state_dir / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#elif defined(__APPLE__)
  return state_dir / "Library" / "Application Support" / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#else
  return state_dir / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#endif
}

void RequireTelemetryCache(const std::filesystem::path& storage_dir) {
#if ORTGENAI_TEST_EXPECT_TELEMETRY
  if (!std::filesystem::is_regular_file(storage_dir / "onnxruntime-genai.db")) {
    throw std::runtime_error("Telemetry did not create its isolated offline cache");
  }
#else
  (void)storage_dir;
#endif
}

void RunCpuWorkload() {
  const std::vector<int32_t> input_ids{0, 0, 0, 52};
  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOption("batch_size", 1);
  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }
  if (generator->GetSequenceCount(0) == 0) {
    throw std::runtime_error("Generation produced no tokens");
  }
}

void RunConcurrentCpuWorkloads() {
  constexpr int kThreadCount = 4;
  std::atomic<bool> start{false};
  std::exception_ptr failure;
  std::mutex failure_mutex;
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (int i = 0; i < kThreadCount; ++i) {
    workers.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
      }
      try {
        RunCpuWorkload();
      } catch (...) {
        std::scoped_lock lock{failure_mutex};
        if (!failure) {
          failure = std::current_exception();
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& worker : workers) {
    worker.join();
  }
  if (failure) {
    std::rethrow_exception(failure);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: implicit_shutdown_tests <implicit|reinit|concurrent> <state_dir>" << std::endl;
    return 2;
  }

  try {
    const std::filesystem::path state_dir = argv[2];
    ConfigureIsolatedTelemetry(state_dir);
    const std::filesystem::path telemetry_storage_dir = TelemetryStorageDir(state_dir);
    const std::string mode = argv[1];

    if (mode == "concurrent") {
      RunConcurrentCpuWorkloads();
    } else {
      RunCpuWorkload();
    }
    RequireTelemetryCache(telemetry_storage_dir);
    if (mode == "reinit") {
      OgaShutdown();

#if ORTGENAI_TEST_EXPECT_TELEMETRY
      std::filesystem::path first_cycle_storage_dir = telemetry_storage_dir;
      first_cycle_storage_dir += "-first-cycle";
      std::error_code ec;
      std::filesystem::remove_all(first_cycle_storage_dir, ec);
      ec.clear();
      std::filesystem::rename(telemetry_storage_dir, first_cycle_storage_dir, ec);
      if (ec) {
        throw std::runtime_error("Telemetry did not release its first-cycle cache: " + ec.message());
      }
#endif

      RunCpuWorkload();
      RequireTelemetryCache(telemetry_storage_dir);
    } else if (mode != "implicit" && mode != "concurrent") {
      std::cerr << "Unknown mode: " << argv[1] << std::endl;
      return 2;
    }

    // Intentionally omit OgaShutdown(): process-exit guards must tear down telemetry and ORT.
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
