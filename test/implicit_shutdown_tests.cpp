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

void UnsetEnv(const char* name) {
#ifdef _WIN32
  if (_putenv_s(name, "") != 0) {
    throw std::runtime_error(std::string{"Failed to clear "} + name);
  }
#else
  if (unsetenv(name) != 0) {
    throw std::runtime_error(std::string{"Failed to clear "} + name);
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

  // ORT_RUNNING_UNIT_TESTS keeps ONNX Runtime's own telemetry suppressed. The GenAI test hook
  // bypasses only that harness signal, so the 1DS lifecycle under test still runs, using a dummy
  // token with no collector URL.
  SetEnv("ORT_RUNNING_UNIT_TESTS", "1");
  // ORT_DISABLE_TELEMETRY is a user opt-out that the library honors unconditionally, so an
  // ambient setting would suppress the very lifecycle these tests cover. Clear it for this test
  // process only: the cache below is redirected into a scratch directory that the test wipes, and
  // the hook's empty collector URL means nothing leaves the machine.
  UnsetEnv("ORT_DISABLE_TELEMETRY");
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

void RequireNoTelemetryCache(const std::filesystem::path& storage_dir) {
  if (std::filesystem::exists(storage_dir / "onnxruntime-genai.db")) {
    throw std::runtime_error("ORT_DISABLE_TELEMETRY was set but telemetry still created its cache");
  }
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
    std::cerr << "Usage: implicit_shutdown_tests <implicit|reinit|concurrent|optout> <state_dir>" << std::endl;
    return 2;
  }

  try {
    const std::filesystem::path state_dir = argv[2];
    ConfigureIsolatedTelemetry(state_dir);
    const std::filesystem::path telemetry_storage_dir = TelemetryStorageDir(state_dir);
    const std::string mode = argv[1];

    if (mode == "optout") {
      // Regression guard for the user-facing opt-out. This binary already runs with the internal
      // test-mode variable set, so it reproduces the one combination where a bypass could hide:
      // a build compiled with ENABLE_TELEMETRY_TEST_HOOKS=ON, test mode active, and the user
      // opting out. Telemetry must stay fully suppressed and leave no cache behind.
      SetEnv("ORT_DISABLE_TELEMETRY", "1");
      RunCpuWorkload();
      RequireNoTelemetryCache(telemetry_storage_dir);
      return 0;
    }

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
