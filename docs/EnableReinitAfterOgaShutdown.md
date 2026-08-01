# Plan: Clean `OgaShutdown` and re-initialization

`OgaShutdown` returns genai to a *just-loaded* state. It tears down all of
genai's ONNX Runtime-derived global state -- the `OrtEnv`, the device
interfaces, the genai add-on libraries, and the trivial-session allocators --
and a subsequent genai call transparently re-initializes with a fresh `OrtEnv`.
This lets a host that recreates its wrapper in-process (for example Foundry
Local's `Manager`) get a clean environment -- fresh logging and env
configuration -- without restarting the process.

## Background

`OrtEnv` is a process-wide, refcounted singleton. genai holds one reference to
it inside its global state (`OrtGlobals`). Previously that state was created
once via a function-static and never rebuilt, so after `OgaShutdown` reset it
genai could not be used again in the same process. The changes here move all
env-scoped state into a single owner (`OrtGlobals`) that is destroyed on
shutdown and rebuilt on demand.

## `OrtGlobals` owns all env-scoped state

`OrtGlobals` (`src/generators.cpp`, `src/generators.h`) is the single owner of
everything whose lifetime is tied to the `OrtEnv`:

- `env_` -- the `OrtEnv`.
- `owned_interfaces_` -- the in-process `DeviceInterface` instances (CPU,
  WebGPU, QNN, OpenVINO, RyzenAI), owned directly.
- `cuda_library_` -- the genai CUDA add-on library handle (see below).
- `device_interfaces_` -- a non-owning `DeviceType -> DeviceInterface*` lookup
  that indexes the owners above (and the module-owned DML interface).
- `device_allocators_` -- the trivial-session allocators (`{session, allocator}`
  per device type) that bootstrap the non-CPU device allocators.
- `graph_session_cache_` -- cached dynamically-built sessions (Cast, TopK, ...).

Because all of this lives in one object, tearing genai down is a single
`OrtGlobals` destruction and re-initialization is a single `OrtGlobals`
construction.

## Re-creatable globals

`g_ort_globals` is a `std::unique_ptr<OrtGlobals>` that `GetOrtGlobals()` builds
on demand:

- `GetOrtGlobals()` lazily (re)creates the globals whenever they are null, so
  the first call after `OgaShutdown` rebuilds them with a fresh `OrtEnv`.
- Creation is guarded by a mutex rather than `std::call_once` (which is
  process-once and would prevent re-init), so concurrent first-use -- or
  first-use after a shutdown -- cannot double-construct the globals.
- A process-exit guard is constructed only after the first `OrtGlobals` has
  finished construction. Its destructor therefore runs before the ORT statics
  used by `~OrtGlobals`, sets `g_process_exiting`, and prevents
  `GetOrtGlobals()` from resurrecting state during static destruction.
- `Shutdown()` (behind `OgaShutdown`) simply resets `g_ort_globals`;
  `~OrtGlobals` performs all of the teardown.

The `OrtGlobals` constructor bootstraps the CPU interface through a member call
(`this->GetDeviceInterface(DeviceType::CPU)`) rather than the free
`GetDeviceInterface()` -> `GetOrtGlobals()` path, so it does not re-enter
`GetOrtGlobals()` while still being constructed.

## Telemetry re-initialization

`OgaShutdown()` releases the active 1DS log manager before tearing down
`OrtGlobals`. The process-wide `GenAiTelemetry` coordinator remains available
with `initialized_` cleared, so the next model load creates a fresh log manager
and offline cache connection. Process identity and the one-shot `ProcessInfo`
event remain process-scoped rather than being emitted again for every
re-initialization cycle.

Telemetry also installs its process-exit guard only after the first event has
traversed 1DS. This makes implicit teardown run before the SDK statics used by
`FlushAndTeardown`, avoiding static-destruction deadlocks.

The exit guard calls `ShutdownAtProcessExit()` rather than `Shutdown()`. It runs
the same full teardown, but waits only a bounded time to acquire the telemetry
lock. Correct guard ordering means in-flight logging drains normally and the
lock is effectively always free; if a logging thread is ever wedged, the manager
is abandoned so the process still exits instead of hanging. Explicit
`OgaShutdown()` keeps the unbounded wait, because there the caller is still
running and a complete flush is the expected behaviour -- unless process-exit
teardown has already begun, in which case it returns immediately rather than
blocking on a lock the exit guard may never be able to take.

Because the guard is registered late it is destroyed early, so telemetry is no
longer live for objects destroyed later in static destruction. A host that keeps
an `OgaGenerator` in a global will not see its final generation event. Hosts that
need those events should destroy their GenAI objects, or call `OgaShutdown()`,
before returning from `main`.

## Device interface ownership

Each in-process EP exposes a `CreateXInterface()` factory that returns a
`std::unique_ptr<DeviceInterface>`. `OrtGlobals::GetDeviceInterface()` creates
the interface on first use (under a lock), stores it in `owned_interfaces_`, and
caches a raw pointer in `device_interfaces_`. This replaces the previous per-EP
`GetXInterface()` accessors that held the interface in a function/namespace
static (and, for RyzenAI, a `std::call_once`), none of which could be rebuilt
after a shutdown. Because the interfaces are now owned by `OrtGlobals`,
`~OrtGlobals` drops them all, and the hard-coded `RyzenAIInterface::Shutdown()`
that `Shutdown()` used to call is gone.

## CUDA add-on library

The genai CUDA add-on library (`onnxruntime-genai-cuda`) provides its
`DeviceInterface` from inside the DLL (`g_cuda_device`), along with the CUDA
stream and a file-static allocator pointer. `OrtGlobals` owns the loaded library
handle (`cuda_library_`) and holds only a non-owning pointer to the interface in
`device_interfaces_`. Unloading the library at teardown runs the add-on's static
destructors, so the interface, the stream, and the file-static allocator all
cease to exist between cycles -- genai returns to a just-loaded state with no
cached add-on state.

## Teardown order

`~OrtGlobals` destroys state in the reverse of construction order, so everything
that uses env-owned resources is gone before the env itself:

1. `graph_session_cache_` -- genai-side session caches of dynamically created models for pre/post processing.
2. `device_interfaces_` (the non-owning index), then `owned_interfaces_`, then
   `cuda_library_` -- the device interfaces and the CUDA add-on library.
3. `device_allocators_` -- the trivial-session allocators. Each entry bundles its
   own dedicated dummy `OrtSession`; the allocator wraps that bundled session (not
   any session in `graph_session_cache_`), so the relative order of steps 1 and 3
   is not a correctness constraint. Within each entry the `OrtSession` is declared
   before the `Ort::Allocator`, so the allocator is destroyed before its session.
4. `env_` -- the `OrtEnv`. If genai held the last reference, ORT destroys the
   environment here.

Ending all allocator use before the env is destroyed keeps teardown correct even
if a future interface or buffer were to dereference a cached allocator in its own
destructor.

## CPU allocator

The CPU interface allocates through ONNX Runtime's process-global default
allocator (`Ort::Allocator::GetWithDefaultOptions()`), not an env-scoped one, so
nothing genai caches for CPU dangles when the env is destroyed. The only change
CPU needs for re-init is that its `InitOrt` is idempotent: `OrtGlobals` re-runs
`CreateAndRegisterAllocator` on the new env and calls `InitOrt` again with the
same process-global allocator each cycle, so the previous
`assert(!ort_allocator_)` is removed.

## DML

DML is a deliberate exception to the ownership model above, for two reasons:

1. Its interface is built from the model's `luid` / `device_index` provider
   options (parsed in `src/dml/session_options.cpp` and passed to
   `InitDmlInterface`), which the generic `GetDeviceInterface(DeviceType::DML)`
   cannot supply.
2. DML objects launch background threads that must be released promptly, so
   `Model::~Model` tears the DML interface down per-`Model` via
   `CloseDmlInterface()`.

Because DML is destroyed per-`Model`, `OrtGlobals::GetDeviceInterface` must
**not** cache the DML interface -- a cached pointer would dangle after the first
DML model is freed and be handed to a later one. The `#if USE_DML` arm therefore
returns the live `GetDmlInterface()` instead of memoizing it (all other EPs,
which live for the env cycle, are cached). DML is already re-init-safe through
this per-`Model` teardown: after the last DML model is destroyed, no DML
interface or `Dml::` static survives into the next env cycle.

## RyzenAI

RyzenAI is owned by `OrtGlobals` like the other in-process EPs (created via
`CreateRyzenAIInterface(env)`), and the hard-coded `RyzenAIInterface::Shutdown()`
is removed -- teardown now happens when `~OrtGlobals` drops the interface. EP-library
registration is per-`OrtEnv` (ORT keys it by registration name), so on each env cycle
the interface registers the RyzenAI EP library on the fresh env even when the EP module
is already resident in the process from an earlier cycle. Because an `OrtEnv` can outlive
an `OgaShutdown` when the host holds its own reference, the registration call tolerates
ORT's "library is already registered" status for that env rather than skipping
registration up front, so RyzenAI is re-init-safe in both cases.

Teardown of the EP module itself (the `RyzenAI_Shutdown` call in the interface destructor)
is gated on genai having loaded it: each interface records at construction, before it
registers/loads the EP, whether the module was already resident. genai calls
`RyzenAI_Shutdown` only when it was not -- i.e. genai's own registration loaded it. If an
external owner had already loaded it, genai never shuts it down, so it cannot tear down EP
state another owner still relies on. ORT unloads the EP library when the interface's env is
torn down, so the next re-init cycle re-evaluates this against a freshly unloaded module.

## Lifetime contract

No `Model`, `Generator`, `Tokenizer`, `OgaTensor`, `Engine`, `Request`, or any
object that holds device memory may outlive `OgaShutdown`. The caller must
destroy every such object before calling `OgaShutdown`; doing otherwise is
undefined behavior (typically a crash when the buffer is freed through a
now-invalid allocator). Debug builds report any leaked genai objects at shutdown.

## Testing

Re-initialization is covered by a dedicated `reinit_tests` executable, kept
separate from `unit_tests` because it calls `OgaShutdown`, which resets
process-global state and so must not share a process with the public-API suite.
`reinit_tests` links the genai object library (`onnxruntime-genai-obj`) rather
than the shared library, giving white-box access to `GetDeviceInterface` so it
can force each available device interface to be created and torn down directly
across repeated shutdown -> re-init cycles (no model load required). The
object-library split -- the shipped shared library and the white-box tests both
consume `onnxruntime-genai-obj` -- lets the tests reach internal symbols without
exporting them from the shipped DLL.

`implicit_shutdown_tests` exercises the shipped shared library in isolated
processes. One case returns after a real model workload without calling
`OgaShutdown`; the other performs model load -> `OgaShutdown` -> model reload
and then returns implicitly. A third starts model workloads concurrently to
cover first-use guard registration. The re-init case renames the first telemetry
cache after shutdown and requires the second manager to create a fresh cache,
proving that the first manager released its file and the replacement initialized.
Telemetry lifecycle coverage is enabled with
`-DENABLE_TELEMETRY_TEST_HOOKS=ON`; the hook is off by default and uses a dummy
token with no collector URL. It bypasses only the CI and unit-test harness
heuristics; `ORT_DISABLE_TELEMETRY` is a user-facing opt-out that the library
honors unconditionally, so no build configuration can override it.
