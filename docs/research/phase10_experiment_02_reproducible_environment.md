# Phase 10 Experiment 2: Reproducible Build and Benchmark Environment

## Abstract

This hardening step replaced implicit workstation knowledge with a versioned
environment record and corrected the README to match the implemented server
state. Compiler, CUDA, Python, JavaScript, hardware, build modes, complete test
matrix, production commands, experimental switches, and performance confounds
are now stated in `docs/ENVIRONMENT.md`.

## Method

Versions were queried from the executing tools rather than inferred from
package filenames. The record includes:

- Ubuntu/WSL kernel and CPU ISA;
- NVIDIA device, driver, VRAM, and compute capability;
- GCC and nvcc versions;
- Python, PyTorch, Transformers, and safetensors versions;
- Node.js and npm versions.

Every documented command was compared with the current Makefile, CLI argument
surface, and web package scripts. Runtime CUDA enablement is separated from
compile-time `CUDA=1`. Diagnostic switches rejected by performance studies are
explicitly labeled experimental rather than appearing as recommended
defaults.

The README status was updated to distinguish:

- completed Gates 1–6;
- active 397B Gate 7;
- completed real-35B web/lifecycle qualification;
- still-open warm continuous batching and 397B Gate 9.

## Results

The reference stack is GCC 13.3.0, CUDA 12.9.86, Python 3.12.3,
Transformers 5.14.1, CPU PyTorch 2.13.0, Node.js 22.22.1, and npm 10.9.4 on a
Ryzen 7 7700X plus RTX 5070 Ti (`sm_120`).

The documented expected suite is 21 C executables, 59 Python tests, 10,000
tokenizer cases, and 18 web tests. Commands now cover CPU, portable
x86-64-v3, CUDA, web, doctor, real web smoke, continuous-batch comparison, and
cancellation latency.

## Interpretation

Reproducibility requires both dependency pins and experimental controls. A
CUDA binary can silently execute CPU fallback unless its runtime switches are
set; a numerically identical model can produce incomparable throughput while
the converter owns storage and page cache. Recording both prevents an
apparently valid command from being misinterpreted as a controlled benchmark.

## Decision

- Retain `docs/ENVIRONMENT.md` as release-controlled documentation.
- Treat changes to expected test counts or production defaults as requiring an
  update to that file and README.
- Keep model conversion and benchmark contention warnings prominent.
- Leave the combined `make check` target open until it executes every
  documented release check rather than only the C subset.
