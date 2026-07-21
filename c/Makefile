# colib build — adapted from colibri's c/Makefile (github.com/JustVugg/colibri,
# Apache-2.0; see NOTICE). Same structure: target detection via -dumpmachine,
# .build-config stamp so flag changes force a relink, opt-in CUDA.

DETECT_CC := $(if $(filter default,$(origin CC)),gcc,$(CC))
TRIPLET   := $(shell $(DETECT_CC) -dumpmachine 2>/dev/null)

MINGW  := $(findstring mingw,$(TRIPLET))
CYGWIN := $(findstring cygwin,$(TRIPLET))
DARWIN := $(findstring darwin,$(TRIPLET))
LINUX  := $(findstring linux,$(TRIPLET))
IS_WIN := $(MINGW)$(CYGWIN)

ifneq (,$(DARWIN))
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(and $(OMPDIR),$(wildcard $(OMPDIR)/include/omp.h),$(wildcard $(OMPDIR)/lib/libomp.*)),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp not found: building single-threaded. For multithreading: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
ifneq ($(ARCH),)
CFLAGS += -mcpu=$(ARCH)
endif
LDFLAGS = -lm $(OMPL)
EXE     =
else ifneq ($(IS_WIN),)
CC      = gcc
ARCH   ?= x86-64-v3
CFLAGS  = -D_FILE_OFFSET_BITS=64 -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -static -lpsapi
EXE     = .exe
else
# Linux / *BSD x86-64 — primary target (WSL2 Zen4: ARCH=native unlocks
# the AVX-512/VNNI int8 and grouped-int4 kernels).
CC      = gcc
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -pthread
EXE     =
endif

PREFIX     ?= /usr/local
BINDIR     ?= $(PREFIX)/bin
LIBEXECDIR ?= $(PREFIX)/libexec/colib
INSTALL    ?= install

# CUDA=1: opt-in backend (Phase 5). Linux direct-link only for now.
CUDA      ?= 0
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra
CUDA_OBJ   =
ifeq ($(CUDA),1)
ifneq (,$(DARWIN))
$(error CUDA=1 is supported only on Linux)
endif
CFLAGS   += -DCOLI_CUDA
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
CUDA_OBJ  = backend_cuda.o
endif

ifeq ($(OS),Windows_NT)
PYTHON ?= python
else ifneq (,$(wildcard ../.venv/bin/python))
PYTHON ?= ../.venv/bin/python
else
PYTHON ?= python3
endif

# Vendored header-only tests (green from Phase 0). qwen.c-internal tests
# (quant kernels, sampling, deltanet, rope, router) join in Phases 2–3.
TEST_BINS = tests/test_json$(EXE) tests/test_st$(EXE) tests/test_st_pread$(EXE) \
            tests/test_tier$(EXE) tests/test_grammar$(EXE) tests/test_schema_gbnf$(EXE) \
            tests/test_decode_batch$(EXE) tests/test_compat_direct$(EXE) \
            tests/test_deltanet$(EXE) tests/test_gqa_rope$(EXE) tests/test_router$(EXE) \
            tests/test_quant_matmul$(EXE) tests/test_gdn_chunk$(EXE)

all: qwen$(EXE)

ifneq ($(EXE),)
qwen: qwen$(EXE)
endif

# Config stamp: make tracks file timestamps, not flag changes; record the
# build-affecting flags so a flag change forces a relink (colibri #306).
BUILD_CONFIG     := $(CC)|$(CFLAGS)|$(LDFLAGS)|CUDA=$(CUDA)|ARCH=$(ARCH)|CUDA_ARCH=$(CUDA_ARCH)
BUILD_CONFIG_OLD := $(shell cat .build-config 2>/dev/null)
ifneq "$(BUILD_CONFIG)" "$(BUILD_CONFIG_OLD)"
$(shell printf '%s' '$(BUILD_CONFIG)' > .build-config)
endif
.build-config: ;

qwen$(EXE): qwen.c st.h uring.h json.h tok.h tok_unicode.h tok_nfc.h compat.h grammar.h tier.h decode_batch.h $(CUDA_OBJ) .build-config
	$(CC) $(CFLAGS) qwen.c $(CUDA_OBJ) -o qwen$(EXE) $(LDFLAGS)

backend_cuda.o: backend_cuda.cu backend_cuda.h .build-config
	@command -v "$(NVCC)" >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	"$(NVCC)" $(NVCCFLAGS) -c backend_cuda.cu -o $@

iobench$(EXE): iobench.c compat.h
	$(CC) $(CFLAGS) iobench.c -o iobench$(EXE) $(LDFLAGS)

tests/test_json$(EXE): tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st$(EXE): tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st_pread$(EXE): tests/test_st_pread.c st.h json.h compat.h
	$(CC) $(CFLAGS) -DST_PREAD_CHUNK=7 $< -o $@ $(LDFLAGS)

tests/test_tier$(EXE): tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_grammar$(EXE): tests/test_grammar.c grammar.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_schema_gbnf$(EXE): tests/test_schema_gbnf.c schema_gbnf.h grammar.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_decode_batch$(EXE): tests/test_decode_batch.c decode_batch.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_compat_direct$(EXE): tests/test_compat_direct.c compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_deltanet$(EXE): tests/test_deltanet.c tests/fixture_util.h qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_gqa_rope$(EXE): tests/test_gqa_rope.c tests/fixture_util.h qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_router$(EXE): tests/test_router.c tests/fixture_util.h qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_quant_matmul$(EXE): tests/test_quant_matmul.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_gdn_chunk$(EXE): tests/test_gdn_chunk.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/bench_qmat$(EXE): tests/bench_qmat.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/bench_layer_stack$(EXE): tests/bench_layer_stack.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

bench-qmat: tests/bench_qmat$(EXE)
	./tests/bench_qmat$(EXE)

bench-stack: tests/bench_layer_stack$(EXE)
	./tests/bench_layer_stack$(EXE)

# tokenizer parity harness (Phase 1 gate): ./tests/test_tok tokenizer.json < cases
tests/test_tok$(EXE): tests/test_tok.c tok.h tok_unicode.h tok_nfc.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	$(PYTHON) tools/run_tests.py $(TEST_BINS)

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test-tokenizer: tests/test_tok$(EXE)
	$(PYTHON) tests/test_tok_qwen.py --binary ./tests/test_tok$(EXE)

test: test-c

check:
	$(MAKE) clean
	$(MAKE) qwen$(EXE) ARCH=x86-64-v3
	$(MAKE) test

clean:
	rm -f qwen$(EXE) iobench$(EXE) backend_cuda.o .build-config $(TEST_BINS) tests/test_tok$(EXE) tests/bench_qmat$(EXE) tests/bench_layer_stack$(EXE)

install: qwen$(EXE)
	$(INSTALL) -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(LIBEXECDIR) $(DESTDIR)$(LIBEXECDIR)/tools
	$(INSTALL) -m 755 qwen$(EXE) $(DESTDIR)$(LIBEXECDIR)/qwen$(EXE)
	$(INSTALL) -m 644 tools/*.py $(DESTDIR)$(LIBEXECDIR)/tools/

.PHONY: all qwen test-c test-python test-tokenizer test check bench-qmat bench-stack clean install
