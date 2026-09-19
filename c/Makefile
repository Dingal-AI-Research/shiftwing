# Shiftwing build — adapted from colibri's c/Makefile (github.com/JustVugg/colibri,
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
LIBEXECDIR ?= $(PREFIX)/libexec/shiftwing
INSTALL    ?= install

# CUDA=1: opt-in backend (Phase 5). Linux direct-link only for now.
CUDA      ?= 0
CUDA_HOME ?= $(if $(wildcard ../.toolchains/cuda-12.9/bin/nvcc),$(abspath ../.toolchains/cuda-12.9),/usr/local/cuda)
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra
CUDA_OBJ   =
GLM_CUDA_OBJ =
GLM_CUDA_FLAGS =
GLM_CUDA_LIBS =
CUBLAS_HOME ?= $(if $(wildcard ../.toolchains/cublas/nvidia/cublas/include/cublas_v2.h),$(abspath ../.toolchains/cublas/nvidia/cublas),$(CUDA_HOME))
CUBLAS_LIBDIR ?= $(if $(wildcard $(CUBLAS_HOME)/lib/libcublas.so*),$(CUBLAS_HOME)/lib,$(CUBLAS_HOME)/lib64)
ifeq ($(CUDA),1)
ifneq (,$(DARWIN))
$(error CUDA=1 is supported only on Linux)
endif
CFLAGS   += -DCOLI_CUDA
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
CUDA_OBJ  = backend_cuda.o
GLM_CUDA_OBJ = glm53_cuda.o
GLM_CUDA_FLAGS = -DGLM53_CUDA_BACKEND
GLM_CUDA_LIBS = -L$(CUBLAS_LIBDIR) -Wl,-rpath,$(CUBLAS_LIBDIR) -l:libcublas.so.12
endif

CHECK_CUDA ?= $(if $(wildcard $(NVCC)),1,0)
ifeq ($(CHECK_CUDA),1)
CHECK_CUDA_RUN = $(MAKE) CUDA=1 CUDA_ARCH=$(CUDA_ARCH) qwen test-cuda
else
CHECK_CUDA_RUN = @echo "CUDA release checks skipped (set CHECK_CUDA=1 with NVCC/CUDA_HOME to require them)"
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
            tests/test_st_batch$(EXE) \
            tests/test_tier$(EXE) tests/test_grammar$(EXE) tests/test_schema_gbnf$(EXE) \
            tests/test_serve_mux$(EXE) tests/test_serve_scheduler$(EXE) tests/test_qwen_session$(EXE) \
            tests/test_decode_batch$(EXE) tests/test_compat_direct$(EXE) \
            tests/test_deltanet$(EXE) tests/test_gqa_rope$(EXE) tests/test_router$(EXE) \
            tests/test_quant_matmul$(EXE) tests/test_gdn_chunk$(EXE) tests/test_gdn_slots$(EXE) \
            tests/test_attn_slots$(EXE) tests/test_attn_prefill$(EXE) \
            tests/test_hybrid_slots$(EXE) tests/test_hyper_connection$(EXE) \
            tests/test_ple_ngram$(EXE) tests/test_qsa_indexer$(EXE) \
            tests/test_qwen4_exp_config$(EXE) tests/test_ple_layer$(EXE) \
            tests/test_resident_qwen4_exp$(EXE)
TEST_BINS += tests/test_resident_model$(EXE) tests/test_prefill_pipeline$(EXE)
TEST_BINS += tests/test_tok_digit_run$(EXE)

all: qwen$(EXE) glm53$(EXE)

ifneq ($(EXE),)
qwen: qwen$(EXE)
glm53: glm53$(EXE)
endif

# Config stamp: make tracks file timestamps, not flag changes; record the
# build-affecting flags so a flag change forces a relink (colibri #306).
BUILD_CONFIG     := $(CC)|$(CFLAGS)|$(LDFLAGS)|CUDA=$(CUDA)|ARCH=$(ARCH)|CUDA_ARCH=$(CUDA_ARCH)|CUBLAS=$(CUBLAS_HOME)
BUILD_CONFIG_OLD := $(shell cat .build-config 2>/dev/null)
ifneq "$(BUILD_CONFIG)" "$(BUILD_CONFIG_OLD)"
$(shell printf '%s' '$(BUILD_CONFIG)' > .build-config)
endif
.build-config: ;

qwen$(EXE): qwen.c st.h uring.h json.h tok.h tok_unicode.h tok_nfc.h compat.h grammar.h tier.h decode_batch.h serve_mux.h serve_scheduler.h $(CUDA_OBJ) .build-config
	$(CC) $(CFLAGS) qwen.c $(CUDA_OBJ) -o qwen$(EXE) $(LDFLAGS)

# GLM-5.3-Flash engine. Separate binary, not linked into qwen: the two engines
# have their own Cfg/Layer/forward and only share the leaf headers. Keeping them
# apart means GLM work cannot destabilise the lane that is serving reviews.
GLM53_HEADERS = st.h json.h tok.h tok_unicode_o200k.h compat.h quant.h cli_args.h glm53_compat.h hyper_connections.h delta_attention.h sparse_index.h vision_tower.h glm53_gpu.inc glm53_cuda.h

glm53$(EXE): glm53.c $(GLM53_HEADERS) $(GLM_CUDA_OBJ) .build-config
	$(CC) $(CFLAGS) $(GLM_CUDA_FLAGS) glm53.c $(GLM_CUDA_OBJ) -o glm53$(EXE) $(LDFLAGS) $(GLM_CUDA_LIBS)

glm53_cuda.o: glm53_cuda.cu glm53_cuda.h .build-config
	"$(NVCC)" $(NVCCFLAGS) -I$(CUBLAS_HOME)/include -c $< -o $@

tests/test_glm53_cuda$(EXE): tests/test_glm53_cuda.c glm53_cuda.o glm53_cuda.h
	$(CC) $(CFLAGS) $< glm53_cuda.o -o $@ $(LDFLAGS) $(GLM_CUDA_LIBS)

test-glm53-cuda: tests/test_glm53_cuda$(EXE)
	./tests/test_glm53_cuda$(EXE)

backend_cuda.o: backend_cuda.cu backend_cuda.h .build-config
	@command -v "$(NVCC)" >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	"$(NVCC)" $(NVCCFLAGS) -c backend_cuda.cu -o $@

tests/test_backend_cuda$(EXE): tests/test_backend_cuda.c backend_cuda.o backend_cuda.h
	$(CC) $(CFLAGS) $< backend_cuda.o -o $@ $(LDFLAGS)

tests/test_qwen_session_cuda$(EXE): tests/test_qwen_session.c qwen.c serve_mux.h serve_scheduler.h backend_cuda.o backend_cuda.h
	$(CC) $(CFLAGS) $< backend_cuda.o -o $@ $(LDFLAGS)

test-cuda: tests/test_backend_cuda$(EXE) tests/test_qwen_session_cuda$(EXE) tests/test_attn_prefill_cuda$(EXE)
	./tests/test_backend_cuda$(EXE)
	SESSION_CUDA=1 ./tests/test_qwen_session_cuda$(EXE)
	./tests/test_attn_prefill_cuda$(EXE)

iobench$(EXE): iobench.c compat.h
	$(CC) $(CFLAGS) iobench.c -o iobench$(EXE) $(LDFLAGS)

tests/test_json$(EXE): tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st$(EXE): tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st_pread$(EXE): tests/test_st_pread.c st.h json.h compat.h
	$(CC) $(CFLAGS) -DST_PREAD_CHUNK=7 $< -o $@ $(LDFLAGS)

tests/test_st_batch$(EXE): tests/test_st_batch.c st.h uring.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tier$(EXE): tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_grammar$(EXE): tests/test_grammar.c grammar.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_schema_gbnf$(EXE): tests/test_schema_gbnf.c schema_gbnf.h grammar.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_serve_mux$(EXE): tests/test_serve_mux.c serve_mux.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_serve_scheduler$(EXE): tests/test_serve_scheduler.c serve_scheduler.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_qwen_session$(EXE): tests/test_qwen_session.c qwen.c serve_mux.h serve_scheduler.h
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

tests/test_hyper_connection$(EXE): tests/test_hyper_connection.c tests/fixture_util.h qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_ple_ngram$(EXE): tests/test_ple_ngram.c tests/fixture_util.h qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_qsa_indexer$(EXE): tests/test_qsa_indexer.c tests/fixture_util.h qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_qwen4_exp_config$(EXE): tests/test_qwen4_exp_config.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_ple_layer$(EXE): tests/test_ple_layer.c tests/fixture_util.h qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_resident_qwen4_exp$(EXE): tests/test_resident_qwen4_exp.c qwen.c serve_mux.h serve_scheduler.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_quant_matmul$(EXE): tests/test_quant_matmul.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_gdn_chunk$(EXE): tests/test_gdn_chunk.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_gdn_slots$(EXE): tests/test_gdn_slots.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_attn_slots$(EXE): tests/test_attn_slots.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_attn_prefill$(EXE): tests/test_attn_prefill.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_attn_prefill_cuda$(EXE): tests/test_attn_prefill.c qwen.c backend_cuda.o backend_cuda.h
	$(CC) $(CFLAGS) $< backend_cuda.o -o $@ $(LDFLAGS)

tests/test_hybrid_slots$(EXE): tests/test_hybrid_slots.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_resident_model$(EXE): tests/test_resident_model.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_prefill_pipeline$(EXE): tests/test_prefill_pipeline.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/bench_qmat$(EXE): tests/bench_qmat.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/bench_layer_stack$(EXE): tests/bench_layer_stack.c qwen.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/bench_tier_io$(EXE): tests/bench_tier_io.c st.h uring.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/bench_gqa_prefill$(EXE): tests/bench_gqa_prefill.c backend_cuda.o backend_cuda.h
	$(CC) $(CFLAGS) $< backend_cuda.o -o $@ $(LDFLAGS)

bench-qmat: tests/bench_qmat$(EXE)
	./tests/bench_qmat$(EXE)

bench-stack: tests/bench_layer_stack$(EXE)
	./tests/bench_layer_stack$(EXE)

# Needs CUDA=1: measures the full-attention prefill path at Ornith397 shapes.
bench-gqa-prefill: tests/bench_gqa_prefill$(EXE)
	./tests/bench_gqa_prefill$(EXE) $(T)

bench-tier-io: tests/bench_tier_io$(EXE)
	@test -n "$(SNAP)" || { echo "set SNAP=<converted snapshot>" >&2; exit 1; }
	./tests/bench_tier_io$(EXE) "$(SNAP)"

# Weight-free DeepSeek exact-count helper; deliberately requires no CUDA object.
deepseek_v4_tokenize$(EXE): deepseek_v4_tokenize.c deepseek_v4_tokenizer.h deepseek_v4_unicode.h tok.h tok_unicode.h tok_nfc.h json.h
	$(CC) $(filter-out -DCOLI_CUDA,$(CFLAGS)) $< -o $@ -lm

# tokenizer parity harness (Phase 1 gate): ./tests/test_tok tokenizer.json < cases
tests/test_tok$(EXE): tests/test_tok.c tok.h tok_unicode.h tok_nfc.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tok_digit_run$(EXE): tests/test_tok_digit_run.c tok.h tok_unicode.h tok_nfc.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	$(PYTHON) tools/run_tests.py $(TEST_BINS)

test-python: qwen$(EXE)
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test-web: qwen$(EXE) web
	cd ../web && npm test
	$(PYTHON) -m unittest tests.test_cli_qwen.ShiftwingCliTests.test_web_serves_bundle_and_real_tiny_chat

test-tokenizer: tests/test_tok$(EXE)
	$(PYTHON) tests/test_tok_qwen.py --binary ./tests/test_tok$(EXE)

# GLM-5.3 against the HuggingFace tiny oracle. The fixture is generated from
# transformers' own Glm5NextTextModel, so this checks the engine against an
# independent implementation rather than against itself. f32 proves the port;
# int8/int4 measure what quantisation costs, and the tokens must stay exact at
# every precision. Skips cleanly when the fixture has not been generated.
GLM53_FIXTURE ?= glm53_tiny

$(GLM53_FIXTURE)/ref.json:
	$(PYTHON) tools/make_glm53_tiny.py --output $(GLM53_FIXTURE)

# The converted fixture: quantised experts, so it streams and exercises the
# expert cache. The f32 fixture keeps its experts resident and never reaches
# that code at all.
GLM53_CONVERTED ?= glm53_tiny_i4

$(GLM53_CONVERTED)/model.safetensors: $(GLM53_FIXTURE)/ref.json
	$(PYTHON) tools/convert_glm53.py --indir $(GLM53_FIXTURE) \
	    --outdir $(GLM53_CONVERTED) --keep-source --min-free-c-gb 1
	cp $(GLM53_FIXTURE)/config.json $(GLM53_FIXTURE)/tokenizer.json \
	   $(GLM53_FIXTURE)/ref.json $(GLM53_CONVERTED)/

test-glm53: glm53$(EXE) $(GLM53_FIXTURE)/ref.json $(GLM53_CONVERTED)/model.safetensors
	$(PYTHON) tools/test_glm53_tiny.py --binary ./glm53$(EXE) --fixture $(GLM53_FIXTURE) --bits 32
	$(PYTHON) tools/test_glm53_tiny.py --binary ./glm53$(EXE) --fixture $(GLM53_FIXTURE) --bits 8
	$(PYTHON) tools/test_glm53_tiny.py --binary ./glm53$(EXE) --fixture $(GLM53_FIXTURE) --bits 4
	$(PYTHON) tools/test_glm53_tiny.py --binary ./glm53$(EXE) --fixture $(GLM53_CONVERTED) --bits 32 --logit-tolerance 0.5
	$(PYTHON) tools/test_glm53_cache.py --binary ./glm53$(EXE) --fixture $(GLM53_CONVERTED)

test-docs:
	$(PYTHON) tools/check_docs.py --root ..

test-source-package:
	$(PYTHON) tools/check_source_package.py --root ..

test: test-c

check:
	$(MAKE) clean
	$(MAKE) qwen$(EXE) ARCH=x86-64-v3
	$(MAKE) test-c ARCH=x86-64-v3
	$(MAKE) test-tokenizer ARCH=x86-64-v3
	cd ../web && npm ci
	cd ../web && npm test
	cd ../web && npm run build
	cd ../web && npm audit
	$(MAKE) test-python ARCH=x86-64-v3
	$(MAKE) test-docs
	$(MAKE) test-source-package
	$(CHECK_CUDA_RUN)

clean:
	rm -rf $(GLM53_CONVERTED)
	rm -f qwen$(EXE) glm53$(EXE) iobench$(EXE) backend_cuda.o glm53_cuda.o tests/test_glm53_cuda$(EXE) tests/test_glm53_prefill$(EXE) tests/bench_glm53_chunks$(EXE) .build-config $(TEST_BINS) tests/test_tok$(EXE) tests/test_backend_cuda$(EXE) tests/test_qwen_session_cuda$(EXE) tests/test_attn_prefill_cuda$(EXE) tests/bench_qmat$(EXE) tests/bench_layer_stack$(EXE) tests/bench_gqa_prefill$(EXE)

web:
	cd ../web && npm ci && npm run build

install: qwen$(EXE) web
	$(INSTALL) -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(LIBEXECDIR) $(DESTDIR)$(LIBEXECDIR)/tools $(DESTDIR)$(LIBEXECDIR)/web/dist
	$(INSTALL) -m 755 shiftwing $(DESTDIR)$(BINDIR)/shiftwing
	$(INSTALL) -m 755 colib $(DESTDIR)$(BINDIR)/colib
	$(INSTALL) -m 755 qwen$(EXE) $(DESTDIR)$(LIBEXECDIR)/qwen$(EXE)
	$(INSTALL) -m 644 openai_server.py doctor.py $(DESTDIR)$(LIBEXECDIR)/
	$(INSTALL) -m 644 tools/*.py $(DESTDIR)$(LIBEXECDIR)/tools/
	cp -R ../web/dist/. $(DESTDIR)$(LIBEXECDIR)/web/dist/

.PHONY: all qwen glm53 web test-glm53 test-c test-python test-web test-tokenizer test-docs test-source-package test-cuda test check bench-qmat bench-stack bench-gqa-prefill clean install

.PHONY: test-glm53-cuda

# Opt-in GLM chunk-size checks and loaded-model benchmark.
tests/test_glm53_prefill$(EXE): tests/test_glm53_prefill.c glm53.c $(GLM53_HEADERS) $(GLM_CUDA_OBJ) .build-config
	$(CC) $(CFLAGS) $(GLM_CUDA_FLAGS) $< $(GLM_CUDA_OBJ) -o $@ $(LDFLAGS) $(GLM_CUDA_LIBS)

tests/bench_glm53_chunks$(EXE): tests/bench_glm53_chunks.c glm53.c $(GLM53_HEADERS) $(GLM_CUDA_OBJ) .build-config
	$(CC) $(CFLAGS) $(GLM_CUDA_FLAGS) $< $(GLM_CUDA_OBJ) -o $@ $(LDFLAGS) $(GLM_CUDA_LIBS)

test-glm53-prefill: tests/test_glm53_prefill$(EXE) $(GLM53_CONVERTED)/model.safetensors
	./tests/test_glm53_prefill$(EXE) $(GLM53_CONVERTED)

.PHONY: test-glm53-prefill

# Native DeepSeek remains opt-in until reviewer qualification passes.
DSV4_HEADERS = $(wildcard deepseek_v4*.h) st.h uring.h json.h backend_cuda.h
deepseek_v4$(EXE): deepseek_v4.c $(DSV4_HEADERS) $(CUDA_OBJ) .build-config
	$(CC) $(CFLAGS) $< $(CUDA_OBJ) -o $@ $(LDFLAGS)
backend_cuda.o: deepseek_v4_cuda.inc
DSV4_PROBES = tests/probe_deepseek_v4_expert tests/probe_deepseek_v4_prefill tests/probe_deepseek_v4_dense
$(DSV4_PROBES): %: %.c $(DSV4_HEADERS) $(CUDA_OBJ) .build-config
	$(CC) $(CFLAGS) $< $(CUDA_OBJ) -o $@ $(LDFLAGS)
DSV4_CONTRACT = tests/test_deepseek_v4_contract
DSV4_TESTS = $(filter-out $(DSV4_CONTRACT),$(patsubst %.c,%,$(wildcard tests/test_deepseek_v4*.c)))
$(DSV4_TESTS) $(DSV4_CONTRACT): %: %.c $(DSV4_HEADERS) $(CUDA_OBJ) .build-config
	$(CC) $(CFLAGS) $< $(CUDA_OBJ) -o $@ $(LDFLAGS)
test-deepseek: $(DSV4_TESTS)
	$(PYTHON) tools/run_tests.py $(DSV4_TESTS)
.PHONY: test-deepseek

# This check needs a complete converted checkpoint, unlike the fixture suite.
test-deepseek-contract: $(DSV4_CONTRACT)
	@test -n "$(DSV4_SNAPSHOT)" || (echo "Set DSV4_SNAPSHOT to a complete converted checkpoint" >&2; exit 2)
	./$(DSV4_CONTRACT) "$(DSV4_SNAPSHOT)"
.PHONY: test-deepseek-contract
