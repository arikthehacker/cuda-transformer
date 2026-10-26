CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wpedantic
BUILD    = build
SRC      = tokenizer/bpe.cpp
HDR      = tokenizer/bpe.h

# make DEBUG=1 builds with address + undefined behavior checks (like valgrind)
ifeq ($(DEBUG),1)
  CXXFLAGS = -std=c++17 -g -O0 -Wall -Wextra -fsanitize=address,undefined
endif

all: $(BUILD)/bpe $(BUILD)/test_bpe

$(BUILD)/bpe: tokenizer/main.cpp $(SRC) $(HDR) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ tokenizer/main.cpp $(SRC)

$(BUILD)/test_bpe: tests/test_bpe.cpp $(SRC) $(HDR) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Itokenizer -o $@ tests/test_bpe.cpp $(SRC)

$(BUILD):
	mkdir -p $(BUILD)

# train on the corpus, then encode train and val into token files
run: $(BUILD)/bpe
	./$(BUILD)/bpe -t -i data/corpus_train.txt -n 4096 -o data/tokenizer.bpe -v
	./$(BUILD)/bpe -e -m data/tokenizer.bpe -i data/corpus_train.txt -o data/train.bin
	./$(BUILD)/bpe -e -m data/tokenizer.bpe -i data/corpus_val.txt -o data/val.bin

test: $(BUILD)/test_bpe
	./$(BUILD)/test_bpe

clean:
	rm -rf $(BUILD)

.PHONY: all run test clean


# ==================================================================== stage 3: CUDA

NVCC     = nvcc
NVFLAGS  = -O3 -std=c++17 -arch=sm_75 -lineinfo -Xcompiler -Wall
# NEVER add --use_fast_math in stage 3: it swaps expf, tanhf and division for
# faster, less accurate versions and the dump will stop matching.
ifeq ($(DEBUG),1)
  NVFLAGS = -std=c++17 -arch=sm_75 -g -G -DCUDA_DEBUG -Xcompiler -Wall
endif
CU_SRC   = cuda/common.cu cuda/dump.cu cuda/gpt.cu cuda/weights.cu cuda/data.cu $(wildcard cuda/kernels/*.cu)
CU_HDR   = $(wildcard cuda/*.h cuda/kernels/*.h cuda/kernels/*.cuh)
DUMP     = data/ref/debug.bin
DUMP_FULL = $(wildcard data/ref/debug_full.bin)

cuda: $(BUILD)/test_kernels $(BUILD)/test_gpt $(BUILD)/train_gpt

$(BUILD)/hello: cuda/tests/hello.cu cuda/common.h | $(BUILD)
	$(NVCC) $(NVFLAGS) -Icuda -o $@ cuda/tests/hello.cu

$(BUILD)/test_kernels: cuda/tests/test_kernels.cu $(CU_SRC) $(CU_HDR) | $(BUILD)
	$(NVCC) $(NVFLAGS) -Icuda -o $@ cuda/tests/test_kernels.cu $(CU_SRC)

$(BUILD)/test_gpt: cuda/tests/test_gpt.cu $(CU_SRC) $(CU_HDR) | $(BUILD)
	$(NVCC) $(NVFLAGS) -Icuda -o $@ cuda/tests/test_gpt.cu $(CU_SRC)

$(BUILD)/train_gpt: cuda/train.cu $(CU_SRC) $(CU_HDR) | $(BUILD)
	$(NVCC) $(NVFLAGS) -Icuda -o $@ cuda/train.cu $(CU_SRC)

# step 1 of the setup: a one line kernel, copied back and printed
cuda-hello: $(BUILD)/hello
	./$(BUILD)/hello

test-cuda: $(BUILD)/test_kernels $(BUILD)/test_gpt
	./$(BUILD)/test_kernels $(DUMP)
	./$(BUILD)/test_gpt $(DUMP) $(DUMP_FULL)

# catches out of bounds reads/writes and race conditions on the GPU
sanitize-cuda: $(BUILD)/test_kernels
	compute-sanitizer --tool memcheck ./$(BUILD)/test_kernels $(DUMP)
	compute-sanitizer --tool racecheck ./$(BUILD)/test_kernels $(DUMP)

train-cuda: $(BUILD)/train_gpt
	./$(BUILD)/train_gpt

.PHONY: cuda cuda-hello test-cuda sanitize-cuda train-cuda


# ==================================================================== stage 4: fast CUDA

# Every stage 4 program comes in two builds of the same code:
#   build/x     -DFP16: floatX = half, tensor cores, flash attention
#   build/x32   floatX = float: the correctness anchor, matches the dump at
#               stage 3 tolerances
# Each source file compiles once per build into build/o16 or build/o32.
NVFLAGS16 = $(NVFLAGS) -DFP16
FAST_SRC  = $(CU_SRC) cuda/gpt_fast.cu cuda/graph.cu
GEN_SRC   = $(FAST_SRC) cuda/generator.cu tokenizer/bpe.cpp
GEN_HDR   = $(CU_HDR) tokenizer/bpe.h
OBJ16     = $(patsubst %,$(BUILD)/o16/%.o,$(basename $(FAST_SRC)))
OBJ32     = $(patsubst %,$(BUILD)/o32/%.o,$(basename $(FAST_SRC)))
GEN16     = $(OBJ16) $(BUILD)/o16/cuda/generator.o $(BUILD)/o16/tokenizer/bpe.o
GEN32     = $(OBJ32) $(BUILD)/o32/cuda/generator.o $(BUILD)/o32/tokenizer/bpe.o

$(BUILD)/o16/%.o: %.cu $(CU_HDR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS16) -Icuda -Itokenizer -c $< -o $@

$(BUILD)/o32/%.o: %.cu $(CU_HDR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -Icuda -Itokenizer -c $< -o $@

$(BUILD)/o16/%.o: %.cpp $(GEN_HDR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/o32/%.o: %.cpp $(GEN_HDR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

FAST_BIN  = $(BUILD)/test_gemm $(BUILD)/test_gemm32 $(BUILD)/test_attn $(BUILD)/test_fast $(BUILD)/test_fast32 \
            $(BUILD)/train_fast $(BUILD)/train_fast32
GEN_BIN   = $(BUILD)/generate $(BUILD)/generate32 $(BUILD)/test_generate $(BUILD)/test_generate32

fast: $(FAST_BIN) $(GEN_BIN)

$(BUILD)/test_gemm: $(BUILD)/o16/cuda/tests/test_gemm.o $(OBJ16)
	$(NVCC) $(NVFLAGS16) -o $@ $^

$(BUILD)/test_gemm32: $(BUILD)/o32/cuda/tests/test_gemm.o $(OBJ32)
	$(NVCC) $(NVFLAGS) -o $@ $^

$(BUILD)/test_attn: $(BUILD)/o16/cuda/tests/test_attn.o $(OBJ16)
	$(NVCC) $(NVFLAGS16) -o $@ $^

$(BUILD)/test_fast: $(BUILD)/o16/cuda/tests/test_fast.o $(OBJ16)
	$(NVCC) $(NVFLAGS16) -o $@ $^

$(BUILD)/test_fast32: $(BUILD)/o32/cuda/tests/test_fast.o $(OBJ32)
	$(NVCC) $(NVFLAGS) -o $@ $^

$(BUILD)/train_fast: $(BUILD)/o16/cuda/train_fast.o $(OBJ16)
	$(NVCC) $(NVFLAGS16) -o $@ $^

$(BUILD)/train_fast32: $(BUILD)/o32/cuda/train_fast.o $(OBJ32)
	$(NVCC) $(NVFLAGS) -o $@ $^

$(BUILD)/bench_gemm: $(BUILD)/o16/bench/bench_gemm.o $(OBJ16)
	$(NVCC) $(NVFLAGS16) -o $@ $^ -lcublas

# the fp32 build first: it is the one that says the code is right
test-fast: $(FAST_BIN) $(BUILD)/test_generate $(BUILD)/test_generate32 test-cuda
	./$(BUILD)/test_gemm32
	./$(BUILD)/test_fast32 $(DUMP)
	./$(BUILD)/test_generate32 $(DUMP)
	./$(BUILD)/test_gemm
	./$(BUILD)/test_attn $(DUMP) $(DUMP_FULL)
	./$(BUILD)/test_fast $(DUMP) $(DUMP_FULL)
	./$(BUILD)/test_generate $(DUMP)

sanitize-fast: $(BUILD)/test_fast $(BUILD)/test_attn
	compute-sanitizer --tool memcheck ./$(BUILD)/test_attn $(DUMP)
	compute-sanitizer --tool racecheck ./$(BUILD)/test_attn $(DUMP)
	compute-sanitizer --tool memcheck ./$(BUILD)/test_fast $(DUMP)

train-fast: $(BUILD)/train_fast
	./$(BUILD)/train_fast

# registers and spills per kernel (want 0 bytes spill), then HMMA / LDSM counts in the real machine code
ptxas: | $(BUILD)
	@for f in gemm_mma gemm_wmma flash_attn decode; do \
		echo "== $$f"; \
		$(NVCC) $(NVFLAGS16) -Icuda -Xptxas -v -c cuda/kernels/$$f.cu -o $(BUILD)/ptxas_$$f.o 2>&1 | \
			grep -E "Compiling entry|registers|spill" | sed -E 's/ptxas info *: //'; \
	done

sass: $(BUILD)/bench_gemm
	cuobjdump -sass $(BUILD)/bench_gemm | awk '/Function :/ { f = $$3 } /HMMA/ { h[f]++ } /LDSM/ { l[f]++ } \
		END { for (k in h) printf "%5d HMMA %5d LDSM  %s\n", h[k], l[k], k }' | sort -k5

bench-gemm: $(BUILD)/bench_gemm
	./$(BUILD)/bench_gemm -o bench/results_gemm.md

# shared memory bank conflicts of the tensor core kernels (want 0); WSL needs GPU counters
# allowed for all users in the NVIDIA Control Panel (Developer -> Manage GPU Performance Counters)
bank-conflicts: $(BUILD)/bench_gemm
	ncu --kernel-name regex:"gemm_mma|gemm_wmma" --launch-count 6 \
		--metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
		./$(BUILD)/bench_gemm -r 1 -o /dev/null

bench-train: $(BUILD)/train_gpt $(BUILD)/train_fast $(BUILD)/train_fast32
	bash bench/bench_train.sh

# step 0 of stage 4: where the time goes, per kernel
profile: $(BUILD)/train_gpt $(BUILD)/train_fast
	mkdir -p bench/logs
	nsys profile --force-overwrite true -o bench/logs/stage3 --stats=true ./$(BUILD)/train_gpt -n 50 -e 50 \
		> bench/logs/profile_stage3.txt
	nsys profile --force-overwrite true -o bench/logs/stage4 --stats=true ./$(BUILD)/train_fast -n 50 -e 50 -S "" \
		> bench/logs/profile_stage4.txt
	grep -A25 "cuda_gpu_kern_sum" bench/logs/profile_stage4.txt | head -40

curves:
	python3 bench/curves.py

results:
	python3 bench/results.py --readme README.md

.PHONY: fast test-fast sanitize-fast train-fast ptxas sass bench-gemm bank-conflicts bench-train profile curves results


# ==================================================================== stage 5: generation

$(BUILD)/generate: $(BUILD)/o16/cuda/generate.o $(GEN16)
	$(NVCC) $(NVFLAGS16) -o $@ $^

$(BUILD)/generate32: $(BUILD)/o32/cuda/generate.o $(GEN32)
	$(NVCC) $(NVFLAGS) -o $@ $^

$(BUILD)/test_generate: $(BUILD)/o16/cuda/tests/test_generate.o $(GEN16)
	$(NVCC) $(NVFLAGS16) -o $@ $^

$(BUILD)/test_generate32: $(BUILD)/o32/cuda/tests/test_generate.o $(GEN32)
	$(NVCC) $(NVFLAGS) -o $@ $^

WEIGHTS = data/ref/weights.bin

sample: $(BUILD)/generate
	./$(BUILD)/generate -w $(WEIGHTS) -t bible,austen -s 1

bench-generate: $(BUILD)/generate
	./$(BUILD)/generate -w $(WEIGHTS) -b -n 500 -o bench/results_generate.md

.PHONY: sample bench-generate


# ==================================================================== stage 5 stretch: the browser demo

# needs clang and lld (sudo apt-get install -y clang lld), and node for the test
WASMFLAGS = --target=wasm32 -O3 -msimd128 -mbulk-memory -nostdlib -fno-exceptions -fno-rtti -std=c++17 \
            -Wall -Wextra -Wl,--no-entry

demo/forward.wasm: demo/forward.cpp
	clang++ $(WASMFLAGS) -o $@ demo/forward.cpp

# ONLY a model trained on the public sources goes into demo/weights16.bin
demo: demo/forward.wasm
	cp data/tokenizer.bpe demo/tokenizer.bpe
	@test -f demo/weights16.bin || echo "next: python.exe -m reference.export --fp16 --out demo/weights16.bin"

test-demo: demo/forward.wasm
	python3 -m reference.export --from-dump $(DUMP) --out data/ref/weights_debug.bin
	node demo/test_forward.mjs demo/forward.wasm data/ref/weights_debug.bin $(DUMP)

serve-demo: demo
	python3 -m http.server 8000 -d demo

.PHONY: demo test-demo serve-demo
