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

# the tokenizer's checks, the pretokenizer v2 ones included
test-tok: test

clean:
	rm -rf $(BUILD)

.PHONY: all run test test-tok clean


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
FAST_SRC  = $(CU_SRC) cuda/gpt_fast.cu cuda/graph.cu cuda/checkpoint.cu
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
            $(BUILD)/train_fast $(BUILD)/train_fast32 $(BUILD)/test_checkpoint $(BUILD)/test_checkpoint32
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

$(BUILD)/test_checkpoint: $(BUILD)/o16/cuda/tests/test_checkpoint.o $(OBJ16)
	$(NVCC) $(NVFLAGS16) -o $@ $^

$(BUILD)/test_checkpoint32: $(BUILD)/o32/cuda/tests/test_checkpoint.o $(OBJ32)
	$(NVCC) $(NVFLAGS) -o $@ $^

$(BUILD)/train_fast: $(BUILD)/o16/cuda/train_fast.o $(OBJ16) $(BUILD)/o16/tokenizer/bpe.o
	$(NVCC) $(NVFLAGS16) -o $@ $^

$(BUILD)/train_fast32: $(BUILD)/o32/cuda/train_fast.o $(OBJ32) $(BUILD)/o32/tokenizer/bpe.o
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

test-demo: demo/forward.wasm $(BUILD)/bpe
	python3 -m reference.export --from-dump $(DUMP) --out data/ref/weights_debug.bin
	node demo/test_forward.mjs demo/forward.wasm data/ref/weights_debug.bin $(DUMP)
	node demo/test_bpe.mjs $(BUILD)/bpe data/tokenizer.bpe data/corpus_val.txt

# serves whichever model make demo or make demo-chat put in demo/
serve-demo: demo/forward.wasm
	python3 -m http.server 8000 -d demo

.PHONY: demo test-demo serve-demo


# ==================================================================== stage 6: a chat model

# FineWeb-Edu (ODC-By) for pretraining, smol-smoltalk (Apache 2.0) for chat.
# Only these public sources go into this model: no my_writing/ here.
CHAT       = data/chat
CHAT_TOK   = $(CHAT)/tokenizer.bpe
# the tag ids, in order: the three the data has, then four reserved for later stages
CHAT_TAGS  = <web>,<user>,<assistant>,<system>,<tool>,<code>,<book>
MASKED     = data/ref/debug_chat_masked.bin
# B 16 x T 512 = 8,192 tokens a step: 122,000 steps is the 1B token set.
# SFT: 26,000 steps draw 416k rows, about one of each of the 411k conversations
# whose answer fits in 512 tokens. SFT_STEPS=6000 is a 10 minute run.
CHAT_STEPS ?= 122000
SFT_STEPS  ?= 26000
# the pretraining schedule: WSD with the last 20% decaying (DECAY_STEPS=0 is the
# old cosine), lr and beta2 from make sweep, KEEP lists steps whose checkpoints
# stay as ckpt_<step>.bin for branches (make chat-branch)
LR          ?= 1e-3
MIN_LR      ?= 1e-4
BETA2       ?= 0.99
DECAY_STEPS ?= 24400
KEEP        ?=
# make chat-pretrain RESUME=1 (or chat-sft) picks a stopped run up from its last checkpoint
RESUME_FLAG = $(if $(RESUME),-r)
# where the pretraining run lives: a 100M token test run can go elsewhere,
# e.g. make chat-pretrain CHAT_STEPS=12200 PRE_DIR=data/chat/pre100m
PRE_DIR   ?= $(CHAT)/pre

chat-data:
	python.exe prepare_chat.py

# the tokenizer, then every text file as BPE1 tokens, the SFT targets, and PyTorch's starting weights
chat-tokens: $(BUILD)/bpe
	cat $(CHAT)/pretrain_[0-9][0-9][0-9].txt | head -c 420000000 | sed '$$d' > $(CHAT)/tok_sample.txt
	head -c 180000000 $(CHAT)/sft_train.txt | sed '$$d' >> $(CHAT)/tok_sample.txt
	./$(BUILD)/bpe -t -i $(CHAT)/tok_sample.txt -n 16384 -r "$(CHAT_TAGS)" -o $(CHAT_TOK) -v
	python.exe -m reference.chat --check-tokenizer $(CHAT_TOK)
	for f in $(CHAT)/pretrain_[0-9][0-9][0-9].txt; do \
		./$(BUILD)/bpe -e -m $(CHAT_TOK) -i $$f -o $${f%.txt}.bin || exit 1; \
	done
	python.exe prepare_chat.py --join
	./$(BUILD)/bpe -e -m $(CHAT_TOK) -i $(CHAT)/pretrain_val.txt -o $(CHAT)/pretrain_val.bin
	./$(BUILD)/bpe -e -m $(CHAT_TOK) -i $(CHAT)/sft_train.txt -o $(CHAT)/sft_train.bin
	./$(BUILD)/bpe -e -m $(CHAT_TOK) -i $(CHAT)/sft_val.txt -o $(CHAT)/sft_val.bin
	python.exe -m reference.chat --tokens $(CHAT)/sft_train.bin --tokenizer $(CHAT_TOK) --n-ctx 512
	python.exe -m reference.chat --tokens $(CHAT)/sft_val.bin --tokenizer $(CHAT_TOK) --n-ctx 512
	python.exe -m reference.export --init --config chat --vocab-from $(CHAT_TOK) --out $(CHAT)/init.bin
	rm -f $(MASKED)

# a checkpoint (about 400 MB) after every eval; -e 1000 keeps the writes to one every couple of minutes
chat-pretrain: $(BUILD)/train_fast
	./$(BUILD)/train_fast -w $(CHAT)/init.bin -D $(CHAT)/pretrain --warmup 2000 -n $(CHAT_STEPS) -e 1000 \
		--lr $(LR) --min-lr $(MIN_LR) --beta2 $(BETA2) --decay-steps $(DECAY_STEPS) $(if $(KEEP),--keep $(KEEP)) \
		-c $(PRE_DIR) $(RESUME_FLAG)

# a shorter model from a kept checkpoint of a long WSD run: its weights and AdamW state,
# its own length and decay, e.g.
#   make chat-branch FROM=data/chat/trunk/ckpt_97600.bin N=122000 DECAY=24400 OUT=data/chat/b1
# and make chat-branch OUT=data/chat/b1 RESUME=1 picks a stopped branch up like any run
chat-branch: $(BUILD)/train_fast
	$(if $(RESUME),./$(BUILD)/train_fast -c $(OUT) -r,./$(BUILD)/train_fast --branch $(FROM) -D $(CHAT)/pretrain \
		-n $(N) --decay-steps $(DECAY) -c $(OUT))

# 7 runs of 100M tokens on the same batches: lr {6e-4, 2e-3, 6e-3} x beta2 {0.95, 0.99}, and the
# baseline twice (the A/A run gives the noise floor) -> bench/results_sweep.md
sweep: $(BUILD)/train_fast
	bash bench/sweep_lr.sh
	python3 bench/sweep_summary.py

# lr 1e-4, warmup 100, a fresh AdamW (the pretrain's moments stay in its own checkpoint)
chat-sft: $(BUILD)/train_fast
	./$(BUILD)/train_fast -w $(PRE_DIR)/weights.bin -D $(CHAT)/sft --sft -n $(SFT_STEPS) -e 500 \
		-c $(CHAT)/sft $(RESUME_FLAG)

# PyTorch and train_fast on the same 500 batches at the chat size, from the same starting weights
chat-compare: $(BUILD)/train_fast
	python.exe -m reference.train --fast --init $(CHAT)/init.bin --data $(CHAT)/pretrain --steps 500 \
		--out $(CHAT)/compare --starts $(CHAT)/compare/starts.txt
	./$(BUILD)/train_fast -w $(CHAT)/init.bin -D $(CHAT)/pretrain -n 500 -s $(CHAT)/compare/starts.txt \
		-o $(CHAT)/compare -S ""
	python3 bench/curves.py --torch $(CHAT)/compare/log.csv --cuda $(CHAT)/compare/fast_log.csv \
		--out bench/curves_chat.svg --md bench/results_curves_chat.md

chat: $(BUILD)/generate
	./$(BUILD)/generate -w $(CHAT)/sft/weights.bin -m $(CHAT_TOK) --chat

$(MASKED):
	python.exe -m reference.dump --config chat --masked

# masked cross entropy against PyTorch, checkpoints and resuming, chat in the KV cache
test-chat: $(BUILD)/test_fast $(BUILD)/test_fast32 $(BUILD)/test_checkpoint $(BUILD)/test_checkpoint32 \
           $(BUILD)/test_generate $(BUILD)/test_generate32 $(MASKED)
	python.exe -m reference.test_chat
	./$(BUILD)/test_fast32 $(DUMP) $(MASKED)
	./$(BUILD)/test_fast $(DUMP) $(MASKED)
	./$(BUILD)/test_checkpoint32 $(DUMP)
	./$(BUILD)/test_checkpoint $(DUMP)
	./$(BUILD)/test_generate32 $(DUMP)
	./$(BUILD)/test_generate $(DUMP)

# the chat model in the browser: about 68 MB of fp16 weights (34M parameters)
demo-chat: demo/forward.wasm $(BUILD)/bpe
	cp $(CHAT_TOK) demo/tokenizer.bpe
	node demo/test_bpe.mjs $(BUILD)/bpe $(CHAT_TOK) $(CHAT)/sft_val.txt
	python.exe -m reference.export --weights $(CHAT)/sft/weights.bin --fp16 --out demo/weights16.bin

# the chat rows of the results section, from both training logs
chat-results:
	python3 bench/chat_results.py

# how often the cross entropy kernels round the right token's gradient to 0, on both val sets
chat-collapse:
	python.exe -m bench.collapse

.PHONY: chat-data chat-tokens chat-pretrain chat-branch sweep chat-sft chat-compare chat test-chat demo-chat \
        chat-results chat-collapse
