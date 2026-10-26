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
CU_SRC   = cuda/common.cu cuda/dump.cu cuda/gpt.cu $(wildcard cuda/kernels/*.cu)
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
