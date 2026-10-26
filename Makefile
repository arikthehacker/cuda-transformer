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
