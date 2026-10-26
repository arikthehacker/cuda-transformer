# cuda-transformer

![language](https://img.shields.io/badge/language-C%2B%2B%20%26%20CUDA-blue) ![platform](https://img.shields.io/badge/platform-Linux%20%28WSL2%29-lightgrey) ![gpu](https://img.shields.io/badge/gpu-RTX%202080%20%28Turing%29-lightgrey)
<!-- ![CI](https://github.com/arikthehacker/cuda-transformer/actions/workflows/ci.yml/badge.svg) -->

A GPT-style transformer that I'm building in C++ and CUDA. A GPT-style transformer is a
language model that writes text by predicting what comes next, one small piece of text
at a time. A PyTorch version of the same model is used for two things only, checking
that my CUDA code gets the same numbers and comparing the speed of the two.

The training text is the Berean Standard Bible and Jane Austen's six novels, and you
can add your own writing to it.

## quickstart

```
git clone https://github.com/arikthehacker/cuda-transformer.git
cd cuda-transformer
sudo apt-get install -y build-essential python3-pypdf
python3 prepare_data.py
python3 build_corpus.py
make run
```

The two Python scripts download the training text and write it to
`data/corpus_train.txt` and `data/corpus_val.txt`. `make run` builds the tokenizer,
which is the program that turns text into the numbers the model reads. It trains the
tokenizer on `corpus_train.txt`, then turns both text files into token files.

expected output (from `make run`, the times change from run to run):

```
./build/bpe -t -i data/corpus_train.txt -n 4096 -o data/tokenizer.bpe -v
words  29,844 unique of 1,709,937 pieces, 1,142 pairs, 0.09 s
merge   256 / 4096   (32, 116)    -> 256     " t"           count 181,309
merge   512 / 4096   (108, 97)    -> 512     "la"           count 3,054
...
merge  4095 / 4096   (2520, 554)  -> 4095    " Perry"       count 76
merges 3840, specials 2 (<bible> <austen>), vocab 4098, 0.15 s -> data/tokenizer.bpe
./build/bpe -e -m data/tokenizer.bpe -i data/corpus_train.txt -o data/train.bin
tokens written 2,027,109, chars per token 3.62 (bytes per token 3.66), 0.13 s -> data/train.bin
./build/bpe -e -m data/tokenizer.bpe -i data/corpus_val.txt -o data/val.bin
tokens written 224,728, chars per token 3.54 (bytes per token 3.57), 0.02 s -> data/val.bin
```

`make test` runs the tokenizer's checks. `make DEBUG=1 test` runs them again with the
compiler's address and undefined behavior sanitizers turned on, which stop the program
at the first out of bounds memory access or undefined operation.

## options

To add your own writing, copy `sources.example.txt` to `sources.local.txt` and list
your files in it. The file is split into sections, and each section heading in square
brackets, like `[essays]`, becomes a separate source with that name. A line under a
heading can be a file or folder (`.pdf`, `.docx`, `.txt`, `.md`), `!name` to skip any
file with that name, or `wattpad:<username>` to download every public story from that
Wattpad account:

```
[essays]
~/Documents/essays
!transcript.pdf
```

`sources.local.txt` and everything under `data/` are listed in `.gitignore`, so your
writing never gets committed.

The tokenizer is a command line tool. Running `./build/bpe -h`:

```
Usage: ./build/bpe [-t|-e|-d] [OPTIONS ...]
  -t            train a tokenizer on -i, save it to -o
  -e            encode -i with tokenizer -m, write token ids to -o
  -d            decode token file -i with tokenizer -m, print the text
  -i file       input file
  -o file       output file
  -m file       tokenizer file (for -e and -d)
  -n #          vocab size: bytes + merges (default 4096)
  -v            verbose: print every 256th merge while training
  -h            print this help message
```

## how it works

### training text

`prepare_data.py` writes one clean text file per source into `data/clean/`. Both public
sources are downloaded once and saved in `data/raw/`. The Bible comes from
bereanbible.com as one verse per line, and each book is rewritten as a title followed
by one verse per paragraph, numbered by chapter and verse (`1:1 In the beginning...`).
Austen's novels come from Project Gutenberg. Each novel is trimmed to the story itself,
from its first chapter to its last line, which also removes the license text Project
Gutenberg adds at the start and end. Project Gutenberg's markup for italics and
illustrations is removed too.

Your own files are read as PDF, Word, or plain text. If two files contain the same
text, for example the same essay saved twice, only one is kept.

Text pulled out of a PDF has a line break after every printed line, and a page header
like `Smith 4` (a surname and a page number) on every page. The `reflow` function in
`prepare_data.py` deletes those headers and joins the lines back into paragraphs. It
treats a line as the end of a paragraph when the line ends in sentence punctuation and
is shorter than 80% of a full printed line, where a full line is the length that 75% of
the document's lines fall under:

```python
lines = [l for l in lines if l and not re.fullmatch(r"([A-Z][a-z]+\s*)?\d{1,3}", l)]
full = sorted(len(l) for l in lines)[int(len(lines) * 0.75)]  # typical full line
paras, cur = [], []
for line in lines:
    cur.append(line)
    if re.search(r"[.!?:\"”)]$", line) and len(line) < 0.8 * full:
        paras.append(" ".join(cur))
        cur = []
```

`build_corpus.py` combines the sources into one training file. It splits each source
into chunks of about 1,000 characters and puts a tag line naming the source at the start
of every chunk, such as `<bible>`, `<austen>`, or `<essays>` for an `[essays]`
section. Once the model is trained, starting it with a tag should make it write in that
source's style.

The Bible and Austen's novels have about 4.0 million characters each, so with only the
public sources they make up half of the training text apiece. A folder of your own
writing is usually far smaller than either, so each of your sources is repeated four
times in the training file. The repeat counts are `REPEATS` and `LOCAL_REPEATS` at the
top of `build_corpus.py`.

The last 10% of each source goes into a separate validation file, `corpus_val.txt`,
and is never repeated or trained on. The model's error on that file shows how well it
handles text it has never seen:

```python
parts = chunks(text)
cut = int(len(parts) * (1 - VAL_FRACTION))
tagged = [f"<{name}>\n{c}\n\n" for c in parts]
train += tagged[:cut] * reps
val += tagged[cut:]
```

### tokenizer

The tokenizer in `tokenizer/bpe.cpp` uses byte pair encoding (BPE). It starts with one
token for each of the 256 possible byte values, then repeatedly takes the pair of
neighboring tokens that appears most often and adds it to the vocabulary as a new
token. With `-n 4096` it learns 3,840 of these merges. Each source tag, such as
`<bible>`, gets one token of its own and is never merged.

Before training, the text is split into words, and a merge never joins two words. A
word is a run of letters, a run of digits, or a run of punctuation, each with at most
one leading space. For example, `Let there be light,` becomes `Let`, ` there`, ` be`,
` light`, and `,`. Curly quotes and dashes count as punctuation, so `God’s` splits the
same way `God's` does.

Training works on the 29,844 unique words and how often each appears, instead of the
1.7 million words in the text. A count is kept for every pair, along with the list of
words that contain it, and a heap keeps the most frequent pair on top. After a merge,
only the words in that pair's list are rewritten, and only the counts of the pairs next
to each merged spot change. Outdated heap entries are skipped when they reach the top:

```cpp
while (tok.merges.size() < num_merges && pop_best(heap, st.counts, best))
{
    ...
    vector<uint32_t> list = move(st.where[best.key]);
    st.where.erase(best.key);

    st.touched.clear();
    for (uint32_t wi : list)
        merge_word(words[wi], a, b, c, wi, st);
    ...
}
```

Encoding splits the text into words the same way and remembers the tokens of every word
it has already encoded, so each unique word is worked out once.

`tests/test_bpe.cpp` checks the trainer against a slow version that recounts every pair
after every merge. Both have to learn the same merges on 400 random texts and on 100 KB
of the real validation text. The tests also check that decoding gives back the exact
bytes that went in, for emoji, Windows line endings, and bytes that are not valid
UTF-8.

## plan

The data scripts and the tokenizer are written. The rest is built in this order:

2. **PyTorch reference model.** A small version of the same model in PyTorch. It saves
   its weights, its intermediate values, and its gradients (how much each weight
   should change during training) to files, so the CUDA code can be compared against
   them number by number.
3. **CUDA in 32-bit floats.** Each layer of the model gets a GPU kernel (a function
   that runs on the graphics card) for the forward pass, which makes a prediction, and
   one for the backward pass, which computes the gradients. Each kernel's output has to
   match the PyTorch files from step 2.
4. **Fast CUDA.** Matrix multiplication runs on the RTX 2080's tensor cores (hardware
   units built for small matrix multiplies) and is written in inline PTX, NVIDIA's
   assembly language for its GPUs. Attention runs as one combined kernel. Training uses
   fp16, a 16-bit float format, with loss scaling, which multiplies the loss so small
   gradients don't round to zero. Most models train in bfloat16, a different 16-bit
   format, but the 2080 has no hardware support for it. CUDA Graphs record a whole
   training step once and replay it, which removes the cost of launching each kernel
   separately.

Step 4 is timed against PyTorch, and its matrix multiplication against cuBLAS (NVIDIA's
own matrix library), on the same model size.
