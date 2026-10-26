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
words  25,239 unique of 1,662,539 pieces, 1,136 pairs, 0.08 s
merge   256 / 4096   (32, 116)    -> 256     " t"           count 186,339
merge   512 / 4096   (272, 101)   -> 512     " fe"          count 3,031
...
merge  4095 / 4096   (744, 4051)  -> 4095    " clouds"      count 74
merges 3840, specials 2 (<bible> <austen>), vocab 4098, 0.12 s -> data/tokenizer.bpe
./build/bpe -e -m data/tokenizer.bpe -i data/corpus_train.txt -o data/train.bin
tokens written 1,949,203, chars per token 3.76 (bytes per token 3.80), 0.10 s -> data/train.bin
./build/bpe -e -m data/tokenizer.bpe -i data/corpus_val.txt -o data/val.bin
tokens written 216,395, chars per token 3.68 (bytes per token 3.71), 0.01 s -> data/val.bin
```

`make test` runs the tokenizer's checks. `make DEBUG=1 test` runs them again with the
compiler's address and undefined behavior sanitizers turned on, which stop the program
at the first out of bounds memory access or undefined operation.

The reference model is written in PyTorch and trains on an NVIDIA GPU. Ubuntu does not
let `pip` install packages for the whole system, so PyTorch goes in a virtual
environment:

```
python3 -m venv .venv
. .venv/bin/activate
pip install torch numpy pytest
python -m pytest reference
python -m reference.train
python -m reference.sample --tag bible austen
```

expected output (from `python -m reference.train`):

```
model  12,343,296 parameters | L 6 H 6 C 384 n_ctx 256 | V 4098 Vp 4160
data   train 1,949,203 tokens | val 216,395 tokens | 4,096 tokens per step
mode   exact (fp32, deterministic) on cuda
step    0 | val 8.366 | ln(V) = 8.318
step  250 | train 5.741 | val 5.030 | lr 6.0e-04 | norm 0.97 | 56,352 tok/s | mfu 41.7%
step  500 | train 4.536 | val 4.655 | lr 5.9e-04 | norm 0.78 | 56,153 tok/s | mfu 41.6%
...
step 3000 | train 2.925 | val 3.818 | lr 2.6e-04 | norm 1.28 | 55,779 tok/s | mfu 41.3%
...
step 4000 | train 2.589 | val 3.844 | lr 1.2e-04 | norm 1.60 | 55,510 tok/s | mfu 41.1%
val has not improved for 4 evals, stopping early
done   best val 3.818 at step 3000 -> data/ref/best.pt
speed  55,922 tok/s over 3990 timed steps | MFU 41.4% of 10 TFLOPS
```

Training stops on its own once the loss, the model's average error on the validation
text, has not improved for 4 checks in a row, and `data/ref/best.pt` keeps the model
from the best check. Then `python -m reference.sample --tag bible austen` prints a
sample from each tag:

```
===== bible, seed 1
<bible>
8:16 “Woe to us, O LORD, for we have sinned to be wicked!”

8:17 Restore us Your inheritance, and Your inheritance is our possession.

8:18 Then we are to take away our descendants together; we will sing to our our God on the day we have not finished our land, nor our fathers have obeyed our father David.

8:19 But now, therefore, please let

===== austen, seed 1
<austen>
He was a woman whose spirit of mind might have been risen from the folly of a child; the knowledge of his own vanities, and the greater evil than the warmth of her mind.


[the model started a new chunk: <bible>]
```

The CUDA version builds with NVIDIA's CUDA Toolkit. Under WSL, only the toolkit gets
installed, from NVIDIA's repository for WSL, because the Windows driver already gives
WSL the GPU and a Linux driver must not be installed on top of it:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit
export PATH=/usr/local/cuda/bin:$PATH
python -m reference.dump
python -m reference.dump --full
make test-cuda
make train-cuda
```

The two dump commands write the answer key the CUDA tests compare against.

expected output (from `make test-cuda`, with most rows of each table left out):

```
./build/test_kernels data/ref/debug.bin
data/ref/debug.bin: V 4098 Vp 4160 n_ctx 64 L 2 H 4 C 64, batch B 4 T 64
...
all kernels match the dump
./build/test_gpt data/ref/debug.bin data/ref/debug_full.bin
...
==== data/ref/debug_full.bin: V 4098 Vp 4160 n_ctx 256 L 6 H 6 C 384, batch B 4 T 256
...
backward, every grad.* record (activations in backward order, then parameters)
  name                                   shape                    max abs     max rel
  grad.loss                              ()                     0.000e+00   0.000e+00
  grad.logits                            (4, 256, 4160)         5.821e-11   5.961e-08
  grad.ln_f                              (4, 256, 384)          4.147e-10   4.369e-06
...
the whole model matches the dump
```

expected output (from `make train-cuda`):

```
./build/train_gpt
model  12,343,296 parameters | L 6 H 6 C 384 n_ctx 256 | V 4098 Vp 4160 | weights from data/ref/debug_full.bin
data   train 1,949,203 tokens | val 216,395 tokens | 4,096 tokens per step
step    0 | val 8.366 | ln(V) = 8.318
step  250 | train 5.740 | val 5.006 | lr 6.0e-04 | norm 0.83 | 35,049 tok/s
step  500 | train 4.533 | val 4.626 | lr 5.9e-04 | norm 0.83 | 34,887 tok/s
...
step 3000 | train 2.902 | val 3.819 | lr 2.6e-04 | norm 1.32 | 34,221 tok/s
step 3250 | train 2.820 | val 3.807 | lr 2.2e-04 | norm 1.34 | 34,140 tok/s
...
step 5000 | train 2.411 | val 3.874 | lr 6.0e-05 | norm 1.69 | 34,130 tok/s
speed  34,387 tok/s over 4990 timed steps -> data/ref/cuda_log.csv
```

The CUDA model starts from the same weights as `reference/train.py`, so both start at
a validation loss of 8.366, and at step 3,000 they reach 3.819 and 3.818. The curves
are close but not identical, because the two pick their random training batches with
different random number generators. The CUDA trainer does not stop early, so its
validation loss rises again after step 3,250, once the model starts memorizing the
training text. At 34,387 tokens per second it runs at 61% of the speed of PyTorch's
exact mode, before any speed tuning.

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

The reference model's scripts each take `-h`. Running `python -m reference.train -h`:

```
usage: train.py [-h] [--steps STEPS] [--batch BATCH] [--out OUT] [--fast] [--resume]
                [--device DEVICE]

Train the reference GPT. Run from the repo root.

options:
  -h, --help       show this help message and exit
  --steps STEPS    optimizer steps (default 5000)
  --batch BATCH    sequences per step (default 16)
  --out OUT        folder for checkpoints and logs (default data/ref)
  --fast           fp16 autocast + scaled_dot_product_attention
  --resume         continue from OUT/last.pt
  --device DEVICE  cuda or cpu (default: cuda when there is one)
```

Sampling and the answer key:

```
python -m reference.sample --tag bible austen --seed 1 2    # four samples
python -m reference.dump                                    # tiny model -> data/ref/debug.bin
python -m reference.dump --full                             # full size -> data/ref/debug_full.bin
python -m reference.dump --compare data/ref/debug.bin mine.bin   # error table
```

The CUDA trainer takes flags too. Running `./build/train_gpt -h`:

```
Usage: ./build/train_gpt [OPTIONS ...]
  -d file       dump with the model sizes and starting weights (default data/ref/debug_full.bin)
  -n #          optimizer steps (default 5000)
  -b #          sequences per step (default 16)
  -e #          steps between evals (default 250)
  -o dir        folder for cuda_log.csv (default data/ref)
  -h            print this help message
```

`make sanitize-cuda` runs the kernel tests under NVIDIA's `compute-sanitizer`, which
reports any out of bounds memory access (memcheck) and any two threads racing on the
same shared memory (racecheck). Under WSL it only starts once Windows lets debuggers
attach to the GPU. That takes two registry values, added from an administrator
terminal on Windows:

```
REG ADD HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm /f /v EnableDebugInterface /t REG_DWORD /d 1
REG ADD "HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger" /f /v EnableInterface /t REG_DWORD /d 1
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
illustrations is removed too, along with the title page that Pride and Prejudice
repeats between its volumes.

The ebooks also break every line at about 71 characters, and a model trained on that
learns to start a new line in the middle of a sentence. The `unwrap` function joins a
line to the next one when the next line's first word would not have fit on it, which
is the test a word wrapper uses. Poems and the closing lines of letters stop well short
of the width, so they keep their line breaks:

```python
next_word = line.split(" ", 1)[0]
if len(prev) + 1 + len(next_word) > width - WRAP_SLACK:  # the word wrap broke here
    glue = "" if prev.endswith("—") or line.startswith("—") else " "
    joined += glue + line
else:                                               # a line break on purpose
    joined += "\n" + line
```

This removes 52,308 of the 52,349 line breaks inside paragraphs. `WRAP_SLACK` is 10
characters, because some lines in these editions wrap a few characters early, while a
poem's lines end 15 or more characters short. An em dash at a break joins with no
space, since Austen writes her dashes without spaces.

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

Training works on the 25,239 unique words and how often each appears, instead of the
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

### reference model

`reference/` is the model written in PyTorch. Its job is to be the answer key. Every
number the CUDA code computes later is checked against a number this model wrote to a
file. It follows GPT-2's layout at a smaller size, with 6 layers, 6 attention heads of
64 dimensions each, a 384 wide hidden state, and a 256 token context, for 12,343,296
parameters. The vocabulary is padded up to a multiple of 64 (4,098 ids become 4,160), so
every matrix divides evenly into the 64 wide blocks that GPU code works on. The padded
ids are never sampled and their weights stay 0. The output layer reuses the token
embedding matrix.

The model runs in two modes. Exact mode uses 32-bit floats and PyTorch's deterministic
algorithms, so the same seed gives the same numbers on every run. It writes LayerNorm
and attention out step by step, and each step can record its output under a name:

```python
scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)                    # (B, H, T, T)
scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))      # (B, H, T, T) future = -inf
_rec(rec, prefix + "scores", scores)
probs = torch.softmax(scores, dim=-1)           # (B, H, T, T) subtracts each row's max first
_rec(rec, prefix + "probs", probs)
y = probs @ v                                   # (B, H, T, D)
```

Recording also keeps the gradient of every recorded value. A gradient says how much the
loss changes when that value changes, and training moves each weight against its
gradient. Fast mode uses 16-bit floats on the RTX 2080's tensor cores (hardware units
built for small matrix multiplies) and PyTorch's fused attention. It is the speed the
CUDA code has to beat. In a 500 step test, both modes reached the same losses to three
decimal places. Exact mode trains at about 56,000 tokens per second, 41% of what the
card's 32-bit cores can do at most, and fast mode at about 185,000, 34% of what the
tensor cores can do at most.

`reference/dump.py` runs a small version of the model (2 layers, 64 wide) through one
forward pass, which makes a prediction and its loss, one backward pass, which computes
the gradients, and one step of AdamW, the optimizer that updates the weights. It writes
every weight, recorded value, gradient and updated weight to `data/ref/debug.bin`, 164
tensors in 20 MB. Writing it twice gives byte-identical files. `--compare` reads two of
these files and prints the largest error of every tensor in file order, and it marks the
first tensor over the tolerance, because that is where a bug starts.

`python -m pytest reference` runs 12 checks, including these four. A fresh model's loss
has to be the natural log of the vocabulary size, changing a later token can never
change an earlier prediction, the model has to be able to memorize a single batch, and
PyTorch's `gradcheck` has to agree with the backward pass in 64-bit floats.

### CUDA

`cuda/` is the same model in C++ and CUDA, in 32-bit floats. Every layer has a GPU
kernel, a function that runs on the graphics card, for its forward pass and another for
its backward pass, one `.cu` file per operation in `cuda/kernels/`. All the weights sit
back to back in one block of GPU memory, in the same order as the dump, and the
gradients and the two AdamW buffers each get one more block with the same layout. Every
intermediate value has a buffer named after its record in the dump, so a test compares
the two by name.

Every matrix multiply in the model, attention included, goes through one function,
`sgemm`, modeled on cuBLAS's strided batched multiply. Each linear layer is
three calls to it, and passing `beta = 1` makes the backward ones add into their
gradient instead of overwriting it:

```cpp
// dinp (N,IC) += dout (N,OC) @ W (OC,IC):  M N, N IC, K OC, beta 1
sgemm(false, false, N, IC, OC, 1.0f, dout, OC, 0, w, IC, 0, 1.0f, dinp, IC, 0, 1);
// dW (OC,IC) += dout^T @ inp:  transA, M OC, N IC, K N, beta 1
sgemm(true, false, OC, IC, N, 1.0f, dout, OC, 0, inp, IC, 0, 1.0f, dw, IC, 0, 1);
```

The fast `sgemm` loads 64 by 16 tiles into shared memory and gives each thread 4 by 4
outputs. A simple version with one thread per output stays in the code as its test,
and the two give identical results on every shape tested.

`cuda/tests/test_kernels.cu` runs each kernel on inputs taken straight from the dump,
so one bug can't hide behind another. When a value feeds two places, its gradient is
the sum of two paths, and the test separates them by subtraction: the first
LayerNorm's own share of `grad.emb` is `grad.emb - grad.h.0.res1`. In the small dump,
`reference/dump.py` adds random noise to every bias and LayerNorm weight, because at
their starting values of 0 and 1 a kernel that forgets a bias still matches.
`cuda/tests/test_gpt.cu` then runs the whole model on both dumps and compares every
recorded value, every gradient and the weights after one AdamW step, 462 comparisons
in all. The largest difference at the full size is 4.4e-6 of a tensor's largest value,
against a limit of 1e-4.

One slice is compared differently. The key bias, the middle third of each
`attn.c_attn.b`, adds the same amount to every score in a row, and softmax ignores
that, so its true gradient is 0 and the dump holds only rounding noise there. That
slice's AdamW update is checked against an absolute limit of 1e-6.

## plan

The data scripts, the tokenizer, the PyTorch reference model and the 32-bit CUDA
version are written. The last step:

4. **Fast CUDA.** Matrix multiplication runs on the tensor cores and is written in
   inline PTX, NVIDIA's assembly language for its GPUs. Attention runs as one combined
   kernel. Training uses fp16, a 16-bit float format, with loss scaling, which
   multiplies the loss so small gradients don't round to zero. Most models train in
   bfloat16, a different 16-bit format, but the 2080 has no hardware support for it.
   CUDA Graphs record a whole training step once and replay it, which removes the cost
   of launching each kernel separately.

Step 4 is timed against PyTorch, and its matrix multiplication against cuBLAS (NVIDIA's
own matrix library), on the same model size.
