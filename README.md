# cuda-transformer

![language](https://img.shields.io/badge/language-C%2B%2B%20%26%20CUDA-blue) ![platform](https://img.shields.io/badge/platform-Linux%20%28WSL2%29-lightgrey) ![gpu](https://img.shields.io/badge/gpu-RTX%202080%20%28Turing%29-lightgrey)
<!-- ![CI](https://github.com/arikthehacker/cuda-transformer/actions/workflows/ci.yml/badge.svg) -->

A GPT-style transformer that I'm building in C++ and CUDA. A GPT-style transformer is a
language model that writes text by predicting what comes next, one small piece of text
at a time. A PyTorch version of the same model is used for two things only, checking
that my CUDA code gets the same numbers and comparing the speed of the two. The fast
version trains in 16-bit floats on the RTX 2080's tensor cores, with its matrix
multiplies written in inline PTX, and runs about 7% faster than PyTorch on the same
card.

The training text is the Berean Standard Bible and Jane Austen's six novels, and you
can add your own writing to it.

A second, bigger model is for chatting. It learns English from FineWeb-Edu, a large set
of educational web pages, and then learns to answer from the conversations in
smol-smoltalk. `make chat` talks to it in the terminal.

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

The fast version builds every program twice from the same code, once with 16-bit floats
and once with 32-bit floats:

```
make fast
make test-fast
make train-fast
```

The 32-bit build is there to prove the code right. It has to match the dump as closely
as the first CUDA version does, so a speedup that breaks the math fails a test instead
of showing up as a better number. `make test-fast` runs the first CUDA version's tests,
then the 32-bit build's, then the 16-bit build's.

expected output (from `make test-fast`, with each test's error table left out):

```
./build/test_kernels data/ref/debug.bin
all kernels match the dump
./build/test_gpt data/ref/debug.bin data/ref/debug_full.bin
the whole model matches the dump
./build/test_gemm32
every GEMM matches
./build/test_fast32 data/ref/debug.bin
the fast model matches
./build/test_generate32 data/ref/debug.bin
generation matches
./build/test_gemm
every GEMM matches
./build/test_attn data/ref/debug.bin data/ref/debug_full.bin
flash attention matches
./build/test_fast data/ref/debug.bin data/ref/debug_full.bin
the fast model matches
./build/test_generate data/ref/debug.bin
generation matches
```

expected output (from `make train-fast`, with most steps left out):

```
./build/train_fast
model  12,343,296 parameters | L 6 H 6 C 384 n_ctx 256 | V 4098 Vp 4160 | weights from data/ref/debug_full.bin
data   train 1,949,203 tokens | val 216,395 tokens | 4,096 tokens per step | batches own (mt19937_64, seed 1337)
mode   fp16 | gemm mma.sync | attention flash | graph on | loss scale dynamic from 65,536
step    0 | val 8.366 | ln(V) = 8.318
step  250 | train 5.740 | val 5.005 | lr 6.0e-04 | norm 0.85 | S 65536 | skipped 0 | 194,940 tok/s | mfu 36.1%
...
       step 2000: 2000 finite steps in a row, loss scale up to 131072
step 2000 | train 3.280 | val 3.906 | lr 4.3e-04 | norm 1.14 | S 131072 | skipped 0 | 193,505 tok/s | mfu 35.8%
...
step 3500 | train 2.725 | val 3.807 | lr 1.8e-04 | norm 1.39 | S 131072 | skipped 0 | 193,394 tok/s | mfu 35.8%
...
step 5000 | train 2.409 | val 3.869 | lr 6.0e-05 | norm 1.70 | S 262144 | skipped 0 | 193,378 tok/s | mfu 35.8%
speed  193,860 tok/s | MFU 35.9% of 40 TFLOPS over 4990 timed steps -> data/ref/fast_log.csv
done   best val 3.807 at step 3500 | 0 of 5000 steps skipped | final loss scale 262144
saved  data/ref/cuda_weights.bin
```

At 193,860 tokens per second it runs 5.6 times as fast as the 32-bit CUDA version, and
it reaches a validation loss of 3.807, where PyTorch's exact mode reached 3.818. `S` is
the loss scale, which [how it works](#fast-cuda) explains. It doubles after every 2,000
steps in a row that don't overflow.

`generate` writes text with the trained weights on the GPU:

```
./build/generate -w data/ref/cuda_weights.bin -t bible,austen -n 120
```

expected output:

```
data/ref/cuda_weights.bin: 12,343,296 parameters, L 6 H 6 C 384 n_ctx 256, fp16

===== bible, seed 1
<bible>
4:11 and the chief priests, Levites, and the Levites in all, were numbered 67,600.

4:12 These were the descendants of Aaron by their divisions numbered 1,52; these were the heads of the families of the descendants of Hezon, the descendants of Nathan, and the heads of the families of families.

4:13 These were the heads of the families: the descendants of Noadiah son of Merari.

4:14 These were the sons of Zechariah, the son of Hashab

(120 tokens in 75.6 ms, 1,588 tok/s)
===== austen, seed 1
<austen>
Concent on hearing this, my dear, I hope you will be very glad to see you; and I am so afraid, when I put all my heart to do very well, and then I may be sure our dear madam, if I am so nice, and I will be a very good dinner for one, I wish we shall not be in half a hint. But my dear, do not let me be so happy with her. I always think if she would.”


[the model started a new chunk: <bible>]

(120 tokens in 38.5 ms, 3,114 tok/s)
```

The first sample's speed includes warming up the GPU. `python -m reference.export`
turns a PyTorch checkpoint into the same `weights.bin` format, and
`python -m reference.sample --ckpt` reads either kind of file, so a model trained by one
side can write on the other.

The browser demo runs the same model on the CPU, compiled to WebAssembly, a compact
instruction format that web browsers run at close to native speed. It needs clang, lld
and Node.js:

```
sudo apt-get install -y clang lld nodejs python3-numpy
make test-demo
python -m reference.export --fp16 --out demo/weights16.bin
make serve-demo
```

Then open http://localhost:8000. Only a model trained on the public sources should go
into `demo/weights16.bin`, because the page hands that file to anyone who opens it.

expected output (from `make test-demo`):

```
clang++ --target=wasm32 -O3 -msimd128 -mbulk-memory -nostdlib -fno-exceptions -fno-rtti -std=c++17 -Wall -Wextra -Wl,--no-entry -o demo/forward.wasm demo/forward.cpp
python3 -m reference.export --from-dump data/ref/debug.bin --out data/ref/weights_debug.bin
data/ref/debug.bin -> data/ref/weights_debug.bin: 370,432 values, fp32
node demo/test_forward.mjs demo/forward.wasm data/ref/weights_debug.bin data/ref/debug.bin
wasm model: L 2 H 4 C 64 n_ctx 64 V 4098 Vp 4160
logits at every position vs the dump: worst 4.47e-7 at pos 11 (tol 0.00001) ok
greedy sample() is the argmax: ok
top_k 40 keeps the 40 largest (u = 0 picks id 282, u = 1 picks id 3931): ok
64 tokens in 19.5 ms here
```

## your own writing

To train the model on your own writing, put your files in a folder called `my_writing/`
at the top of the repo, with one folder inside it for each style of writing. Each
folder's name becomes the tag the model is started with, so the files in
`my_writing/essays/` train under `<essays>`. Files placed directly in `my_writing/` go
under `<mine>`. PDF, Word (`.docx`), `.txt` and `.md` files work, and git ignores the
whole folder.

`pipeline.py` runs every step in order, from your files to a trained model:

```
python pipeline.py add essays ~/Documents/essays
python pipeline.py build
python playground.py
```

`add` copies the files into `my_writing/essays/` and leaves the originals where they
are. `build` gathers the text, mixes it with the Bible and Austen, trains the tokenizer,
and trains the model in fast mode, which stops on its own once the validation loss
stops improving. `build --no-public` leaves out the Bible and Austen and trains on your
writing alone. That needs at least 20,000 characters, and the model writes more than
scraps of your own sentences only once it has a few hundred thousand.

On Windows, the tokenizer is built and run inside WSL, so `g++` and `make` have to be
installed there. `python pipeline.py check` lists anything the build is missing, with
the command that installs it.

expected output (from `python pipeline.py build --steps 1500`, with a Sherlock Holmes
collection from Project Gutenberg standing in for your own writing):

```
[1/4] gather your writing
  bible...
    -> data/clean/bible.txt: 4,040,424 chars, 756,414 words
  ...
  detective_stories...
    sherlock.txt (580,876 chars)
    -> data/clean/detective_stories.txt: 580,876 chars, 107,558 words
done in 1.3 s

[2/4] mix it into training text
  bible         4341 chunks  x1 in train
  austen        4535 chunks  x1 in train
  detective_stories   681 chunks  x4 in train
...
[3/4] train the tokenizer
  ...
  merges 3840, specials 3 (<bible> <detective_stories> <austen>), vocab 4099, 0.16 s -> data/tokenizer.bpe
done in 4.0 s

[4/4] train the model
  ...
  step 1500 | train 3.699 | val 4.152 | lr 6.0e-05 | norm 1.06 | 184,228 tok/s | mfu 34.1%
  done   best val 4.152 at step 1500 -> data/ref\best.pt
done in 46.4 s

your model is ready: python playground.py, or python gui.py
```

`python gui.py` does the same on one screen that looks like a terminal. The left side
lists your writing with buttons to add and remove it, shows the four steps with a
progress bar on the one running, and writes text in any of the styles, starting from
your own words if you give it some. The right side shows the build log and what the
model wrote. The window uses tkinter, which comes with Python, and opens in about a
tenth of a second.

## chat model

The chat model has 8 layers, 512 numbers per token and 512 tokens of context, 33.9M
parameters in all, against 12.3M for the first model. It trains in two parts.
Pretraining on FineWeb-Edu teaches it English, and fine-tuning on smol-smoltalk teaches
it to answer. Both data sets are public, and nothing from `my_writing/` goes into this
model. The download needs two more packages for the Windows Python and about 10 GB of
free disk:

```
pip install pyarrow huggingface_hub
python.exe prepare_chat.py --max-chars 20000000    # a 20 MB test run first
make chat-data          # FineWeb-Edu and smol-smoltalk -> data/chat/*.txt
make chat-tokens        # tokenizer, token files, SFT targets, starting weights
make test-chat          # masked loss, checkpoints, chat in the KV cache
make chat-pretrain      # 30,000 steps on web text
make chat-sft           # 6,000 steps on the answers of the conversations
make chat               # talk to it
```

expected output (the lines of `make chat-tokens` that check the tokenizer, the times
change from run to run):

```
./build/bpe -t -i data/chat/tok_sample.txt -n 16384 -o data/chat/tokenizer.bpe -v
...
merges 16128, specials 3 (<web> <user> <assistant>), vocab 16387, ...
python.exe -m reference.chat --check-tokenizer data/chat/tokenizer.bpe
data/chat/tokenizer.bpe: 16,387 ids, specials 3 (<web> <user> <assistant>), nothing else
```

`make chat-data` runs `prepare_chat.py`. It downloads FineWeb-Edu's `sample/10BT`
parquet files one at a time, with their names read from the data set's file list, and
writes every web page as `<web>` and its text, about 4.5 GB in files of 256 MB. One page
in every 100 goes to `pretrain_val.txt`. smol-smoltalk's train and test splits become
`sft_train.txt` and `sft_val.txt`, each conversation as `<user>` and `<assistant>`
turns that end with a `<user>` line of its own, so the model learns where an answer
stops. System messages are left out, since `make chat` has no system prompt.

`make chat-tokens` trains the tokenizer on the first 300 MB of web text and the first
300 MB of conversations, then stops if the tokenizer has any tag besides the three. It encodes
every text file, joins the web text's token files into `pretrain_train.bin`, and writes
the SFT targets (see the chat model part of how it works). Last, `reference/export.py --init` writes PyTorch's starting
weights for the chat size to `data/chat/init.bin`, so the CUDA trainer starts from the
same numbers PyTorch would.

`make chat-pretrain` trains on batches of 16 x 512 tokens, so its 30,000 steps cover
246M tokens. The whole set is about 1B tokens, and `make chat-pretrain
CHAT_STEPS=122000` trains on all of it, about 3.5 hours at the 82,000 tokens per
second a short test run measured. A checkpoint goes to `data/chat/pre/ckpt.bin` after every eval, and if the
computer stops, `make chat-pretrain RESUME=1` carries on from the last one with the
same batches and settings. If the PC shuts down under the load, `nvidia-smi -pl 180`
in an administrator terminal on Windows lowers the card's power limit until the next
restart.

`make chat-sft` starts from the pretrained weights with a fresh optimizer, a learning
rate of 1e-4 instead of 6e-4 and 100 warmup steps. Every batch row starts at the
beginning of a conversation, and the loss counts only the answers. 412,948 of the
460,083 conversations have an answer that fits in 512 tokens. 6,000 steps draw 96,000
of them, and `SFT_STEPS=26000` draws each one once on average. `make chat-results` and then `make results` put both runs into the results
section.

`make chat` runs `./build/generate --chat`. Each line typed in is one message, and an
empty line ends the conversation. The whole conversation stays in the KV cache, so
each message only runs its own tokens through the model. The answer prints as it is
written, and after each one comes a line holding only `\x1e`, the ASCII record
separator, so a program reading the output can tell where an answer ends. The prompt
and everything else go to stderr.

## menu

`python playground.py` opens a menu for trying each part on your own text. Run it from
the repo root with the Python that has PyTorch. On Windows, the tokenizer and the CUDA
programs are Linux builds, so the playground runs them through WSL. On start it prints:

```
cuda-transformer playground
try each part on your own text. 0 goes back, Ctrl+C quits.

main menu
---------
  1. tokenizer       see how your text splits into tokens
  2. model           generate text, and see what surprises the model
  3. CUDA            the CUDA tests and a short CUDA training run
  4. training data   sources, sizes and a random chunk
  0. quit
menu choice:
```

- **tokenizer** sends your text through `build/bpe` and shows each token in its own
  colored box, with the token count and a check that decoding gives your text back. It
  also looks up token ids, searches the vocabulary, and lists the longest tokens.
- **model** loads a model saved by `reference/train.py` in `data/ref/`, or trains a
  quick one in fast mode. It writes text from a tag and an opening of your own, or
  scores your text by coloring each token by how likely the model found it. Its chat
  turns each message into a line of dialogue in an Austen novel, like
  `“Do you walk out today?” said Mr. Darcy.`, and the model writes the next paragraph
  as the reply. The model has never seen a chat, so this is the closest it gets to one.
- **CUDA** checks that `nvcc` and the GPU work, runs `make test-cuda`, and trains with
  the CUDA code for a few hundred steps.
- **training data** shows the size of each source and prints a random chunk of the
  training text.

Its scratch files go in `data/playground/`, which git ignores.

## options

Running `python pipeline.py -h`:

```
usage: pipeline.py [-h] {status,add,remove,check,build,gui} ...

Train the model on your own writing.

positional arguments:
  {status,add,remove,check,build,gui}
    status              list your writing and what is built
    add                 copy files or folders into my_writing/TAG/
    remove              delete my_writing/TAG/ (your original files stay)
    check               check that everything the build needs is installed
    build               gather, mix, train the tokenizer, train the model
    gui                 open the window version
```

and `python pipeline.py build -h`:

```
  --no-public           leave out the Bible and Austen
  --steps STEPS         most training steps, it stops early once it stops
                        improving (default 5000)
  --exact               train in exact fp32 mode instead of fast fp16 mode
  --from {gather,mix,tokenize,train}
                        start at this step
```

Writing that you'd rather not copy into `my_writing/` can be listed where it is. Copy
`sources.example.txt` to `sources.local.txt` and list your files in it. The file is
split into sections, and each section heading in square brackets, like `[essays]`,
becomes a separate source with that name. A line under a heading can be a file or
folder (`.pdf`, `.docx`, `.txt`, `.md`), `!name` to skip any file with that name, or
`wattpad:<username>` to download every public story from that Wattpad account:

```
[essays]
~/Documents/essays
!transcript.pdf
```

`my_writing/`, `sources.local.txt` and everything under `data/` are listed in
`.gitignore`, so your writing never gets committed.

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
usage: train.py [-h] [--steps STEPS] [--batch BATCH] [--out OUT] [--fast] [--resume] [--starts FILE]
                [--device DEVICE] [--config {chat,small}] [--data PREFIX] [--tokenizer FILE]
                [--init FILE] [--sft] [--lr LR] [--min-lr MIN_LR] [--warmup WARMUP]

Train the reference GPT. Run from the repo root.

options:
  -h, --help            show this help message and exit
  --steps STEPS         optimizer steps (default 5000)
  --batch BATCH         sequences per step (default 16)
  --out OUT             folder for checkpoints and logs (default data/ref)
  --fast                fp16 autocast + scaled_dot_product_attention
  --resume              continue from OUT/last.pt
  --starts FILE         write every step's batch starts to FILE, one line of B numbers per step,
                        so cuda/train_fast -s FILE trains on the same batches (costs a little
                        speed)
  --device DEVICE       cuda or cpu (default: cuda when there is one)
  --config {chat,small}
                        model size: small (L 6, C 384, n_ctx 256, the default) or chat (L 8, C
                        512, n_ctx 512)
  --data PREFIX         PREFIX_train.bin and PREFIX_val.bin, e.g. data/chat/pretrain (default
                        data/train.bin, data/val.bin)
  --tokenizer FILE      default data/tokenizer.bpe, or tokenizer.bpe in the folder of --data
  --init FILE           start from this weights.bin (and its sizes), not a fresh init
  --sft                 train on the answers only (reference/chat.py); lr 1e-4, warmup 100
  --lr LR               peak learning rate (default 6e-4, --sft 1e-4)
  --min-lr MIN_LR       learning rate at the end (default 6e-5, --sft 1e-5)
  --warmup WARMUP       warmup steps (default 200; 2000 for the chat pretrain)
```

Sampling and the answer key:

```
python -m reference.sample --tag bible austen --seed 1 2    # four samples
python -m reference.dump                                    # tiny model -> data/ref/debug.bin
python -m reference.dump --full                             # full size -> data/ref/debug_full.bin
python -m reference.dump --compare data/ref/debug.bin mine.bin   # error table
python -m reference.dump --config chat --masked             # chat tokenizer, half the targets -1
python -m reference.chat --tokens data/chat/sft_train.bin --n-ctx 512    # SFT targets -> sft_train.tgt
python -m reference.export --init --config chat --vocab-from data/chat/tokenizer.bpe --out data/chat/init.bin
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

The fast trainer takes the same flags and a few more. Running `./build/train_fast -h`:

```
Usage: ./build/train_fast [OPTIONS ...]
  -d file       dump with the model sizes and starting weights (default data/ref/debug_full.bin)
  -w file       start from these weights instead (weights.bin); without -d its header gives the sizes
  -D prefix     data: prefix_train.bin and prefix_val.bin (default data/train.bin, data/val.bin)
  --sft         train on the answers only: targets from prefix_train.tgt and prefix_val.tgt
                (python.exe -m reference.chat), lr 1e-4, min lr 1e-5, warmup 100
  -n #          optimizer steps (default 5000)
  -b #          sequences per step (default 16)
  -e #          steps between evals (default 250)
  --lr x        peak learning rate (default 6e-4, --sft 1e-4)
  --min-lr x    learning rate at the end (default 6e-5, --sft 1e-5)
  --warmup #    warmup steps (default 200, --sft 100; 2000 for the chat pretrain)
  -s file       batch starts from reference/train.py --starts (default: own random batches)
  -c dir        checkpoint dir: dir/ckpt.bin after every eval, and the log and weights.bin go here
  -r            resume from <-c dir>/ckpt.bin, with the settings saved in it
  -g 0|1        capture the step as a CUDA Graph (default 1)
  -G name       GEMM: mma (inline PTX), wmma, simt (fp32 cores) (default mma, fp32 build: simt)
  -A name       attention: flash or unfused (the stage 3 kernels) (default flash, fp32 build: unfused)
  -o dir        folder for the log and the weights (default data/ref, or the -c dir)
  -l name       log file name inside -o (default fast_log.csv)
  -S file       save the trained weights here (default <-o>/cuda_weights.bin, with -c <dir>/weights.bin,
                "" for none)
  -h            print this help message
```

`-G`, `-A` and `-g` turn each speedup off at run time, so one build measures all of
them:

```
./build/train_fast -G simt -A unfused -g 0     # 16-bit floats, matrix multiplies on the regular cores
./build/train_fast -G wmma -A unfused -g 0     # + tensor cores through CUDA's WMMA API
./build/train_fast -G mma -A unfused -g 0      # + tensor cores through inline PTX
./build/train_fast -G mma -A flash -g 0        # + flash attention
./build/train_fast                             # + CUDA Graph
```

Running `./build/generate -h`:

```
Usage: ./build/generate [OPTIONS ...]
  -w file       weights.bin (default data/ref/weights.bin)
  -m file       tokenizer (default data/tokenizer.bpe)
  -t tags       tag(s) to start from, comma separated (default bible)
  -n #          new tokens per sample (default 300)
  -T #          temperature, 0 = always the most likely token (default 0.8)
  -k #          sample from the k most likely tokens, 0 = all (default 40)
  -s seeds      seed(s), comma separated (default 1)
  -c mode       graph (KV cache + CUDA Graph), plain (KV cache), nocache (default graph)
  -K            keep going past the next tag (by default a sample ends where a new chunk would start)
  --chat        a conversation on stdin / stdout (the stage 6 chat model); -n caps each answer
  -b            benchmark the three modes instead of printing text
  -o file       with -b: also write the table as markdown (bench/results.py reads it)
  -h            print this help message
```

The benchmarks behind the results section:

```
make bench-gemm               # every matrix multiply shape against cuBLAS -> bench/results_gemm.md
bash bench/bench_train.sh     # every trainer and speedup -> bench/results_train.md
python -m reference.train --fast --steps 1000 --out data/compare --starts data/compare/starts.txt
./build/train_fast -n 1000 -s data/compare/starts.txt -o data/compare
make curves                   # both loss curves -> bench/curves.svg
python -m reference.export    # data/ref/best.pt -> data/ref/weights.bin
make bench-generate           # generation speed -> bench/results_generate.md
make results                  # copies all of it into this README
make ptxas                    # registers and spills per kernel
make sanitize-fast            # compute-sanitizer on the fast kernels
```

`bench_train.sh` runs PyTorch with `python.exe`, the Windows Python, unless the `PY`
variable names another one. Every number in the results section came from these
commands, run on the public sources only.

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
of every chunk, such as `<bible>`, `<austen>`, or `<essays>` for an `essays` folder.
Once the model is trained, starting it with a tag should make it write in that source's
style.

The Bible and Austen's novels have about 4.0 million characters each, so with only the
public sources they make up half of the training text apiece. A folder of your own
writing is usually far smaller than either, so each of your sources is repeated four
times in the training file. The repeat counts are `REPEATS` and `LOCAL_REPEATS` at the
top of `build_corpus.py`. Without the public sources nothing needs to be repeated, so
each of your sources appears once.

`pipeline.py` also picks the tokenizer's size from the amount of text. It uses 4,096
tokens from 2 million characters up, 2,048 from 500,000, and 1,024 below that, so a
small text isn't split into thousands of rare tokens the model sees only a few times.

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

### fast CUDA

`cuda/train_fast.cu` trains the same model with the same math as the first CUDA
version and changes how the work gets done. Each change can be switched off with a
flag, so the results section measures each one on its own.

**16-bit floats.** fp16 is a 16-bit float format. Its largest value is 65,504 and its
smallest is about 6e-8, far less range than a 32-bit float, but the tensor cores
multiply it about 4 times as fast. Most models train in bfloat16, a 16-bit format with
the range of a 32-bit float, but the 2080 has no hardware support for it. So the code
follows PyTorch's mixed precision rules. The weights the optimizer updates, the weight
gradients, the running sums inside every matrix multiply, and anything that can grow
past 65,504, like attention scores and the loss, stay in 32-bit floats. The matrix
multiply inputs and most activations, the values passed from one layer to the next,
are 16-bit. `precision.h` picks the type at compile time, so every kernel is written
once and builds both ways.

**Loss scaling.** Many gradients are smaller than 6e-8 and would round to 0 in fp16.
Backward starts from the loss gradient multiplied by a scale `S` of 65,536, and the
weight gradients are divided by `S` again before the optimizer uses them. `S` is a power
of two, so multiplying and dividing by it never changes a digit. If a gradient still
overflows to infinity, the step is skipped and `S` halves, and after 2,000 good steps
in a row it doubles, the same rules as PyTorch's `GradScaler`. One small kernel after
backward makes these decisions on the GPU, so the CPU never waits for the GPU in the
middle of a step:

```cpp
if (s.found_inf)
{
    s.skipped++;
    s.good_steps = 0;
    if (cfg.dynamic_scale)
    {
        s.loss_scale *= 0.5f;
    }
}
else
{
    s.adam_t++;
    ...
```

`test_fast` forces each case. It writes an infinity into one gradient, and it sets `S`
to 2^30 so that the loss gradient really overflows. In both, the step has to be
skipped, the weights must not move and `S` has to halve.

**The matrix multiply.** A tensor core multiplies small tiles of fp16 matrices in one
instruction. CUDA's WMMA API reaches them from C++. The fastest version skips the API
and writes the instructions in inline PTX, NVIDIA's assembly language for its GPUs.
Turing's tensor core instruction multiplies a 16 by 8 tile by an 8 by 8 tile:

```cpp
// d += a b on one 16 x 8 x 8 tile, fp16 in, fp32 accumulate
__device__ __forceinline__ void mma_16808(float d[4], const uint32_t a[2], uint32_t b)
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b));
}
```

Each block of threads computes a 128 by 128 tile of the output with 4 warps, the groups
of 32 threads a GPU runs in lockstep, and each warp computes 64 by 64. The input tiles
wait in shared memory, the small fast memory that one block's threads share, and
`ldmatrix` loads them into the exact registers the multiply instruction reads. Shared
memory is split into 32 banks, and threads that read the same bank at the same time wait
for each other, so each row of a tile is shifted across the banks with an XOR:

```cpp
// element offset of chunk `chunk` of row `row` in a tile with 32-half rows
__device__ __forceinline__ int kmaj(int row, int chunk)
{
    return row * BK + ((chunk ^ ((row >> 1) & 3)) << 3);
}
```

Turing can't copy from GPU memory straight into shared memory in the background, so the
next tile travels through registers while the tensor cores work on the current one. The
bias and GELU are applied as the results are written, instead of in kernels of their
own. Weight gradients have few output tiles and a long inner dimension, so they split
that dimension across extra blocks, which add their parts together at the end. Over
every multiply in a training step, these kernels run at 81% of the speed of cuBLAS,
NVIDIA's own matrix library.

**Flash attention.** The first CUDA version writes the full 256 by 256 table of
attention scores for every head to GPU memory, then reads it back for the softmax and
again for the next multiply. Flash attention computes the scores one tile at a time and
keeps a running softmax in registers, so the table never exists in memory. Backward
recomputes the scores from one saved number per row, the log of the row's softmax sum,
instead of storing them. The output of one tensor core multiply is laid out exactly like
the input of the next, so the probabilities go from one multiply into the next without
passing through shared memory. This is the biggest single speedup, from 112,718 to
193,578 tokens per second.

**Smaller fusions.** The residual add and the next LayerNorm run as one kernel. Cross
entropy writes its gradient over the logits in place, since the logits are the largest
activation. GELU is recomputed in backward instead of saving its output, and AdamW
updates every weight in one launch instead of one launch per tensor.

**CUDA Graph.** Launching a kernel costs the CPU a few microseconds, and a training step
launches hundreds of them. A CUDA Graph records the whole step once and replays it with
one launch. That only works if nothing in the step depends on the CPU, so the batch
positions of every step are uploaded before training starts, and the step counter,
learning rate and loss scale all live in GPU memory:

```cpp
auto one_step = [&]()
{
    load_batch(model.x.ptr, model.y.ptr, d_train.ptr, d_starts.ptr, tc.max_steps, model.state.ptr, B, T, s);
    model.train_step();
};
CudaGraph graph;
if (use_graph)
{
    graph.capture(s, one_step);
}
```

Flash attention already removed most of the launches, so the graph adds less than 1%.

`test_gemm` runs every multiply shape and transpose case against the simple 32-bit
`sgemm`. One of its inputs is a matrix of index numbers times the identity matrix, so a
thread that holds the wrong piece of a tile shows up as a visible pattern. `test_fast`
compares every recorded value and gradient against the dump. The 32-bit build has to
stay within 1e-5 forward and 1e-4 backward, like the first CUDA version, and the 16-bit
build within 1e-2 and 3e-2. The last check is the loss curve. `reference/train.py
--starts` saves the position of every batch it trains on, `train_fast -s` trains on the
same batches, and after 1,000 steps the two validation losses differ by 0.11%.

### generation

`generate` writes text one token at a time. Each new token needs the keys and values of
every earlier token in every layer. Instead of running the whole text through the model
again for each token, it keeps them in a KV cache on the GPU and computes only the new
token's row. That turns each layer into a matrix times a vector, which is limited by
reading the 25 MB of fp16 weights once per token, not by the math. Each warp reads one
row of weights with 16-byte loads:

```cpp
// one warp, one row: W[row] . xs, every lane gets the sum
__device__ __forceinline__ float row_dot(const floatX *W, int row, int IC, const float *xs, int lane)
{
    const floatX *w = W + static_cast<long long>(row) * IC;
    float acc = 0.0f;
    for (int c = lane; c < IC / VEC; c += 32)
    {
        acc += dot_vec(w + c * VEC, xs + c * VEC);
    }
    return warp_sum(acc);
}
```

Sampling runs on the GPU too. Temperature comes first, then top-k by a radix select,
which finds the k largest scores in 4 passes that each sort the scores into 256 bins by
8 bits of their value. Then comes softmax, and one draw with a random number from the
CPU, so a seed repeats a sample. Each step reads its token and position from GPU
memory, so one step is recorded as a CUDA Graph and replayed for every token. With the
cache and the graph it writes 3,259 tokens per second, against 734 when it runs the
whole model for every token.

Positions come from a learned table with 256 entries, so when the cache fills up, the
last 128 tokens go through the full model again at positions 0 to 127 and writing
carries on from there. `test_generate` checks that the cached path gives the same
logits as the full forward pass at every position.

### browser demo

`demo/forward.cpp` is the same decode step for the CPU, compiled to WebAssembly with
SIMD instructions, which work on 4 floats at once. It uses no C library and no
Emscripten, and builds to 13 KB. The weights travel as fp16, 25 MB for the full model,
and are widened to 32-bit floats once when the page loads them. On a desktop CPU the
page writes about 200 tokens per second, and like `generate`, it keeps the last half of
the text when the cache fills. `demo/test_forward.mjs` checks its logits against the
dump in Node.js.

`make demo-chat` puts the chat model in the page instead. Its weights are about 68 MB as
fp16 and 136 MB once widened, against 25 MB for the first model, so the page takes
longer to load them, and each token reads almost three times as many weights.

### chat model

**Tags in web text.** The tokenizer turns any line that is only `<letters>` into a tag
of its own, and adds a new one for every such line it sees while training. Web pages
have lines like `<div>`, and one of them would become a token forever. So every line of
web and chat text goes through `safe_lines` first, and `make chat-tokens` stops if the
tokenizer comes out with any tag besides `<web>`, `<user>` and `<assistant>`:

```python
TAG_LINE = re.compile(r"^<[a-z_]+>$")

def safe_lines(text: str) -> str:
    """Every tag-shaped line gets one leading space, so it stays plain text."""
    return "\n".join(" " + line if TAG_LINE.match(line) else line for line in text.split("\n"))
```

**Training on the answers only.** In fine-tuning, the target of a token is the next
token only when that one is part of an answer, and -1 otherwise, which the loss skips.
The `<user>` tag after an answer counts too, since predicting it is how the model
learns to stop. `reference/chat.py` finds each answer as the span from just after
`<assistant>` and its newline up to the next tag:

```python
first = asst + 1 + (t[np.minimum(asst + 1, n - 1)] == NEWLINE)
nxt = np.searchsorted(tag_pos, asst, side="right")
last = np.where(has_next, tag_pos[np.minimum(nxt, tag_pos.size - 1)], n - 1)
counted = _ranges(first[ok], last[ok], n)          # token j is part of an assistant turn
y[:-1] = np.where(counted[1:], t[1:], -1)
```

Batch rows start only where a conversation starts, so a row never begins halfway
through an answer. An answer that does not fit in the 512 tokens from its
conversation's start is dropped with every answer after it, which cuts the
conversation at a turn boundary, and a conversation with no answer left is dropped.
`reference/chat.py` prints how many were kept whole, cut and dropped, and writes the
targets and starts to a `.tgt` file for `train_fast --sft`.

**The masked loss on the GPU.** The loss is the mean over the counted targets, not over
all 8,192 tokens of the batch, or short answers would get smaller gradients than long
ones. One block counts the targets that are not -1 into the GPU-side step state, and
the cross entropy kernel reads that count when it scales the gradient. A row whose
target is -1 gets a loss of 0 and a gradient of 0:

```cpp
int target = targets[n];
if (target < 0)                         // not counted: no loss, no gradient (the whole block takes this branch)
{
    if (threadIdx.x == 0)
    {
        losses[n] = 0.0f;
    }
    if (grad)
    {
        for (int j = threadIdx.x; j < Vp; j += blockDim.x)
        {
            l[j] = from_float<floatX>(0.0f);
        }
    }
    return;
}
```

The count is at least 1, so a batch with nothing counted gives a loss of 0 instead of
dividing by 0. Outside fine-tuning every target counts, the count is 16 x 512, and the
numbers are the same as before. `test_fast` checks the masked loss, the gradients and
the zero rows against `reference/dump.py --config chat --masked`, a dump with about
half its targets set to -1 and one row with none counted at all.

**1B tokens on the GPU.** The pretraining tokens stay on the GPU as 16-bit ids, 2 GB
for 1B tokens instead of 4 GB as 32-bit ints, and the batch kernel widens them as it
copies a batch out. On the PyTorch side, a token file that big stays in CPU memory and
only each batch goes to the GPU.

**Checkpoints.** A 4.5 hour run should survive the PC shutting down, so `train_fast -c`
saves the weights, both AdamW buffers, the step state (loss scale, step, AdamW's step
count) and the run's settings after every eval. The file is written under a temporary
name, flushed to the disk, and only then renamed over the old one, so a crash never
leaves half a checkpoint:

```cpp
bool flushed = fflush(f) == 0;
bool synced = fsync(fileno(f)) == 0;
bool closed = fclose(f) == 0;
...
if (rename(tmp.c_str(), path.c_str()) != 0)
```

`train_fast -r` reads it back, makes the same list of batch starts, and carries on at
the saved step. `test_checkpoint` trains 20 steps straight, then 10 steps, a
checkpoint, a new model loaded from the file and 10 more, and the losses must match
bit for bit. On a GPU, the sums that use atomics add in a different order on every run,
so two straight runs already drift apart, by up to about 1e-5 in the loss. The test
then holds the resumed run to fixed bars, 1e-4 in the losses and 1e-2 in the
parameters. A resume that loses AdamW's m and v or its step count moves the losses by
1e-2 or more, so it still fails.

**Chat in the KV cache.** A message goes in one token at a time. Each token runs
through every layer to put its keys and values in the cache, and the next token is
known, so nothing is sampled:

```cpp
void Generator::feed_one(int token)
{
    cudaStream_t s = model.stream;
    if (chat_pos == cfg.n_ctx)
    {
        int m = cfg.n_ctx / 2;
        prefill(m, chat_n, SampleConfig{0.0f, 0}, true, false);
        push_token(state.ptr, history.ptr, token, m, s);
        chat_pos = m;
    }
    else
    {
        decode_layers();
        push_token(state.ptr, history.ptr, token, -1, s);
        chat_pos++;
    }
    chat_n++;
    chat_total++;
}
```

Then the answer is sampled one token at a time until the model writes a tag. If the tag
is not `<user>`, it is swapped for `<user>`, which then waits as the first token of the
next message. When the cache fills, it keeps the last half, the same as `generate`.
`test_generate` checks that a conversation kept in the cache over 8 turns, past cache
rollovers, answers exactly like the same conversation fed in all at once.

<!-- results start: bench/results.py writes everything up to the end marker -->
## results

### training speed

Same model (L 6, H 6, C 384), batch 16 x 256 tokens, same data. MFU = 6 x parameters x tok/s / peak.

| run | tok/s | MFU | peak |
|---|---|---|---|
| PyTorch exact (fp32) | 55,180 | 40.9% | 10 TFLOPS |
| PyTorch fast (fp16 autocast + SDPA) | 181,885 | 33.7% | 40 TFLOPS |
| stage 3 (fp32 CUDA) | 34,055 | 25.2% | 10 TFLOPS |
| stage 4 fp32 build (simt GEMMs, stage 3 attention, graph) | 38,550 | 28.6% | 10 TFLOPS |
| fp16 + loss scaling, simt GEMMs | 39,087 | 7.2% | 40 TFLOPS |
| + wmma tensor core GEMMs | 94,564 | 17.5% | 40 TFLOPS |
| + mma.sync + ldmatrix GEMMs | 112,718 | 20.9% | 40 TFLOPS |
| + flash attention | 193,578 | 35.8% | 40 TFLOPS |
| + CUDA Graph (all of stage 4) | 194,250 | 36.0% | 40 TFLOPS |

300 steps per run, measured 2026-10-26 on NVIDIA GeForce RTX 2080.
The fused small kernels (residual + LayerNorm, in place cross entropy, one pass unscale + norm,
one launch AdamW, GELU recompute) are part of every train_fast row.

### GEMMs against cuBLAS

Every GEMM shape of one training step, fp16 in and fp32 accumulate. Each cell: TFLOPS, and speed as a percent of cuBLAS (100% = as fast as cuBLAS).

| GEMM | M x N x K | cuBLAS | mma.sync | wmma | simt |
|---|---|---|---|---|---|
| c_attn fwd | 4096 x 1152 x 384 | 26.8 TFLOPS | 21.7 TFLOPS, 81% | 15.0 TFLOPS, 56% | 3.4 TFLOPS, 13% |
| attn c_proj fwd | 4096 x 384 x 384 | 22.6 TFLOPS | 15.8 TFLOPS, 70% | 15.0 TFLOPS, 66% | 3.4 TFLOPS, 15% |
| c_fc fwd | 4096 x 1536 x 384 | 29.8 TFLOPS | 25.7 TFLOPS, 86% | 20.0 TFLOPS, 67% | 4.0 TFLOPS, 14% |
| mlp c_proj fwd | 4096 x 384 x 1536 | 33.3 TFLOPS | 22.2 TFLOPS, 67% | 18.2 TFLOPS, 54% | 3.8 TFLOPS, 11% |
| logits fwd | 4096 x 4160 x 384 | 35.7 TFLOPS | 28.9 TFLOPS, 81% | 18.8 TFLOPS, 53% | 4.1 TFLOPS, 11% |
| c_attn dinp | 4096 x 384 x 1152 | 31.6 TFLOPS | 24.4 TFLOPS, 77% | 14.0 TFLOPS, 44% | 4.0 TFLOPS, 13% |
| attn c_proj dinp | 4096 x 384 x 384 | 25.1 TFLOPS | 20.7 TFLOPS, 82% | 14.0 TFLOPS, 56% | 3.9 TFLOPS, 16% |
| c_fc dinp | 4096 x 384 x 1536 | 30.9 TFLOPS | 23.5 TFLOPS, 76% | 13.8 TFLOPS, 45% | 4.0 TFLOPS, 13% |
| mlp c_proj dinp | 4096 x 1536 x 384 | 31.7 TFLOPS | 27.2 TFLOPS, 86% | 16.7 TFLOPS, 53% | 4.2 TFLOPS, 13% |
| logits dinp | 4096 x 384 x 4160 | 37.6 TFLOPS | 24.0 TFLOPS, 64% | 14.0 TFLOPS, 37% | 4.0 TFLOPS, 11% |
| c_attn dW | 1152 x 384 x 4096 | 25.3 TFLOPS | 23.8 TFLOPS, 94% | 12.3 TFLOPS, 49% | 3.2 TFLOPS, 13% |
| attn c_proj dW | 384 x 384 x 4096 | 21.0 TFLOPS | 18.4 TFLOPS, 88% | 8.8 TFLOPS, 42% | 2.5 TFLOPS, 12% |
| c_fc dW | 1536 x 384 x 4096 | 32.3 TFLOPS | 26.8 TFLOPS, 83% | 15.9 TFLOPS, 49% | 3.3 TFLOPS, 10% |
| mlp c_proj dW | 384 x 1536 x 4096 | 29.6 TFLOPS | 27.0 TFLOPS, 91% | 16.1 TFLOPS, 55% | 3.3 TFLOPS, 11% |
| logits dW | 4160 x 384 x 4096 | 27.4 TFLOPS | 24.4 TFLOPS, 89% | 13.0 TFLOPS, 47% | 4.0 TFLOPS, 15% |
| one training step | | 10.08 ms | 12.43 ms (81%) | 19.75 ms (51%) | 82.11 ms (12%) |

### loss curves

PyTorch fast mode and train_fast on the same batches (reference/train.py --starts, train_fast -s).

![loss curves of PyTorch fast mode and train_fast](bench/curves.svg)

- PyTorch fast: val 4.3128 at step 1000
- CUDA (train_fast): val 4.3081 at step 1000
- difference at step 1000: -0.11% (the bar is about 1%)

### generation

Tokens per second for one sample, prompt and every cache rollover included.

| generation (fp16, 500 new tokens, temperature 0.8, top_k 40) | tok/s | ms |
|---|---|---|
| KV cache + CUDA Graph | 3,259 | 153.4 |
| KV cache, plain launches | 2,981 | 167.7 |
| no cache (full forward per token) | 734 | 681.1 |

### chat model

L 8, H 8, C 512, n_ctx 512, 33.9M parameters, batch 16 x 512 tokens. Pretrained on FineWeb-Edu, then fine-tuned on smol-smoltalk (bench/chat_results.py reads both train_fast logs).

not measured yet (`bench/results_chat.md` does not exist)

<!-- results end -->

## plan

The data scripts, the tokenizer, the PyTorch reference model, the 32-bit CUDA version,
the fast version, generation and the chat model's code are written. Next:

5. **Training the chat model.** Pretraining on FineWeb-Edu, then fine-tuning on
   smol-smoltalk, with the numbers in the results section.
6. **A chat box.** The playground and `gui.py` get a chat box over
   `./build/generate --chat`, which keeps the conversation in the KV cache.

## credits

The chat model is trained on two public data sets:

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), the
  `sample-10BT` subset, by Hugging Face, under the
  [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/).
- [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), by
  Hugging Face, under the Apache 2.0 license.
