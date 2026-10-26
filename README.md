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
pip install pypdf
python prepare_data.py
python build_corpus.py
```

expected output (from `build_corpus.py`):

```
bible         4341 chunks  x1 in train
austen        4536 chunks  x1 in train
data/corpus_train.txt: 7,330,232 chars
data/corpus_val.txt: 796,034 chars
distinct characters: 93
```

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

## how it works

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

## plan

The data scripts above are written. The rest is built in this order:

1. **Tokenizer, in C++.** The tokenizer turns text into a list of numbers the model can
   read. It uses byte pair encoding (BPE), which starts from single
   characters and repeatedly merges the pair of neighboring pieces that appears most
   often into one new piece. A priority queue keeps the most frequent pair on top, so
   the pair counts don't have to be recomputed after every merge.
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
