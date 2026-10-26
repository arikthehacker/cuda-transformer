// generate.cu: text from a weights.bin file, with a KV cache on the GPU.
//
//   ./build/generate -w data/ref/weights.bin -t bible -n 300 -T 0.8 -k 40 -s 1
//   ./build/generate -t bible,austen -s 1,2          every pair, like reference/sample.py
//   ./build/generate -b                              tokens per second: graph, plain launches, no cache
//   ./build/generate -w data/chat/sft/weights.bin -m data/chat/tokenizer.bpe --chat
//                                                    a conversation: one line in, one answer out
//
// --chat reads one message per line from stdin (an empty line or the end of
// input stops) and keeps the whole conversation in the KV cache. Each
// message goes in as "<user>\n" + line + "\n<assistant>\n", one decode step
// per token; then the answer is sampled until the model writes a tag, and
// printed as it comes. After each answer stdout gets a line holding only
// "\x1e" (the ASCII record separator), so a program reading the pipe knows
// where an answer ends. Everything else (the banner, the "you>" prompt) goes
// to stderr.
//
// The weights can come from PyTorch (reference/export.py) or from
// train_fast. Only a model trained on the public sources goes anywhere public.

#include <getopt.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "bpe.h"
#include "common.h"
#include "data.h"
#include "generator.h"
#include "weights.h"

using namespace std;

static void usage(const char *prog)
{
    printf("Usage: %s [OPTIONS ...]\n", prog);
    printf("  -w file       weights.bin (default data/ref/weights.bin)\n");
    printf("  -m file       tokenizer (default data/tokenizer.bpe)\n");
    printf("  -t tags       tag(s) to start from, comma separated (default bible)\n");
    printf("  -n #          new tokens per sample (default 300)\n");
    printf("  -T #          temperature, 0 = always the most likely token (default 0.8)\n");
    printf("  -k #          sample from the k most likely tokens, 0 = all (default 40)\n");
    printf("  -s seeds      seed(s), comma separated (default 1)\n");
    printf("  -c mode       graph (KV cache + CUDA Graph), plain (KV cache), nocache (default graph)\n");
    printf("  -K            keep going past the next tag (by default a sample ends where a new chunk would start)\n");
    printf("  --chat        a conversation on stdin / stdout (the stage 6 chat model); -n caps each answer\n");
    printf("  -b            benchmark the three modes instead of printing text\n");
    printf("  -o file       with -b: also write the table as markdown (bench/results.py reads it)\n");
    printf("  -h            print this help message\n");
}

static vector<string> split(const string &s)
{
    vector<string> out;
    stringstream in(s);
    string part;
    while (getline(in, part, ','))
    {
        if (!part.empty())
        {
            out.push_back(part);
        }
    }
    return out;
}

static int tag_id(const Tokenizer &tok, string tag)
{
    if (tag.front() != '<')
    {
        tag = "<" + tag + ">";
    }
    for (size_t k = 0; k < tok.specials.size(); k++)
    {
        if (tok.specials[k] == tag)
        {
            return 256 + static_cast<int>(tok.merges.size() + k);
        }
    }
    string all;
    for (const string &s : tok.specials)
    {
        all += " " + s;
    }
    fprintf(stderr, "error: unknown tag %s, the tokenizer has:%s\n", tag.c_str(), all.c_str());
    exit(1);
}

// The tag id of name, or -1
static int find_tag(const Tokenizer &tok, const string &name)
{
    for (size_t k = 0; k < tok.specials.size(); k++)
    {
        if (tok.specials[k] == name)
        {
            return 256 + static_cast<int>(tok.merges.size() + k);
        }
    }
    return -1;
}

// What prepare_chat.py does to every line of training text, for one line typed in
static string clean_line(string line)
{
    string out;
    for (char c : line)
    {
        if (c != '\r' && c != '\x1e')
        {
            out += c;
        }
    }
    size_t end = out.find_last_not_of(" \t");
    out = end == string::npos ? "" : out.substr(0, end + 1);
    static const regex tag_line("^<[a-z_]+>$");
    if (regex_match(out, tag_line))
    {
        out = " " + out;                    // a line like "<div>" would become a tag: keep it text
    }
    return out;
}

static int chat(Generator &gen, Tokenizer &tok, int max_new, SampleConfig sc, unsigned long long seed,
                Generator::Mode mode)
{
    int user = find_tag(tok, "<user>"), assistant = find_tag(tok, "<assistant>");
    if (user < 0 || assistant < 0)
    {
        fprintf(stderr, "error: this tokenizer has no <user> and <assistant> tags, so --chat needs the stage 6 chat "
                        "model (data/chat/tokenizer.bpe)\n");
        return 1;
    }
    int first_tag = 256 + static_cast<int>(tok.merges.size());
    int longest = 2 * gen.cfg.n_ctx;        // a longer message keeps its last part
    gen.chat_start(max_new, seed);
    fprintf(stderr, "chat: one message per line, an empty line ends. Each answer ends with a line holding only \\x1e.\n");
    string line;
    bool started = false;
    fprintf(stderr, "you> ");
    while (getline(cin, line))
    {
        line = clean_line(line);
        if (line.empty())
        {
            break;
        }
        vector<uint32_t> ids = encode(tok, "<user>\n" + line + "\n<assistant>\n");   // ids[0] is <user>
        vector<int> feed;
        if (!started)                       // after an answer the <user> tag is already in place
        {
            feed.push_back(user);
        }
        size_t body = 1;
        if (ids.size() - 1 > static_cast<size_t>(longest))
        {
            fprintf(stderr, "(the message is %zu tokens, only its last %d go in)\n", ids.size() - 1, longest);
            body = ids.size() - longest;
            feed.push_back('\n');          // still a newline after the tag, like every turn
        }
        feed.insert(feed.end(), ids.begin() + body, ids.end());
        gen.chat_feed(feed);
        started = true;

        string held;                        // newlines at the end of what came so far: printed only if more text follows
        gen.chat_answer(max_new, sc, first_tag, user, mode, [&](int t)
                        {
                            if (t >= first_tag)
                            {
                                return;
                            }
                            const string &b = tok.vocab[t];
                            size_t end = b.find_last_not_of('\n');
                            if (end == string::npos)
                            {
                                held += b;
                                return;
                            }
                            fwrite(held.data(), 1, held.size(), stdout);
                            fwrite(b.data(), 1, end + 1, stdout);
                            held = b.substr(end + 1);
                            fflush(stdout);
                        });
        printf("\n\x1e\n");
        fflush(stdout);
        fprintf(stderr, "you> ");
    }
    fprintf(stderr, "\n");
    return 0;
}

int main(int argc, char *argv[])
{
    string weights_path = "data/ref/weights.bin";
    string tok_path = "data/tokenizer.bpe";
    string tags = "bible", seeds = "1";
    int max_new = 300;
    SampleConfig sc = {0.8f, 40};
    Generator::Mode mode = Generator::Mode::CachedGraph;
    bool keep_going = false, bench = false, chat_mode = false;
    string md_path;
    static const option long_opts[] = {
        {"chat", no_argument, nullptr, 1000},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };
    int opt = 0;
    while ((opt = getopt_long(argc, argv, "w:m:t:n:T:k:s:c:Kbo:h", long_opts, nullptr)) != -1)
    {
        switch (opt)
        {
        case 1000:
            chat_mode = true;
            break;
        case 'w':
            weights_path = optarg;
            break;
        case 'm':
            tok_path = optarg;
            break;
        case 't':
            tags = optarg;
            break;
        case 'n':
            max_new = atoi(optarg);
            break;
        case 'T':
            sc.temperature = static_cast<float>(atof(optarg));
            break;
        case 'k':
            sc.top_k = atoi(optarg);
            break;
        case 's':
            seeds = optarg;
            break;
        case 'c':
            if (string(optarg) == "graph")
            {
                mode = Generator::Mode::CachedGraph;
            }
            else if (string(optarg) == "plain")
            {
                mode = Generator::Mode::Cached;
            }
            else if (string(optarg) == "nocache")
            {
                mode = Generator::Mode::NoCache;
            }
            else
            {
                fprintf(stderr, "error: -c takes graph, plain or nocache\n");
                return 1;
            }
            break;
        case 'K':
            keep_going = true;
            break;
        case 'b':
            bench = true;
            break;
        case 'o':
            md_path = optarg;
            break;
        case 'h':
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 1;
        }
    }
    if (max_new < 1)
    {
        fprintf(stderr, "error: -n must be positive\n");
        return 1;
    }

    Weights w = read_weights(weights_path);
    Tokenizer tok;
    if (!load(tok, tok_path))
    {
        fprintf(stderr, "error: cannot read %s\n", tok_path.c_str());
        return 1;
    }
    if (static_cast<int>(tok.vocab.size()) != w.cfg.V)
    {
        fprintf(stderr, "error: %s has %zu ids but %s was trained with %d\n", tok_path.c_str(), tok.vocab.size(),
                weights_path.c_str(), w.cfg.V);
        return 1;
    }
    Generator gen(w.cfg, w.params);
    fprintf(chat_mode ? stderr : stdout, "%s: %s parameters, L %d H %d C %d n_ctx %d, %s\n\n", weights_path.c_str(),
            with_commas(static_cast<long long>(w.params.size())).c_str(), w.cfg.L, w.cfg.H, w.cfg.C, w.cfg.n_ctx,
            FLOATX_NAME);
    if (chat_mode)
    {
        if (mode == Generator::Mode::NoCache)
        {
            fprintf(stderr, "error: --chat keeps the conversation in the cache, so -c nocache does not apply\n");
            return 1;
        }
        return chat(gen, tok, max_new, sc, strtoull(split(seeds).at(0).c_str(), nullptr, 10), mode);
    }

    if (bench)
    {
        vector<int> prompt = {tag_id(tok, split(tags).at(0)), 10};
        printf("generation speed, %d new tokens, temperature %g, top_k %d (each mode once to warm up, then timed)\n",
               max_new, sc.temperature, sc.top_k);
        struct Row
        {
            const char *name;
            Generator::Mode mode;
        } rows[] = {{"KV cache + CUDA Graph", Generator::Mode::CachedGraph},
                    {"KV cache, plain launches", Generator::Mode::Cached},
                    {"no cache (full forward per token)", Generator::Mode::NoCache}};
        FILE *md = md_path.empty() ? nullptr : fopen(md_path.c_str(), "w");
        if (md)
        {
            fprintf(md, "| generation (%s, %d new tokens, temperature %g, top_k %d) | tok/s | ms |\n|---|---|---|\n",
                    FLOATX_NAME, max_new, sc.temperature, sc.top_k);
        }
        for (const Row &r : rows)
        {
            gen.generate(prompt, max_new, sc, 1, r.mode);
            gen.generate(prompt, max_new, sc, 1, r.mode);
            string speed = with_commas(llround(max_new / (gen.last_ms / 1000.0)));
            printf("  %-36s %10s tok/s   (%.1f ms)\n", r.name, speed.c_str(), gen.last_ms);
            if (md)
            {
                fprintf(md, "| %s | %s | %.1f |\n", r.name, speed.c_str(), gen.last_ms);
            }
        }
        if (md)
        {
            fclose(md);
            printf("table -> %s\n", md_path.c_str());
        }
        return 0;
    }

    for (const string &t : split(tags))
    {
        for (const string &seed : split(seeds))
        {
            vector<int> prompt = {tag_id(tok, t), 10};
            vector<int> out = gen.generate(prompt, max_new, sc, strtoull(seed.c_str(), nullptr, 10), mode);
            vector<uint32_t> ids(out.begin(), out.end());
            string ended;
            if (!keep_going)
            {
                int first_special = 256 + static_cast<int>(tok.merges.size());
                for (size_t k = prompt.size(); k < ids.size(); k++)
                {
                    if (static_cast<int>(ids[k]) >= first_special)
                    {
                        ended = "\n[the model started a new chunk: " + tok.vocab[ids[k]] + "]";
                        ids.resize(k);
                        break;
                    }
                }
            }
            printf("===== %s, seed %s\n%s%s\n\n", t.c_str(), seed.c_str(), decode(tok, ids).c_str(), ended.c_str());
            fprintf(stderr, "(%d tokens in %.1f ms, %s tok/s)\n", max_new, gen.last_ms,
                    with_commas(llround(max_new / (gen.last_ms / 1000.0))).c_str());
        }
    }
    return 0;
}
