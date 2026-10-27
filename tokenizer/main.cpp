// main.cpp: the "bpe" command line tool
//
//   bpe -t -i corpus.txt -n 4096 -o tokenizer.bpe -v     train
//   bpe -t -i chat.txt -n 16384 -r "<web>,<user>,<assistant>" -o tokenizer.bpe    with reserved tags
//   bpe -e -m tokenizer.bpe -i corpus.txt -o train.bin   encode
//   bpe -d -m tokenizer.bpe -i train.bin [-o out.txt]    decode

#include "bpe.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;

static const uint32_t BIN_MAGIC = 0x31455042;   // "BPE1" as bytes
static const uint32_t BIN_VERSION = 1;

static void usage(const char *prog)
{
    printf("Usage: %s [-t|-e|-d] [OPTIONS ...]\n", prog);
    printf("  -t            train a tokenizer on -i, save it to -o\n");
    printf("  -e            encode -i with tokenizer -m, write token ids to -o\n");
    printf("  -d            decode token file -i with tokenizer -m, print the text\n");
    printf("  -i file       input file\n");
    printf("  -o file       output file\n");
    printf("  -m file       tokenizer file (for -e and -d)\n");
    printf("  -n #          vocab size: bytes + merges (default 4096)\n");
    printf("  -p 1|2        pretokenizer for -t: 2 splits contractions and groups digits by 3 (default 2),\n");
    printf("                1 is the stage 1 rules (encode and decode use what the tokenizer file says)\n");
    printf("  -r tags       reserved tags for -t, comma separated: their ids come first, in this order,\n");
    printf("                even if the text never has them (e.g. \"<web>,<user>,<assistant>\")\n");
    printf("  -v            verbose: print every 256th merge while training\n");
    printf("  -h            print this help message\n");
}

static string with_commas(uint64_t v)
{
    string digits = to_string(v);
    string out;
    for (size_t k = 0; k < digits.size(); k++)
    {
        if (k > 0 && (digits.size() - k) % 3 == 0)
            out += ',';
        out += digits[k];
    }
    return out;
}

static double seconds_since(chrono::steady_clock::time_point t0)
{
    return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
}

static bool read_file(const string &path, string &data)
{
    ifstream in(path, ios::binary);
    if (!in)
        return false;
    in.seekg(0, ios::end);
    streamoff size = in.tellg();
    if (size < 0)
        return false;
    in.seekg(0, ios::beg);
    data.resize(static_cast<size_t>(size));
    if (size > 0)
        in.read(&data[0], size);
    return static_cast<bool>(in);
}

static bool write_file(const string &path, const char *data, size_t size)
{
    ofstream out(path, ios::binary);
    if (!out)
        return false;
    out.write(data, static_cast<streamsize>(size));
    out.close();
    return !out.fail();
}

static void put_u32(vector<unsigned char> &buf, size_t at, uint32_t v)
{
    buf[at + 0] = static_cast<unsigned char>(v);
    buf[at + 1] = static_cast<unsigned char>(v >> 8);
    buf[at + 2] = static_cast<unsigned char>(v >> 16);
    buf[at + 3] = static_cast<unsigned char>(v >> 24);
}

static uint32_t get_u32(const string &buf, size_t at)
{
    const unsigned char *p = reinterpret_cast<const unsigned char *>(buf.data()) + at;
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

// Little endian on any machine: magic, version, count, then uint16 ids.
static bool write_tokens(const string &path, const vector<uint32_t> &ids)
{
    vector<unsigned char> buf(12 + 2 * ids.size());
    put_u32(buf, 0, BIN_MAGIC);
    put_u32(buf, 4, BIN_VERSION);
    put_u32(buf, 8, static_cast<uint32_t>(ids.size()));
    for (size_t k = 0; k < ids.size(); k++)
    {
        buf[12 + 2 * k] = static_cast<unsigned char>(ids[k]);
        buf[13 + 2 * k] = static_cast<unsigned char>(ids[k] >> 8);
    }
    return write_file(path, reinterpret_cast<const char *>(buf.data()), buf.size());
}

static bool read_tokens(const string &path, vector<uint32_t> &ids, string &error)
{
    string buf;
    if (!read_file(path, buf))
    {
        error = "cannot read " + path;
        return false;
    }
    if (buf.size() < 12 || get_u32(buf, 0) != BIN_MAGIC || get_u32(buf, 4) != BIN_VERSION)
    {
        error = path + " is not a BPE1 token file";
        return false;
    }
    uint32_t count = get_u32(buf, 8);
    if (buf.size() != 12 + 2 * static_cast<size_t>(count))
    {
        error = path + " is the wrong size for " + to_string(count) + " tokens";
        return false;
    }
    const unsigned char *p = reinterpret_cast<const unsigned char *>(buf.data()) + 12;
    ids.resize(count);
    for (size_t k = 0; k < count; k++)
        ids[k] = static_cast<uint32_t>(p[2 * k]) | (static_cast<uint32_t>(p[2 * k + 1]) << 8);
    return true;
}

// Tags are only found on lines that end in '\n'. A file saved with Windows
// line endings ("\r\n") would quietly lose all of them.
static void warn_crlf(const string &path, const string &text)
{
    if (text.find('\r') != string::npos)
    {
        fprintf(stderr, "warning: %s has '\\r' characters (Windows line endings?).\n"
                        "         Tags are only found when the line ends in '\\n'. Run dos2unix on it first.\n",
                path.c_str());
    }
}

static size_t count_chars(const string &text)
{
    size_t chars = 0;
    for (char c : text)
    {
        if ((static_cast<unsigned char>(c) & 0xC0) != 0x80)    // skip UTF-8 continuation bytes
            chars++;
    }
    return chars;
}

static int do_train(const string &in_path, const string &out_path, uint32_t vocab_size, bool verbose, int pretok,
                    const vector<string> &reserved)
{
    string text;
    if (!read_file(in_path, text))
    {
        fprintf(stderr, "error: cannot read %s\n", in_path.c_str());
        return 1;
    }
    warn_crlf(in_path, text);

    auto t0 = chrono::steady_clock::now();
    Tokenizer tok;
    train(tok, text, vocab_size, verbose, pretok, reserved);
    double secs = seconds_since(t0);

    if (tok.vocab.size() > 65536)
    {
        fprintf(stderr, "error: %zu merges + 256 bytes + %zu specials is over 65,536 ids\n",
                tok.merges.size(), tok.specials.size());
        return 1;
    }
    if (!save(tok, out_path))
    {
        fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
        return 1;
    }

    printf("merges %zu, specials %zu (", tok.merges.size(), tok.specials.size());
    for (size_t k = 0; k < tok.specials.size(); k++)
        printf("%s%s", k > 0 ? " " : "", tok.specials[k].c_str());
    printf("), vocab %zu, pretokenizer v%d, %.2f s -> %s\n", tok.vocab.size(), tok.pretok, secs, out_path.c_str());

    if (tok.merges.size() + 256 < vocab_size)
        printf("note: stopped early, no pairs left to merge\n");
    return 0;
}

static int do_encode(const string &model_path, const string &in_path, const string &out_path)
{
    Tokenizer tok;
    if (!load(tok, model_path))
    {
        fprintf(stderr, "error: cannot load tokenizer %s\n", model_path.c_str());
        return 1;
    }
    string text;
    if (!read_file(in_path, text))
    {
        fprintf(stderr, "error: cannot read %s\n", in_path.c_str());
        return 1;
    }
    warn_crlf(in_path, text);

    auto t0 = chrono::steady_clock::now();
    vector<uint32_t> ids = encode(tok, text);
    double secs = seconds_since(t0);

    if (ids.size() > 0xFFFFFFFFu)
    {
        fprintf(stderr, "error: too many tokens for a uint32 count\n");
        return 1;
    }
    if (!write_tokens(out_path, ids))
    {
        fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
        return 1;
    }

    size_t chars = count_chars(text);
    double per = ids.empty() ? 0.0 : static_cast<double>(chars) / static_cast<double>(ids.size());
    double bytes_per = ids.empty() ? 0.0 : static_cast<double>(text.size()) / static_cast<double>(ids.size());
    printf("tokens written %s, chars per token %.2f (bytes per token %.2f), %.2f s -> %s\n",
           with_commas(ids.size()).c_str(), per, bytes_per, secs, out_path.c_str());
    return 0;
}

static int do_decode(const string &model_path, const string &in_path, const string &out_path)
{
    Tokenizer tok;
    if (!load(tok, model_path))
    {
        fprintf(stderr, "error: cannot load tokenizer %s\n", model_path.c_str());
        return 1;
    }
    vector<uint32_t> ids;
    string error;
    if (!read_tokens(in_path, ids, error))
    {
        fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    for (size_t k = 0; k < ids.size(); k++)
    {
        if (ids[k] >= tok.vocab.size())
        {
            fprintf(stderr, "error: token %zu is id %u, but the tokenizer only has %zu ids\n",
                    k, ids[k], tok.vocab.size());
            return 1;
        }
    }

    string text = decode(tok, ids);
    if (!out_path.empty())
    {
        if (!write_file(out_path, text.data(), text.size()))
        {
            fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
            return 1;
        }
        return 0;
    }
    fwrite(text.data(), 1, text.size(), stdout);
    return 0;
}

int main(int argc, char *argv[])
{
    char mode = 0;
    string in_path;
    string out_path;
    string model_path;
    uint32_t vocab_size = 4096;
    bool verbose = false;
    int pretok = 2;
    vector<string> reserved;
    int opt = 0;

    while ((opt = getopt(argc, argv, "tedi:o:m:n:p:r:vh")) != -1)
    {
        switch (opt)
        {
        case 't':
        case 'e':
        case 'd':
            if (mode != 0 && mode != opt)
            {
                fprintf(stderr, "error: pick only one of -t, -e, -d\n");
                return 1;
            }
            mode = static_cast<char>(opt);
            break;
        case 'i':
            in_path = optarg;
            break;
        case 'o':
            out_path = optarg;
            break;
        case 'm':
            model_path = optarg;
            break;
        case 'n':
        {
            char *end = nullptr;
            unsigned long v = strtoul(optarg, &end, 10);
            if (*optarg == '\0' || *end != '\0' || v < 256 || v > 65536)
            {
                fprintf(stderr, "error: -n must be a number from 256 to 65536\n");
                return 1;
            }
            vocab_size = static_cast<uint32_t>(v);
            break;
        }
        case 'p':
            if (string(optarg) != "1" && string(optarg) != "2")
            {
                fprintf(stderr, "error: -p takes 1 or 2\n");
                return 1;
            }
            pretok = optarg[0] - '0';
            break;
        case 'r':
        {
            string list = optarg;
            size_t at = 0;
            while (at <= list.size())
            {
                size_t comma = list.find(',', at);
                string name = list.substr(at, comma == string::npos ? string::npos : comma - at);
                if (!is_tag_name(name))
                {
                    fprintf(stderr, "error: -r: \"%s\" is not a tag like <user>\n", name.c_str());
                    return 1;
                }
                if (find(reserved.begin(), reserved.end(), name) != reserved.end())
                {
                    fprintf(stderr, "error: -r: %s is there twice\n", name.c_str());
                    return 1;
                }
                reserved.push_back(name);
                at = comma == string::npos ? list.size() + 1 : comma + 1;
            }
            break;
        }
        case 'v':
            verbose = true;
            break;
        case 'h':
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if (mode == 0)
    {
        fprintf(stderr, "error: pick one of -t, -e, -d\n");
        usage(argv[0]);
        return 1;
    }
    if (in_path.empty())
    {
        fprintf(stderr, "error: -i is required\n");
        return 1;
    }
    if (mode != 'd' && out_path.empty())
    {
        fprintf(stderr, "error: -o is required\n");
        return 1;
    }
    if (mode != 't' && model_path.empty())
    {
        fprintf(stderr, "error: -m is required for -e and -d\n");
        return 1;
    }

    if (mode == 't')
        return do_train(in_path, out_path, vocab_size, verbose, pretok, reserved);
    if (mode == 'e')
        return do_encode(model_path, in_path, out_path);
    return do_decode(model_path, in_path, out_path);
}
