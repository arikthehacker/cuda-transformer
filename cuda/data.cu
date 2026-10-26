// data.cu: token files and number printing (data.h)

#include "data.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace std;

static uint32_t read_u32(ifstream &in)
{
    unsigned char b[4] = {};
    in.read(reinterpret_cast<char *>(b), 4);
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) | (static_cast<uint32_t>(b[2]) << 16) |
           (static_cast<uint32_t>(b[3]) << 24);
}

static size_t file_size(ifstream &in)
{
    in.seekg(0, ios::end);
    size_t n = static_cast<size_t>(in.tellg());
    in.seekg(0, ios::beg);
    return n;
}

// The ids are read straight into the array (x86 and the GPU are little
// endian, like the file), so 1B tokens need 2 GB of host memory, not 6.
vector<uint16_t> load_tokens16(const string &path, int V)
{
    ifstream in(path, ios::binary);
    if (!in)
    {
        fprintf(stderr, "error: cannot open %s (run: make run)\n", path.c_str());
        exit(1);
    }
    size_t bytes = file_size(in);
    uint32_t magic = read_u32(in), version = read_u32(in);
    size_t count = read_u32(in);
    if (bytes < 12 || !in || magic != 0x31455042u || version != 1)
    {
        fprintf(stderr, "error: %s is not a BPE1 token file\n", path.c_str());
        exit(1);
    }
    if (bytes != 12 + 2 * count)
    {
        fprintf(stderr, "error: %s is the wrong size for %zu tokens\n", path.c_str(), count);
        exit(1);
    }
    vector<uint16_t> ids(count);
    in.read(reinterpret_cast<char *>(ids.data()), static_cast<streamsize>(2 * count));
    if (!in)
    {
        fprintf(stderr, "error: could not read the %zu ids of %s\n", count, path.c_str());
        exit(1);
    }
    for (size_t k = 0; k < count; k++)
    {
        if (ids[k] >= V)
        {
            fprintf(stderr, "error: %s has id %d at position %zu, the model only has %d ids\n", path.c_str(),
                    static_cast<int>(ids[k]), k, V);
            exit(1);
        }
    }
    return ids;
}

vector<int> load_tokens(const string &path, int V)
{
    vector<uint16_t> ids = load_tokens16(path, V);
    return vector<int>(ids.begin(), ids.end());
}

Targets load_targets(const string &path, size_t n_tokens, int n_ctx)
{
    ifstream in(path, ios::binary);
    if (!in)
    {
        fprintf(stderr, "error: cannot open %s (run: python.exe -m reference.chat --tokens ...)\n", path.c_str());
        exit(1);
    }
    size_t bytes = file_size(in);
    uint32_t magic = read_u32(in), version = read_u32(in);
    size_t count = read_u32(in), n_starts = read_u32(in);
    uint32_t ctx = read_u32(in);
    if (bytes < 20 || !in || magic != 0x31544754u || version != 1)
    {
        fprintf(stderr, "error: %s is not a TGT1 targets file\n", path.c_str());
        exit(1);
    }
    if (bytes != 20 + 4 * count + 4 * n_starts)
    {
        fprintf(stderr, "error: %s is the wrong size for %zu targets and %zu starts\n", path.c_str(), count, n_starts);
        exit(1);
    }
    if (count != n_tokens)
    {
        fprintf(stderr, "error: %s has %zu targets, the token file has %zu tokens\n", path.c_str(), count, n_tokens);
        exit(1);
    }
    if (static_cast<int>(ctx) != n_ctx)
    {
        fprintf(stderr, "error: %s was cut for n_ctx %u, the model has n_ctx %d (re-run reference.chat --n-ctx %d)\n",
                path.c_str(), ctx, n_ctx, n_ctx);
        exit(1);
    }
    Targets t;
    t.n_ctx = n_ctx;
    t.y.resize(count);
    t.starts.resize(n_starts);
    in.read(reinterpret_cast<char *>(t.y.data()), static_cast<streamsize>(4 * count));
    in.read(reinterpret_cast<char *>(t.starts.data()), static_cast<streamsize>(4 * n_starts));
    if (!in)
    {
        fprintf(stderr, "error: could not read %s\n", path.c_str());
        exit(1);
    }
    for (uint32_t st : t.starts)
    {
        if (static_cast<size_t>(st) + n_ctx > n_tokens)
        {
            fprintf(stderr, "error: %s has a start at %u, too close to the end of %zu tokens\n", path.c_str(), st,
                    n_tokens);
            exit(1);
        }
    }
    if (t.starts.empty())
    {
        fprintf(stderr, "error: %s has no conversation starts\n", path.c_str());
        exit(1);
    }
    return t;
}

vector<long long> val_starts(size_t n_tokens, int T, int count)
{
    long long last = static_cast<long long>(n_tokens) - T - 1;
    vector<long long> starts(count);
    for (int i = 0; i < count; i++)
    {
        starts[i] = count == 1 ? 0 : (i * last) / (count - 1);
    }
    return starts;
}

vector<long long> val_starts_from(const vector<uint32_t> &starts, int count)
{
    long long last = static_cast<long long>(starts.size()) - 1;
    vector<long long> out(count);
    for (int i = 0; i < count; i++)
    {
        out[i] = starts[count == 1 ? 0 : (i * last) / (count - 1)];
    }
    return out;
}

string with_commas(long long v)
{
    string digits = to_string(v < 0 ? -v : v);
    string out = v < 0 ? "-" : "";
    for (size_t k = 0; k < digits.size(); k++)
    {
        if (k > 0 && (digits.size() - k) % 3 == 0)
        {
            out += ',';
        }
        out += digits[k];
    }
    return out;
}
