// data.cu: token files and number printing (data.h)

#include "data.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>

using namespace std;

vector<int> load_tokens(const string &path, int V)
{
    ifstream in(path, ios::binary);
    if (!in)
    {
        fprintf(stderr, "error: cannot open %s (run: make run)\n", path.c_str());
        exit(1);
    }
    vector<unsigned char> buf((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    auto u32 = [&](size_t at)
    {
        return static_cast<uint32_t>(buf[at]) | (static_cast<uint32_t>(buf[at + 1]) << 8) |
               (static_cast<uint32_t>(buf[at + 2]) << 16) | (static_cast<uint32_t>(buf[at + 3]) << 24);
    };
    if (buf.size() < 12 || u32(0) != 0x31455042u || u32(4) != 1)
    {
        fprintf(stderr, "error: %s is not a BPE1 token file\n", path.c_str());
        exit(1);
    }
    size_t count = u32(8);
    if (buf.size() != 12 + 2 * count)
    {
        fprintf(stderr, "error: %s is the wrong size for %zu tokens\n", path.c_str(), count);
        exit(1);
    }
    vector<int> ids(count);
    for (size_t k = 0; k < count; k++)
    {
        ids[k] = buf[12 + 2 * k] | (buf[13 + 2 * k] << 8);
        if (ids[k] >= V)
        {
            fprintf(stderr, "error: %s has id %d at position %zu, the model only has %d ids\n",
                    path.c_str(), ids[k], k, V);
            exit(1);
        }
    }
    return ids;
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
