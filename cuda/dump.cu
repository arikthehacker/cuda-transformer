// dump.cu: GPTREF1 reader, upload, and the compare table

#include "dump.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "common.h"

using namespace std;

static const char MAGIC[8] = {'G', 'P', 'T', 'R', 'E', 'F', '1', '\0'};

size_t Record::count() const
{
    size_t n = 1;
    for (int d : dims)
    {
        n *= static_cast<size_t>(d);
    }
    return n;
}

string Record::shape() const
{
    string s = "(";
    for (size_t k = 0; k < dims.size(); k++)
    {
        s += (k > 0 ? ", " : "") + to_string(dims[k]);
    }
    return s + ")";
}

const Record &Dump::at(const string &name) const
{
    auto it = records.find(name);
    if (it == records.end())
    {
        fprintf(stderr, "error: the dump has no record named \"%s\"\n", name.c_str());
        exit(1);
    }
    return it->second;
}

bool Dump::has(const string &name) const
{
    return records.count(name) > 0;
}

// Reads little endian values one after another, with a bounds check each time.
struct Reader
{
    const vector<char> &buf;
    const string &path;
    size_t pos = 0;

    void need(size_t n)
    {
        if (pos + n > buf.size())
        {
            fprintf(stderr, "error: %s ends early (wanted %zu bytes at offset %zu of %zu)\n",
                    path.c_str(), n, pos, buf.size());
            exit(1);
        }
    }

    uint32_t u32()
    {
        need(4);
        uint32_t v;
        memcpy(&v, buf.data() + pos, 4);
        pos += 4;
        return v;
    }

    float f32()
    {
        need(4);
        float v;
        memcpy(&v, buf.data() + pos, 4);
        pos += 4;
        return v;
    }

    void bytes(void *out, size_t n)
    {
        need(n);
        if (n > 0)
        {
            memcpy(out, buf.data() + pos, n);
        }
        pos += n;
    }
};

Dump read_dump(const string &path)
{
    ifstream in(path, ios::binary);
    if (!in)
    {
        fprintf(stderr, "error: cannot open %s (run: python -m reference.dump)\n", path.c_str());
        exit(1);
    }
    vector<char> buf((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    if (buf.size() < 8 || memcmp(buf.data(), MAGIC, 8) != 0)
    {
        fprintf(stderr, "error: %s is not a GPTREF1 file\n", path.c_str());
        exit(1);
    }

    Reader r{buf, path, 8};
    Dump d;
    d.V = r.u32();
    d.Vp = r.u32();
    d.n_ctx = r.u32();
    d.L = r.u32();
    d.H = r.u32();
    d.C = r.u32();
    d.B = r.u32();
    d.T = r.u32();
    d.lr = r.f32();
    d.beta1 = r.f32();
    d.beta2 = r.f32();
    d.eps = r.f32();
    d.weight_decay = r.f32();

    uint32_t count = r.u32();
    for (uint32_t k = 0; k < count; k++)
    {
        uint32_t name_len = r.u32();
        string name(name_len, ' ');
        r.bytes(&name[0], name_len);
        Record &rec = d.records[name];      // filled in place, no copy
        rec.name = name;
        rec.dims.clear();
        rec.dtype = static_cast<int>(r.u32());
        uint32_t ndim = r.u32();
        for (uint32_t j = 0; j < ndim; j++)
        {
            rec.dims.push_back(static_cast<int>(r.u32()));
        }
        size_t n = rec.count();
        if (rec.dtype == 0)
        {
            rec.f.resize(n);
            r.bytes(rec.f.data(), n * 4);
        }
        else if (rec.dtype == 1)
        {
            rec.i.resize(n);
            r.bytes(rec.i.data(), n * 4);
        }
        else
        {
            fprintf(stderr, "error: %s: record %s has unknown dtype %d\n", path.c_str(), rec.name.c_str(), rec.dtype);
            exit(1);
        }
        d.order.push_back(name);
    }
    if (r.pos != buf.size())
    {
        fprintf(stderr, "error: %s has %zu bytes left after %u records\n", path.c_str(), buf.size() - r.pos, count);
        exit(1);
    }
    return d;
}

void upload(const Record &r, float *device)
{
    if (r.dtype != 0)
    {
        fprintf(stderr, "error: record %s is not float32\n", r.name.c_str());
        exit(1);
    }
    CUDA_CHECK(cudaMemcpy(device, r.f.data(), r.count() * sizeof(float), cudaMemcpyHostToDevice));
}

void upload(const Record &r, int *device)
{
    if (r.dtype != 1)
    {
        fprintf(stderr, "error: record %s is not int32\n", r.name.c_str());
        exit(1);
    }
    CUDA_CHECK(cudaMemcpy(device, r.i.data(), r.count() * sizeof(int), cudaMemcpyHostToDevice));
}

vector<float> download(const float *device, size_t count)
{
    vector<float> host(count);
    if (count > 0)
    {
        CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return host;
}

// ------------------------------------------------------------------ the table

static int g_total_failures = 0;
static int g_table_failures = 0;

void table_header(const string &title)
{
    g_table_failures = 0;
    printf("\n%s\n", title.c_str());
    printf("  %-38s %-20s %11s %11s\n", "name", "shape", "max abs", "max rel");
    fflush(stdout);
}

int table_failures()
{
    return g_total_failures;
}

void table_note(const string &label, const string &why)
{
    printf("  %-38s %s\n", label.c_str(), why.c_str());
    fflush(stdout);
}

static string shape_text(const vector<int> &dims)
{
    string s = "(";
    for (size_t k = 0; k < dims.size(); k++)
    {
        s += (k > 0 ? ", " : "") + to_string(dims[k]);
    }
    return s + ")";
}

bool compare_values(const string &label, const vector<int> &dims, const float *ref, size_t count,
                    const float *device, float tol, float abs_tol)
{
    vector<float> got = download(device, count);
    return compare_host(label, dims, ref, got.data(), count, tol, abs_tol);
}

bool compare_host(const string &label, const vector<int> &dims, const float *ref, const float *got,
                  size_t count, float tol, float abs_tol)
{
    double max_abs = 0.0;
    double scale = 0.0;
    size_t nans = 0;
    for (size_t k = 0; k < count; k++)
    {
        double a = ref[k];
        double b = got[k];
        if (std::isnan(a) || std::isnan(b))
        {
            nans++;
        }
        else if (a != b)                    // equal values, -inf == -inf included, add nothing
        {
            max_abs = max(max_abs, fabs(a - b));
        }
        if (std::isfinite(a))
        {
            scale = max(scale, fabs(a));
        }
    }
    double max_rel = scale > 0.0 ? max_abs / scale : (max_abs == 0.0 ? 0.0 : INFINITY);
    bool ok = nans == 0 && (max_rel <= tol || max_abs <= abs_tol);

    string flag;
    if (!ok)
    {
        flag = g_table_failures == 0 ? "   <-- first mismatch" : "   !";
        g_table_failures++;
        g_total_failures++;
    }
    if (nans > 0)
    {
        flag += "  (" + to_string(nans) + " NaN)";
    }
    printf("  %-38s %-20s %11.3e %11.3e%s\n", label.c_str(), shape_text(dims).c_str(), max_abs, max_rel, flag.c_str());
    fflush(stdout);
    return ok;
}

bool compare(const Dump &d, const string &name, const float *device, float tol, float abs_tol)
{
    const Record &r = d.at(name);
    if (r.dtype != 0)
    {
        fprintf(stderr, "error: compare() on int record %s\n", name.c_str());
        exit(1);
    }
    return compare_values(name, r.dims, r.f.data(), r.count(), device, tol, abs_tol);
}
