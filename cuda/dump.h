#pragma once
// dump.h: reads the GPTREF1 file written by reference/dump.py, and the error
// table every test prints.
//
//   char[8] "GPTREF1\0" | uint32 V, Vp, n_ctx, L, H, C, B, T
//   float32 lr, beta1, beta2, eps, weight_decay | uint32 count | records
//   record: uint32 name_len, name, uint32 dtype (0 f32, 1 i32), uint32 ndim,
//           uint32 dims[ndim], data (row-major)

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct Record
{
    std::string name;
    int dtype = 0;                          // 0 float32, 1 int32
    std::vector<int> dims;
    std::vector<float> f;                   // filled when dtype == 0
    std::vector<int32_t> i;                 // filled when dtype == 1
    size_t count() const;
    std::string shape() const;              // "(4, 64, 64)", "()" for a scalar
};

struct Dump
{
    uint32_t V, Vp, n_ctx, L, H, C, B, T;
    float lr, beta1, beta2, eps, weight_decay;
    std::vector<std::string> order;                     // record names in file order
    std::unordered_map<std::string, Record> records;
    const Record &at(const std::string &name) const;    // clear error if missing
    bool has(const std::string &name) const;
};

Dump read_dump(const std::string &path);               // exits with a message if bad

// Copy a record to the GPU (into an existing buffer of at least count() elements).
void upload(const Record &r, float *device);
void upload(const Record &r, int *device);

// Copy `count` floats back from the GPU.
std::vector<float> download(const float *device, size_t count);

// Starts a new table: prints the title and the column names. The first row
// over tolerance in each table is marked "<-- first mismatch", later ones "!".
void table_header(const std::string &title);

// Rows over tolerance since the program started (use it for the exit code).
int table_failures();

// Copy the record's count() floats back from the GPU, compare against record
// `name`, print one row: name, shape, max abs error, max rel error (relative
// to the largest finite |ref|), and a flag if over tol. Equal values count as
// 0 error, including -inf == -inf (scores holds -inf in its masked spots).
// NaN on either side is always a failure. Passes if max rel <= tol, or if
// max abs <= abs_tol. Returns true if it passed.
bool compare(const Dump &d, const std::string &name, const float *device,
             float tol, float abs_tol = 0.0f);

// Same row, against reference values you built on the host.
bool compare_values(const std::string &label, const std::vector<int> &dims, const float *ref,
                    size_t count, const float *device, float tol, float abs_tol = 0.0f);

// Same row, when both sides are already on the host.
bool compare_host(const std::string &label, const std::vector<int> &dims, const float *ref,
                  const float *got, size_t count, float tol, float abs_tol = 0.0f);

// A row that only says why something was not compared.
void table_note(const std::string &label, const std::string &why);
