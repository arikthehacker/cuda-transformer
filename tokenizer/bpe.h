#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Token ids:
//   0 .. 255                     one id per byte value
//   256 .. 256 + merges - 1      one id per learned merge, in the order learned
//   after that                   one id per special tag, in the order found
// Total must stay under 65,536 so every id fits in a uint16_t.

struct Tokenizer
{
    std::vector<std::pair<uint32_t, uint32_t>> merges;  // merges[i] makes id 256 + i
    std::vector<std::string> vocab;                     // vocab[id] = bytes for that id
    std::unordered_map<uint64_t, uint32_t> merge_id;    // pair_key(a, b) -> new id
    std::vector<std::string> specials;                  // "<bible>", "<austen>", ...
};

inline uint64_t pair_key(uint32_t a, uint32_t b)
{
    return (static_cast<uint64_t>(a) << 32) | b;
}

// A piece of input: plain text, or one special tag.
struct Segment
{
    std::string text;
    int special;    // -1 for plain text, else index into Tokenizer::specials
};

// Split text into plain text and tags. A tag is "<name>" alone on a line.
// New tag names get added to tok.specials when add_new is true (training).
std::vector<Segment> split_specials(Tokenizer &tok, const std::string &text, bool add_new);

// Split plain text into words so merges never cross a word boundary:
//   a run of '\n'                              -> one piece
//   a run of spaces not followed by a word     -> one piece
//   optional ONE leading space, then a run of:
//       letters (a-z, A-Z, or any byte >= 0x80, so UTF-8 stays together)
//       or digits
//       or anything else that is not space or '\n' (punctuation), including
//       the General Punctuation block U+2000 to U+206F (curly quotes, dashes)
// Example: "Let there be light," -> "Let" " there" " be" " light" ","
std::vector<std::string> pretokenize(const std::string &text);

// Learn merges until there are vocab_size ids (bytes + merges, specials not counted).
// Rules both your C++ and any reference version must follow exactly:
//   count every adjacent pair, overlaps included ("aaa" has 2 of (a,a))
//   pick the pair with the highest count
//   tie: pick the smaller pair_key(a, b)
//   merge left to right, without overlap ("aaa" -> [aa] a)
void train(Tokenizer &tok, const std::string &text, uint32_t vocab_size, bool verbose);

// Encode: for each word, repeatedly find the pair with the LOWEST merge id
// present in it and merge it everywhere left to right, until no pair has a merge.
std::vector<uint32_t> encode(Tokenizer &tok, const std::string &text);

std::string decode(const Tokenizer &tok, const std::vector<uint32_t> &ids);

bool save(const Tokenizer &tok, const std::string &path);
bool load(Tokenizer &tok, const std::string &path);
