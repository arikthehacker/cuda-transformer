// bpe.cpp: byte level BPE tokenizer
//
// pretokenize  split text into words (merges never cross a word)
// train        learn merges with incremental pair counts + a lazy max heap
// encode       apply merges to each unique word once, then reuse it (cache)
// decode       join the bytes of each id
// save / load  the plain text "bpe v1" format

#include "bpe.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <queue>
#include <string_view>

using namespace std;


// ============================================================== byte classes

enum ByteClass : uint8_t
{
    NEWLINE,
    SPACE,
    LETTER,     // a-z, A-Z, or any byte >= 0x80 (keeps UTF-8 letters together),
                // except the General Punctuation block, see unit_at()
    DIGIT,
    OTHER       // punctuation and everything else (tabs and '\r' land here too)
};

static array<uint8_t, 256> make_class_table()
{
    array<uint8_t, 256> table{};
    for (int c = 0; c < 256; c++)
    {
        if (c == '\n')
            table[c] = NEWLINE;
        else if (c == ' ')
            table[c] = SPACE;
        else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80)
            table[c] = LETTER;
        else if (c >= '0' && c <= '9')
            table[c] = DIGIT;
        else
            table[c] = OTHER;
    }
    return table;
}

static const array<uint8_t, 256> CLASS = make_class_table();

static inline uint8_t class_of(char c)
{
    return CLASS[static_cast<unsigned char>(c)];
}


// ============================================================== pretokenize

// The General Punctuation block, U+2000 to U+206F, is E2 80 80 to E2 81 AF in
// UTF-8. It holds curly quotes, dashes and the ellipsis. Counting those as
// punctuation makes “Let and God’s split the same way "Let and God's do.
// Returns 3 if a complete character from that block starts at s[i], else 0.
static inline size_t general_punct_len(string_view s, size_t i)
{
    if (i + 2 < s.size() && static_cast<unsigned char>(s[i]) == 0xE2)
    {
        unsigned char b1 = static_cast<unsigned char>(s[i + 1]);
        unsigned char b2 = static_cast<unsigned char>(s[i + 2]);
        if ((b2 & 0xC0) == 0x80 && (b1 == 0x80 || (b1 == 0x81 && b2 <= 0xAF)))
            return 3;
    }
    return 0;
}

// Class of the character at s[i], and how many bytes it covers (len).
static inline uint8_t unit_at(string_view s, size_t i, size_t &len)
{
    len = general_punct_len(s, i);
    if (len > 0)
        return OTHER;
    len = 1;
    return class_of(s[i]);
}

// Returns where the piece that starts at s[i] ends (one past its last byte).
// This is the whole pretokenizer. pretokenize(), train() and encode() all use
// it, so they always agree on the word boundaries.
static size_t piece_end(string_view s, size_t i)
{
    size_t n = s.size();
    size_t len = 0;
    uint8_t c = unit_at(s, i, len);
    size_t j = i + len;

    if (c == SPACE)
    {
        for (; j < n && s[j] == ' '; j++) {}

        bool word_follows = (j < n && s[j] != '\n');
        if (!word_follows)
            return j;           // spaces before '\n' or at the end: one piece
        if (j - i > 1)
            return j - 1;       // leave the last space for the word after it

        c = unit_at(s, j, len); // exactly one space: it leads the word
        j += len;
    }

    // a run of '\n', or a run of one word class
    while (j < n && unit_at(s, j, len) == c)
        j += len;
    return j;
}

vector<string> pretokenize(const string &text)
{
    vector<string> pieces;
    string_view s(text);
    for (size_t i = 0; i < s.size(); )
    {
        size_t j = piece_end(s, i);
        pieces.emplace_back(text, i, j - i);
        i = j;
    }
    return pieces;
}


// ============================================================== special tags

// "<" + one or more of [a-z_] + ">"
static bool is_tag(string_view line)
{
    if (line.size() < 3 || line.front() != '<' || line.back() != '>')
        return false;
    for (size_t k = 1; k + 1 < line.size(); k++)
    {
        char c = line[k];
        if (!((c >= 'a' && c <= 'z') || c == '_'))
            return false;
    }
    return true;
}

static int find_special(Tokenizer &tok, string_view name, bool add_new)
{
    for (size_t k = 0; k < tok.specials.size(); k++)
    {
        if (tok.specials[k] == name)
            return static_cast<int>(k);
    }
    if (!add_new)
        return -1;
    tok.specials.emplace_back(name);
    return static_cast<int>(tok.specials.size()) - 1;
}

// Walks the text line by line (memchr jumps straight to each '\n').
// The '\n' after a tag stays in the plain text, so decode gives back the
// exact input.
vector<Segment> split_specials(Tokenizer &tok, const string &text, bool add_new)
{
    vector<Segment> segs;
    size_t n = text.size();
    size_t plain_start = 0;

    for (size_t line = 0; line < n; )
    {
        const void *nl = memchr(text.data() + line, '\n', n - line);
        size_t line_end = nl ? static_cast<size_t>(static_cast<const char *>(nl) - text.data()) : n;

        if (text[line] == '<')
        {
            string_view s(text.data() + line, line_end - line);
            int idx = is_tag(s) ? find_special(tok, s, add_new) : -1;
            if (idx >= 0)
            {
                if (line > plain_start)
                    segs.push_back({text.substr(plain_start, line - plain_start), -1});
                segs.push_back({string(s), idx});
                plain_start = line_end;
            }
        }
        line = line_end + 1;
    }

    if (plain_start < n)
        segs.push_back({text.substr(plain_start), -1});
    return segs;
}


// ============================================================== training

namespace
{

struct Word
{
    vector<uint32_t> ids;
    uint64_t count;
};

struct HeapItem
{
    int64_t count;
    uint64_t key;
};

// priority_queue keeps the "largest" item on top.
// Larger count wins. On a tie the smaller pair_key wins.
struct HeapLess
{
    bool operator()(const HeapItem &x, const HeapItem &y) const
    {
        if (x.count != y.count)
            return x.count < y.count;
        return x.key > y.key;
    }
};

using Heap = priority_queue<HeapItem, vector<HeapItem>, HeapLess>;

// counts:  pair_key -> total count over all words (weighted by word count)
// where:   pair_key -> indices of words that contain (or used to contain) it
// touched: keys whose count changed during the current merge
struct PairStats
{
    unordered_map<uint64_t, int64_t> counts;
    unordered_map<uint64_t, vector<uint32_t>> where;
    vector<uint64_t> touched;

    void remove(uint64_t key, int64_t n)
    {
        counts[key] -= n;
        touched.push_back(key);
    }

    void add(uint64_t key, int64_t n, uint32_t word)
    {
        counts[key] += n;
        touched.push_back(key);

        // A word is handled start to finish before the next one, so a
        // repeat of the same word always sits at the back of the list.
        vector<uint32_t> &list = where[key];
        if (list.empty() || list.back() != word)
            list.push_back(word);
    }
};

} // namespace

// Merge (a, b) -> c inside one word, left to right without overlap, and
// update only the pairs right next to each merge site.
//
// For a site x[i], x[i+1] = (a, b) with p on the left and q on the right:
//   (a, b) goes away
//   (p, a) goes away, (p, c) appears    unless p came from the site just before,
//                                       then (p, a) was already removed as that
//                                       site's right pair
//   (b, q) goes away, (c, q) appears    unless q starts the next site, then that
//                                       site adds (c, c) as its left pair
static void merge_word(Word &w, uint32_t a, uint32_t b, uint32_t c, uint32_t wi, PairStats &st)
{
    vector<uint32_t> &x = w.ids;
    int64_t n = static_cast<int64_t>(w.count);
    size_t len = x.size();
    size_t out = 0;
    bool last_was_site = false;

    for (size_t i = 0; i < len; )
    {
        bool site = (i + 1 < len && x[i] == a && x[i + 1] == b);
        if (site)
        {
            st.remove(pair_key(a, b), n);

            if (out > 0)
            {
                uint32_t p = x[out - 1];    // already rewritten, so it may be c
                if (!last_was_site)
                    st.remove(pair_key(p, a), n);
                st.add(pair_key(p, c), n, wi);
            }

            if (i + 2 < len)
            {
                uint32_t q = x[i + 2];
                st.remove(pair_key(b, q), n);
                bool next_is_site = (i + 3 < len && q == a && x[i + 3] == b);
                if (!next_is_site)
                    st.add(pair_key(c, q), n, wi);
            }

            x[out] = c;
            i += 2;
        }
        else
        {
            x[out] = x[i];
            i += 1;
        }
        out++;
        last_was_site = site;
    }
    x.resize(out);
}

// Pops stale entries until the top matches its real count (lazy delete).
static bool pop_best(Heap &heap, const unordered_map<uint64_t, int64_t> &counts, HeapItem &best)
{
    while (!heap.empty())
    {
        HeapItem top = heap.top();
        heap.pop();
        auto it = counts.find(top.key);
        if (it != counts.end() && it->second == top.count)
        {
            best = top;
            return true;
        }
    }
    return false;
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

static bool valid_utf8(const string &s)
{
    size_t n = s.size();
    for (size_t i = 0; i < n; )
    {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t extra = 0;               // continuation bytes after c
        if (c >= 0x80 && (c >> 5) == 0x6)
            extra = 1;
        else if (c >= 0x80 && (c >> 4) == 0xE)
            extra = 2;
        else if (c >= 0x80 && (c >> 3) == 0x1E)
            extra = 3;
        else if (c >= 0x80)
            return false;               // a stray continuation byte
        if (i + extra >= n)
            return false;               // cut off at the end
        for (size_t k = 1; k <= extra; k++)
        {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80)
                return false;
        }
        i += extra + 1;
    }
    return true;
}

// Quoted token for the -v output: \n, \t, \" escaped, bytes shown as \xNN
// when the token is not a complete UTF-8 string.
static string show_token(const string &bytes)
{
    bool utf8 = valid_utf8(bytes);
    string out = "\"";
    for (char ch : bytes)
    {
        unsigned char u = static_cast<unsigned char>(ch);
        char hex[8];
        if (ch == '\n')
            out += "\\n";
        else if (ch == '\t')
            out += "\\t";
        else if (ch == '"' || ch == '\\')
        {
            out += '\\';
            out += ch;
        }
        else if (u < 0x20 || u == 0x7F || (u >= 0x80 && !utf8))
        {
            snprintf(hex, sizeof(hex), "\\x%02x", u);
            out += hex;
        }
        else
            out += ch;
    }
    out += "\"";
    return out;
}

void train(Tokenizer &tok, const string &text, uint32_t vocab_size, bool verbose)
{
    auto t0 = chrono::steady_clock::now();

    tok = Tokenizer();
    for (int b = 0; b < 256; b++)
        tok.vocab.push_back(string(1, static_cast<char>(b)));

    // 1. Count unique words. The keys are views into segs, which lives until
    //    the end of this function, so no word gets copied.
    vector<Segment> segs = split_specials(tok, text, true);
    unordered_map<string_view, uint64_t> word_counts;
    word_counts.reserve(1 << 16);
    uint64_t pieces = 0;
    for (const Segment &seg : segs)
    {
        if (seg.special < 0)
        {
            string_view s(seg.text);
            for (size_t i = 0; i < s.size(); )
            {
                size_t j = piece_end(s, i);
                if (j - i >= 2)                     // 1 byte words have no pairs
                    word_counts[s.substr(i, j - i)]++;
                pieces++;
                i = j;
            }
        }
    }

    vector<Word> words;
    words.reserve(word_counts.size());
    for (const auto &wc : word_counts)
    {
        const unsigned char *p = reinterpret_cast<const unsigned char *>(wc.first.data());
        Word w;
        w.ids.assign(p, p + wc.first.size());
        w.count = wc.second;
        words.push_back(move(w));
    }
    word_counts = unordered_map<string_view, uint64_t>();   // free it

    // 2. Count every adjacent pair once per occurrence, weighted by word count.
    PairStats st;
    st.counts.reserve(1 << 16);
    st.where.reserve(1 << 16);
    for (uint32_t wi = 0; wi < words.size(); wi++)
    {
        const vector<uint32_t> &ids = words[wi].ids;
        for (size_t k = 0; k + 1 < ids.size(); k++)
            st.add(pair_key(ids[k], ids[k + 1]), static_cast<int64_t>(words[wi].count), wi);
    }
    st.touched.clear();

    vector<HeapItem> items;
    items.reserve(st.counts.size());
    for (const auto &kv : st.counts)
        items.push_back({kv.second, kv.first});
    Heap heap(HeapLess(), move(items));

    double count_secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    if (verbose)
    {
        printf("words  %s unique of %s pieces, %s pairs, %.2f s\n",
               with_commas(words.size()).c_str(), with_commas(pieces).c_str(),
               with_commas(st.counts.size()).c_str(), count_secs);
        fflush(stdout);
    }

    // 3. Merge loop.
    uint32_t num_merges = vocab_size > 256 ? vocab_size - 256 : 0;
    tok.merges.reserve(num_merges);
    tok.merge_id.reserve(num_merges);
    HeapItem best = {0, 0};

    while (tok.merges.size() < num_merges && pop_best(heap, st.counts, best))
    {
        uint32_t a = static_cast<uint32_t>(best.key >> 32);
        uint32_t b = static_cast<uint32_t>(best.key & 0xFFFFFFFFu);
        uint32_t c = 256 + static_cast<uint32_t>(tok.merges.size());

        // Take the word list out of the map. Words that no longer hold the
        // pair are fine: merge_word finds no site and changes nothing.
        vector<uint32_t> list = move(st.where[best.key]);
        st.where.erase(best.key);

        st.touched.clear();
        for (uint32_t wi : list)
            merge_word(words[wi], a, b, c, wi, st);

        // Push the new count of every pair that changed. A pair at 0 can never
        // come back (new pairs always contain the newest id), so drop it.
        sort(st.touched.begin(), st.touched.end());
        st.touched.erase(unique(st.touched.begin(), st.touched.end()), st.touched.end());
        for (uint64_t key : st.touched)
        {
            auto it = st.counts.find(key);
            if (it->second <= 0)
            {
                st.counts.erase(it);
                st.where.erase(key);
            }
            else
                heap.push({it->second, key});
        }

        tok.merges.push_back({a, b});
        tok.merge_id[best.key] = c;
        tok.vocab.push_back(tok.vocab[a] + tok.vocab[b]);

        bool last = (tok.merges.size() == num_merges);
        if (verbose && (c % 256 == 0 || last))
        {
            string pair_text = "(" + to_string(a) + ", " + to_string(b) + ")";
            printf("merge %5u / %u   %-12s -> %-5u   %-14s count %s\n",
                   c, vocab_size, pair_text.c_str(), c, show_token(tok.vocab[c]).c_str(),
                   with_commas(static_cast<uint64_t>(best.count)).c_str());
            fflush(stdout);
        }
    }

    for (const string &s : tok.specials)
        tok.vocab.push_back(s);
}


// ============================================================== encode / decode

static const uint32_t NO_MERGE = 0xFFFFFFFFu;

static inline uint32_t merge_of(const Tokenizer &tok, uint32_t a, uint32_t b)
{
    auto it = tok.merge_id.find(pair_key(a, b));
    return it == tok.merge_id.end() ? NO_MERGE : it->second;
}

// Index of the leftmost lowest rank, and that rank (NO_MERGE if none).
static inline uint32_t lowest(const vector<uint32_t> &rank, size_t &at)
{
    uint32_t best = NO_MERGE;
    for (size_t k = 0; k < rank.size(); k++)
    {
        if (rank[k] < best)
        {
            best = rank[k];
            at = k;
        }
    }
    return best;
}

// Words longer than this use the heap version below. Short words are faster
// with a plain array scan, long ones would be O(n^2) with it.
static const size_t LONG_WORD = 128;

// Same rule as encode_word, in O(n log n) for very long words: a linked list
// over byte positions plus a min heap of (rank << 32 | position). Removing a
// token never reorders the others, so the smallest position is still the
// leftmost pair. rank[i] == NO_MERGE marks a removed position.
static void encode_long_word(const Tokenizer &tok, string_view word, vector<uint32_t> &ids)
{
    size_t n = word.size();
    vector<uint32_t> token(n);
    vector<uint32_t> rank(n, NO_MERGE);
    vector<int32_t> prev(n);
    vector<int32_t> next(n);
    priority_queue<uint64_t, vector<uint64_t>, greater<uint64_t>> heap;

    for (size_t k = 0; k < n; k++)
    {
        token[k] = static_cast<unsigned char>(word[k]);
        prev[k] = static_cast<int32_t>(k) - 1;
        next[k] = (k + 1 < n) ? static_cast<int32_t>(k + 1) : -1;
    }
    for (size_t k = 0; k + 1 < n; k++)
    {
        rank[k] = merge_of(tok, token[k], token[k + 1]);
        if (rank[k] != NO_MERGE)
            heap.push((static_cast<uint64_t>(rank[k]) << 32) | k);
    }

    while (!heap.empty())
    {
        uint32_t r = static_cast<uint32_t>(heap.top() >> 32);
        int32_t i = static_cast<int32_t>(heap.top() & 0xFFFFFFFFu);
        heap.pop();
        if (rank[i] == r)                   // skip stale entries
        {
            int32_t j = next[i];            // merge token j into token i
            token[i] = r;
            next[i] = next[j];
            if (next[j] >= 0)
                prev[next[j]] = i;
            rank[j] = NO_MERGE;

            int32_t p = prev[i];
            if (p >= 0)
            {
                rank[p] = merge_of(tok, token[p], token[i]);
                if (rank[p] != NO_MERGE)
                    heap.push((static_cast<uint64_t>(rank[p]) << 32) | static_cast<uint32_t>(p));
            }
            rank[i] = (next[i] >= 0) ? merge_of(tok, token[i], token[next[i]]) : NO_MERGE;
            if (rank[i] != NO_MERGE)
                heap.push((static_cast<uint64_t>(rank[i]) << 32) | static_cast<uint32_t>(i));
        }
    }

    ids.clear();
    for (int32_t k = 0; k >= 0; k = next[k])
        ids.push_back(token[k]);
}

// rank[k] is the merge id of (ids[k], ids[k+1]). Each step merges the leftmost
// pair with the lowest id and looks up only the two new neighbor pairs.
// Taking the leftmost one each time is the same as "merge it everywhere left
// to right": a merge only creates pairs with higher ids, so every copy of the
// lowest pair is merged before anything else, in left to right order.
static void encode_word(const Tokenizer &tok, string_view word, vector<uint32_t> &ids, vector<uint32_t> &rank)
{
    size_t n = word.size();
    if (n > LONG_WORD)
    {
        encode_long_word(tok, word, ids);
        return;
    }
    ids.resize(n);
    for (size_t k = 0; k < n; k++)
        ids[k] = static_cast<unsigned char>(word[k]);

    rank.resize(n - 1);
    for (size_t k = 0; k + 1 < n; k++)
        rank[k] = merge_of(tok, ids[k], ids[k + 1]);

    size_t k = 0;
    uint32_t best = lowest(rank, k);
    while (best != NO_MERGE)
    {
        ids[k] = best;
        ids.erase(ids.begin() + k + 1);
        rank.erase(rank.begin() + k);
        if (k > 0)
            rank[k - 1] = merge_of(tok, ids[k - 1], ids[k]);
        if (k < rank.size())
            rank[k] = merge_of(tok, ids[k], ids[k + 1]);
        best = lowest(rank, k);
    }
}

vector<uint32_t> encode(Tokenizer &tok, const string &text)
{
    vector<uint32_t> out;
    out.reserve(text.size() / 3 + 16);

    vector<Segment> segs = split_specials(tok, text, false);
    uint32_t first_special = 256 + static_cast<uint32_t>(tok.merges.size());

    // word -> (offset, length) of its tokens in pool. Keys are views into segs.
    unordered_map<string_view, pair<size_t, uint32_t>> cache;
    cache.reserve(1 << 15);
    vector<uint32_t> pool;
    vector<uint32_t> ids;
    vector<uint32_t> rank;

    for (const Segment &seg : segs)
    {
        if (seg.special >= 0)
            out.push_back(first_special + static_cast<uint32_t>(seg.special));
        else
        {
            string_view s(seg.text);
            for (size_t i = 0; i < s.size(); )
            {
                size_t j = piece_end(s, i);
                if (j - i == 1)
                    out.push_back(static_cast<unsigned char>(s[i]));    // one byte, nothing to merge
                else
                {
                    string_view word = s.substr(i, j - i);
                    auto it = cache.find(word);
                    if (it == cache.end())
                    {
                        encode_word(tok, word, ids, rank);
                        it = cache.emplace(word, make_pair(pool.size(), static_cast<uint32_t>(ids.size()))).first;
                        pool.insert(pool.end(), ids.begin(), ids.end());
                    }
                    const uint32_t *p = pool.data() + it->second.first;
                    out.insert(out.end(), p, p + it->second.second);
                }
                i = j;
            }
        }
    }
    return out;
}

// Ids outside the vocab are skipped (the bpe tool checks for them first).
string decode(const Tokenizer &tok, const vector<uint32_t> &ids)
{
    size_t total = 0;
    for (uint32_t id : ids)
    {
        if (id < tok.vocab.size())
            total += tok.vocab[id].size();
    }

    string out;
    out.reserve(total);
    for (uint32_t id : ids)
    {
        if (id < tok.vocab.size())
            out += tok.vocab[id];
    }
    return out;
}


// ============================================================== save / load

// Binary mode, so Windows does not turn "\n" into "\r\n" and the same
// tokenizer gives the same file on every system.
bool save(const Tokenizer &tok, const string &path)
{
    ofstream out(path, ios::binary);
    if (!out)
        return false;

    out << "bpe v1\n";
    out << "merges " << tok.merges.size() << "\n";
    for (const auto &m : tok.merges)
        out << m.first << " " << m.second << "\n";
    out << "specials " << tok.specials.size() << "\n";
    for (const string &s : tok.specials)
        out << s << "\n";

    out.close();
    return !out.fail();
}

// Builds into a fresh Tokenizer and only replaces tok if the whole file is good.
bool load(Tokenizer &tok, const string &path)
{
    ifstream in(path);
    if (!in)
        return false;

    string magic;
    string version;
    string label;
    size_t count = 0;
    if (!(in >> magic >> version) || magic != "bpe" || version != "v1")
        return false;
    if (!(in >> label >> count) || label != "merges" || count > 65536 - 256)
        return false;

    Tokenizer t;
    for (int b = 0; b < 256; b++)
        t.vocab.push_back(string(1, static_cast<char>(b)));

    for (size_t i = 0; i < count; i++)
    {
        uint32_t a = 0;
        uint32_t b = 0;
        uint32_t id = 256 + static_cast<uint32_t>(i);
        if (!(in >> a >> b) || a >= id || b >= id)      // may only use earlier ids
            return false;
        if (t.merge_id.count(pair_key(a, b)) > 0)       // the same merge twice
            return false;
        t.merges.push_back({a, b});
        t.merge_id[pair_key(a, b)] = id;
        t.vocab.push_back(t.vocab[a] + t.vocab[b]);
    }

    if (!(in >> label >> count) || label != "specials" || count > 65536)
        return false;
    string line;
    getline(in, line);                                  // rest of the "specials" line
    for (size_t i = 0; i < count; i++)
    {
        if (!getline(in, line))
            return false;
        if (!line.empty() && line.back() == '\r')       // file edited on Windows
            line.pop_back();
        if (!is_tag(line))
            return false;
        if (find(t.specials.begin(), t.specials.end(), line) != t.specials.end())
            return false;                               // the same tag twice
        t.specials.push_back(line);
        t.vocab.push_back(line);
    }

    if (t.vocab.size() > 65536)
        return false;

    tok = move(t);
    return true;
}
