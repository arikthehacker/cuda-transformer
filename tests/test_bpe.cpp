// test_bpe.cpp: checks for the BPE tokenizer. Run from the repo root: make test

#include "bpe.h"

#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

static int failures = 0;

static void check(bool ok, const string &what)
{
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok)
        failures++;
}

static bool file_exists(const string &path)
{
    ifstream in(path);
    return static_cast<bool>(in);
}

static string read_all(const string &path)
{
    ifstream in(path, ios::binary);
    stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static uint32_t special_id(const Tokenizer &tok, size_t index)
{
    return 256 + static_cast<uint32_t>(tok.merges.size() + index);
}

// Some text with the same flavor as the corpus, used to train small tokenizers.
static string sample_text()
{
    string base =
        "<bible>\n"
        "1:1 In the beginning God created the heavens and the earth.\n\n"
        "1:2 Now the earth was formless and void, and darkness was over the deep.\n\n"
        "1:3 And God said, Let there be light: and there was light.\n\n"
        "<austen>\n"
        "It is a truth universally acknowledged, that a single man in possession\n"
        "of a good fortune, must be in want of a wife.\n\n"
        "<stories>\n"
        "she said \xE2\x80\x9Cit's 3:16!\xE2\x80\x9D and laughed \xF0\x9F\x98\x82  \xE1\xB4\x80\xC9\xB4\xE1\xB4\x85 left.\n\n";
    string text;
    for (int r = 0; r < 20; r++)
        text += base;
    return text;
}


// ------------------------------------------------------------- slow reference
// Recounts every pair after every merge, over every word occurrence. Same
// rules as the header, but none of the incremental bookkeeping, so it is easy
// to trust. Also returns the final tokens of the whole text.

static vector<pair<uint32_t, uint32_t>> slow_train(const string &text, uint32_t vocab_size, vector<uint32_t> &final_ids,
                                                   int pretok = 1)
{
    Tokenizer scratch;
    vector<Segment> segs = split_specials(scratch, text, true);

    vector<vector<uint32_t>> words;     // every occurrence, in text order
    vector<int> special_at;             // -1, or the special at that position
    for (const Segment &seg : segs)
    {
        if (seg.special >= 0)
        {
            words.push_back({});
            special_at.push_back(seg.special);
        }
        else
        {
            for (const string &piece : pretokenize(seg.text, pretok))
            {
                vector<uint32_t> ids;
                for (char c : piece)
                    ids.push_back(static_cast<unsigned char>(c));
                words.push_back(ids);
                special_at.push_back(-1);
            }
        }
    }

    vector<pair<uint32_t, uint32_t>> merges;
    uint32_t num_merges = vocab_size > 256 ? vocab_size - 256 : 0;
    bool done = false;
    while (!done && merges.size() < num_merges)
    {
        unordered_map<uint64_t, int64_t> counts;
        for (const vector<uint32_t> &w : words)
        {
            for (size_t k = 0; k + 1 < w.size(); k++)
                counts[pair_key(w[k], w[k + 1])]++;
        }

        uint64_t best_key = 0;
        int64_t best_count = 0;
        for (const auto &kv : counts)
        {
            if (kv.second > best_count || (kv.second == best_count && kv.first < best_key))
            {
                best_key = kv.first;
                best_count = kv.second;
            }
        }

        if (best_count == 0)
            done = true;
        else
        {
            uint32_t a = static_cast<uint32_t>(best_key >> 32);
            uint32_t b = static_cast<uint32_t>(best_key & 0xFFFFFFFFu);
            uint32_t c = 256 + static_cast<uint32_t>(merges.size());
            for (vector<uint32_t> &w : words)
            {
                vector<uint32_t> merged;
                size_t i = 0;
                while (i < w.size())
                {
                    if (i + 1 < w.size() && w[i] == a && w[i + 1] == b)
                    {
                        merged.push_back(c);
                        i += 2;
                    }
                    else
                    {
                        merged.push_back(w[i]);
                        i += 1;
                    }
                }
                w = merged;
            }
            merges.push_back({a, b});
        }
    }

    final_ids.clear();
    for (size_t k = 0; k < words.size(); k++)
    {
        if (special_at[k] >= 0)
            final_ids.push_back(256 + static_cast<uint32_t>(merges.size()) + static_cast<uint32_t>(special_at[k]));
        else
            final_ids.insert(final_ids.end(), words[k].begin(), words[k].end());
    }
    return merges;
}


// ------------------------------------------------------------- checks

static void test_pretokenize()
{
    printf("0. pretokenize\n");
    check(pretokenize("Let there be light,", 1) == vector<string>{"Let", " there", " be", " light", ","},
          "\"Let there be light,\" -> Let | there | be | light | ,");
    check(pretokenize("  two  spaces ", 1) == vector<string>{" ", " two", " ", " spaces", " "},
          "extra spaces split off, the last one leads the word");
    check(pretokenize("it's 3:16!", 1) == vector<string>{"it", "'", "s", " 3", ":", "16", "!"},
          "letters, digits, punctuation are separate runs");
    check(pretokenize("end \n\nnext", 1) == vector<string>{"end", " ", "\n\n", "next"},
          "a run of newlines is one piece");
    check(pretokenize("caf\xC3\xA9 \xF0\x9F\x98\x82!", 1) == vector<string>{"caf\xC3\xA9", " \xF0\x9F\x98\x82", "!"},
          "UTF-8 bytes stay inside the word");
    check(pretokenize("", 1).empty(), "empty string gives no pieces");
    check(pretokenize("said \xE2\x80\x9CLet", 1) == vector<string>{"said", " \xE2\x80\x9C", "Let"},
          "a curly quote is punctuation, not part of the word");
    check(pretokenize("God\xE2\x80\x99s", 1) == vector<string>{"God", "\xE2\x80\x99", "s"},
          "a curly apostrophe splits like a straight one");
    check(pretokenize("own.\xE2\x80\x94\xE2\x80\x9CVery", 1) == vector<string>{"own", ".\xE2\x80\x94\xE2\x80\x9C", "Very"},
          "ASCII and curly punctuation join one run");
    check(pretokenize("ab\xE2\x80", 1) == vector<string>{"ab\xE2\x80"},
          "a cut off character stays with the word");
}

static void test_textbook()
{
    printf("1. textbook example \"aaabdaaabac\"\n");
    Tokenizer tok;
    train(tok, "aaabdaaabac", 259, false);

    vector<pair<uint32_t, uint32_t>> want = {{97, 97}, {97, 98}, {256, 257}};
    check(tok.merges == want, "merges (97,97)->256, (97,98)->257, (256,257)->258");
    check(tok.vocab.size() == 259 && tok.vocab[256] == "aa" && tok.vocab[257] == "ab" && tok.vocab[258] == "aaab",
          "vocab strings aa, ab, aaab");
    check(encode(tok, "aaabdaaabac") == vector<uint32_t>{258, 100, 258, 97, 99},
          "encode -> { 258, 100, 258, 97, 99 }");
}

static void test_round_trip()
{
    printf("2. round trip\n");
    Tokenizer tok;
    train(tok, sample_text(), 400, false);

    vector<string> cases = {
        "",
        "\n\n",
        "  two  spaces ",
        "1:1 In the beginning",
        "it's 3:16!",
        "I laughed \xF0\x9F\x98\x82 and \xF0\x9F\x91\x8B\xF0\x9F\x8F\xBD waved",
        "<bible>\nAnd God said",
        "tabs\tand\r\nwindows\r\n",
        "raw bytes \xFF\xFE\x80 that are not UTF-8",
        "<essays>\nan unknown tag is just text",
    };
    for (const string &s : cases)
    {
        string shown = s.size() > 30 ? s.substr(0, 30) + "..." : s;
        string escaped;
        for (char c : shown)
            escaped += (c == '\n') ? string("\\n") : (c == '\r') ? string("\\r") : (c == '\t') ? string("\\t") : string(1, c);
        check(decode(tok, encode(tok, s)) == s, "\"" + escaped + "\"");
    }
}

static void test_tags()
{
    printf("3. tags\n");
    string text;
    for (int r = 0; r < 30; r++)
        text += "<bible>\nthe end of it\n\n<austen>\nthe start of it\n\n";

    Tokenizer tok;
    train(tok, text, 400, false);
    check(tok.specials == vector<string>{"<bible>", "<austen>"}, "found <bible> and <austen> in order");

    vector<uint32_t> ids = encode(tok, "<bible>\nhi");
    check(!ids.empty() && ids[0] == special_id(tok, 0), "\"<bible>\\nhi\" starts with the <bible> id");
    check(decode(tok, ids) == "<bible>\nhi", "and decodes back");

    // Every merge is learned from plain text, so every merged token must show
    // up in the plain text. Pairs from tag bytes like "bi" or "au" never do.
    string plain;
    for (const Segment &seg : split_specials(tok, text, false))
    {
        if (seg.special < 0)
            plain += seg.text;
    }
    bool clean = true;
    for (size_t id = 256; id < 256 + tok.merges.size(); id++)
    {
        const string &t = tok.vocab[id];
        if (plain.find(t) == string::npos || t.find('<') != string::npos || t.find('>') != string::npos)
            clean = false;
    }
    check(clean, "no merge contains a tag's bytes (" + to_string(tok.merges.size()) + " merges checked)");

    vector<uint32_t> inline_ids = encode(tok, "say <bible> now");
    bool no_special = true;
    for (uint32_t id : inline_ids)
    {
        if (id >= special_id(tok, 0))
            no_special = false;
    }
    check(no_special, "a tag that is not alone on its line is plain text");
}

static void test_save_load()
{
    printf("4. save and load\n");
    Tokenizer tok;
    train(tok, sample_text(), 450, false);

    string path = "test_save_tmp.bpe";
    bool saved = save(tok, path);
    Tokenizer back;
    bool loaded = load(back, path);
    remove(path.c_str());

    check(saved && loaded, "save() and load() both succeed");
    check(back.merges == tok.merges, "same merges");
    check(back.specials == tok.specials, "same specials");
    check(back.vocab == tok.vocab, "same vocab");

    string s = sample_text().substr(0, 600) + "\n<stories>\nnew words \xF0\x9F\x98\x82 here";
    check(encode(back, s) == encode(tok, s), "same encode() output");

    Tokenizer junk;
    check(!load(junk, "does_not_exist.bpe"), "load() of a missing file returns false");

    const char *bad_files[][2] = {
        {"bpe v1\nmerges 2\n97 98\n97 98\nspecials 0\n", "load() rejects the same merge twice"},
        {"bpe v1\nmerges 0\nspecials 2\n<bible>\n<bible>\n", "load() rejects the same tag twice"},
        {"bpe v1\nmerges 1\n97 300\nspecials 0\n", "load() rejects a merge that uses a later id"},
    };
    for (const auto &bad : bad_files)
    {
        string bad_path = "test_bad_tmp.bpe";
        {
            ofstream f(bad_path, ios::binary);
            f << bad[0];
        }
        Tokenizer t;
        bool ok = load(t, bad_path);
        remove(bad_path.c_str());
        check(!ok, bad[1]);
    }
}

static void test_corpus_val()
{
    printf("5. data/corpus_val.txt\n");
    string path = "data/corpus_val.txt";
    if (!file_exists(path))
    {
        printf("  skip  %s not found (run from the repo root)\n", path.c_str());
        return;
    }
    string text = read_all(path);

    Tokenizer tok;
    if (load(tok, "data/tokenizer.bpe"))
        printf("        using data/tokenizer.bpe (%zu merges)\n", tok.merges.size());
    else
    {
        printf("        no data/tokenizer.bpe yet, training on the val file itself (4096)\n");
        train(tok, text, 4096, false);
    }

    vector<uint32_t> ids = encode(tok, text);
    check(decode(tok, ids) == text, "whole file round trips");

    size_t chars = 0;
    for (char c : text)
    {
        if ((static_cast<unsigned char>(c) & 0xC0) != 0x80)
            chars++;
    }
    double per = ids.empty() ? 0.0 : static_cast<double>(chars) / static_cast<double>(ids.size());
    printf("        %zu chars, %zu tokens, %.2f chars per token\n", chars, ids.size(), per);
}

static void test_against_slow()
{
    printf("6. fast train() matches a slow recount-everything trainer\n");
    mt19937 rng(12345);
    const string alphabet = "aaaabbbc  \n.";
    int trials = 400;
    int merge_mismatch = 0;
    int encode_mismatch = 0;

    const string no_spaces = "aaaabbbc";     // makes long words (> 128 bytes)
    for (int t = 0; t < trials; t++)
    {
        bool long_words = (t % 4 == 3);
        const string &letters = long_words ? no_spaces : alphabet;
        size_t len = rng() % (long_words ? 1500 : 300);
        string text;
        for (size_t k = 0; k < len; k++)
        {
            if (!long_words && rng() % 40 == 0)
                text += (rng() % 2) ? "\n<x>\n" : "\n<y_z>\n";
            else
                text += letters[rng() % letters.size()];
        }
        uint32_t vocab_size = 256 + rng() % 60;

        Tokenizer tok;
        train(tok, text, vocab_size, false);
        vector<uint32_t> slow_ids;
        vector<pair<uint32_t, uint32_t>> slow = slow_train(text, vocab_size, slow_ids);

        if (tok.merges != slow)
            merge_mismatch++;
        if (encode(tok, text) != slow_ids)
            encode_mismatch++;
    }
    check(merge_mismatch == 0, to_string(trials) + " random texts: same merges");
    check(encode_mismatch == 0, to_string(trials) + " random texts: encode() gives the training split");

    // pretokenizer v2: contractions and digit groups change the words, so the same check with them
    const string v2_alphabet = "aabst1234567890''re  \n.";
    int v2_merge_mismatch = 0, v2_encode_mismatch = 0, v2_round_trip = 0;
    for (int t = 0; t < 200; t++)
    {
        size_t len = rng() % 300;
        string text;
        for (size_t k = 0; k < len; k++)
        {
            if (rng() % 50 == 0)
                text += "\xE2\x80\x99";                // the curly apostrophe
            else
                text += v2_alphabet[rng() % v2_alphabet.size()];
        }
        uint32_t vocab_size = 256 + rng() % 60;
        Tokenizer tok;
        train(tok, text, vocab_size, false, 2);
        vector<uint32_t> slow_ids;
        vector<pair<uint32_t, uint32_t>> slow = slow_train(text, vocab_size, slow_ids, 2);
        vector<uint32_t> ids = encode(tok, text);
        v2_merge_mismatch += tok.merges != slow;
        v2_encode_mismatch += ids != slow_ids;
        v2_round_trip += decode(tok, ids) != text;
    }
    check(v2_merge_mismatch == 0, "200 random texts, pretokenizer v2: same merges");
    check(v2_encode_mismatch == 0, "200 random texts, pretokenizer v2: encode() gives the training split");
    check(v2_round_trip == 0, "200 random texts, pretokenizer v2: decode(encode(x)) == x");

    string path = "data/corpus_val.txt";
    if (file_exists(path))
    {
        string text = read_all(path).substr(0, 100000);
        Tokenizer tok;
        train(tok, text, 512, false);
        vector<uint32_t> slow_ids;
        vector<pair<uint32_t, uint32_t>> slow = slow_train(text, 512, slow_ids);
        check(tok.merges == slow, "first 100 KB of corpus_val.txt, 256 merges: same merges");
        check(encode(tok, text) == slow_ids, "first 100 KB of corpus_val.txt: encode() gives the training split");
    }
}

// ------------------------------------------------------------- pretokenizer v2

static void test_pretokenize_v2()
{
    printf("7. pretokenizer v2: contractions, digits in groups of 3, reserved tags, the v2 file\n");
    struct Case
    {
        const char *text;
        vector<string> want;
        const char *what;
    };
    const Case cases[] = {
        {"it's", {"it", "'s"}, "it's -> it 's"},
        {"don't", {"don", "'t"}, "don't -> don 't"},
        {"God\xE2\x80\x99s", {"God", "\xE2\x80\x99s"}, "God's with the curly apostrophe -> God 's"},
        {"'sup", {"'", "sup"}, "'sup: a letter follows, so no contraction"},
        {"rock 'n' roll", {"rock", " '", "n", "'", " roll"}, "rock 'n' roll: 'n is not a contraction"},
        {"we're they've you'll I'd I'm", {"we", "'re", " they", "'ve", " you", "'ll", " I", "'d", " I", "'m"},
         "'re 've 'll 'd 'm"},
        {"I'M", {"I", "'M"}, "upper case: I'M -> I 'M"},
        {" 1234567", {" 1", "234", "567"}, "\" 1234567\" -> \" 1\" \"234\" \"567\""},
        {"2024", {"2", "024"}, "2024 -> 2 024"},
        {" 7259", {" 7", "259"}, "\" 7259\" -> \" 7\" \"259\""},
        {"3.14159", {"3", ".", "14", "159"}, "3.14159 -> 3 . 14 159"},
        {" 12", {" 12"}, "\" 12\" stays one piece"},
        {"1000000", {"1", "000", "000"}, "1000000 -> 1 000 000"},
        {"it's 3:16!", {"it", "'s", " 3", ":", "16", "!"}, "it's 3:16! -> it 's 3 : 16 !"},
    };
    Tokenizer tok;
    string all;
    for (const Case &c : cases)
        all += string(c.text) + "\n";
    train(tok, sample_text() + all, 600, false, 2, {"<web>", "<user>", "<assistant>", "<system>"});
    for (const Case &c : cases)
    {
        check(pretokenize(c.text, 2) == c.want, c.what);
        check(decode(tok, encode(tok, c.text)) == c.text, string("  and it round trips: ") + c.text);
    }
    check(pretokenize("it's 1234", 1) == vector<string>{"it", "'", "s", " 1234"}, "pretok 1 keeps the stage 1 split");

    // reserved tags come first, in order, whether or not the text has them
    vector<string> want_tags = {"<web>", "<user>", "<assistant>", "<system>", "<bible>", "<austen>", "<stories>"};
    check(tok.specials == want_tags, "reserved tags first, then the text's own tags");
    uint32_t first = 256 + static_cast<uint32_t>(tok.merges.size());
    check(encode(tok, "<user>\nhi\n")[0] == first + 1, "<user> has the second tag id");

    // the file: v2 saves as "bpe v2" and loads back with its rules; v1 files still load as v1
    string path = "test_v2_tmp.bpe";
    save(tok, path);
    string head;
    {
        ifstream in(path);
        getline(in, head);
    }
    Tokenizer back;
    bool loaded = load(back, path);
    remove(path.c_str());
    check(head == "bpe v2", "a pretok 2 tokenizer saves \"bpe v2\"");
    check(loaded && back.pretok == 2 && back.specials == tok.specials && back.merges == tok.merges,
          "and loads back with pretok 2");
    check(encode(back, all) == encode(tok, all), "same encode() output after load");
    {
        ofstream f(path, ios::binary);
        f << "bpe v1\nmerges 1\n49 50\nspecials 0\n";
    }
    Tokenizer old;
    loaded = load(old, path);
    remove(path.c_str());
    check(loaded && old.pretok == 1, "a \"bpe v1\" file loads with pretok 1");
    check(encode(old, "1234") == vector<uint32_t>{256, '3', '4'}, "and encodes with the stage 1 rules (1234 is one word)");
    {
        ofstream f(path, ios::binary);
        f << "bpe v3\nmerges 0\nspecials 0\n";
    }
    Tokenizer future;
    loaded = load(future, path);
    remove(path.c_str());
    check(!loaded, "load() refuses a version it does not know");
}

int main()
{
    test_pretokenize();
    test_textbook();
    test_round_trip();
    test_tags();
    test_save_load();
    test_corpus_val();
    test_against_slow();
    test_pretokenize_v2();

    if (failures == 0)
        printf("\nall checks passed\n");
    else
        printf("\n%d check(s) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
