// test_checkpoint.cu: training checkpoints (checkpoint.h) and resuming a run.
//
//   ./build/test_checkpoint   data/ref/debug.bin       (fp16 build)
//   ./build/test_checkpoint32 data/ref/debug.bin       (fp32 build)
//
// 1. the file: write, read back, every field and value the same; written
//    again it gives the same bytes; a cut-off or foreign file is refused
// 2. resuming: 20 steps straight against 10 steps, a checkpoint file, a NEW
//    model restored from it, then 10 more steps. Same batches (load_batch
//    from the uploaded starts, SFT targets with -1 in them), same graph.
//    The losses of steps 11 to 20 must be identical.
//
// On a GPU the gradient sums that use atomics (split-K GEMMs, the wte
// scatter, the flash attention dq) add in a different order on every run, so
// two runs of the SAME 20 steps can already differ in the last bits. The
// test runs the straight 20 steps twice first: if those match bit for bit
// (the CPU emulator, or a GPU that happens to), the resumed run must match
// bit for bit too. If not, fixed bars: on the 2080 two straight runs differ
// by at most about 1e-5 in the losses and 2e-3 in the parameters (largest
// difference over the largest value), while a resume that loses AdamW's m
// and v or its step count moves the losses by 1e-2 or more. The noise of one
// pair of runs swings by 1,000 times between runs, so it makes a poor bar.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "checkpoint.h"
#include "common.h"
#include "config.h"
#include "dump.h"
#include "gpt_fast.h"
#include "graph.h"
#include "kernels/fast.h"

using namespace std;

static const int STEPS = 20, HALF = 10;
static const float SCALE = 65536.0f;
static int g_checks_failed = 0;

static void check(bool ok, const string &what)
{
    printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        g_checks_failed++;
    }
}

static vector<unsigned char> file_bytes(const string &path)
{
    ifstream in(path, ios::binary);
    return vector<unsigned char>((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
}

static bool exists(const string &path)
{
    ifstream in(path, ios::binary);
    return static_cast<bool>(in);
}

static bool same_state(const StepState &a, const StepState &b)
{
    return memcmp(&a, &b, sizeof(StepState)) == 0;
}

// ---------------------------------------------------------------- the file

static void file_checks(const ModelConfig &cfg)
{
    printf("\ncheckpoint file: write, read back, write again\n");
    Checkpoint c;
    c.cfg = cfg;
    c.run.B = 16;
    c.run.max_steps = 30000;
    c.run.eval_every = 250;
    c.run.warmup_steps = 2000;
    c.run.sft = 1;
    c.run.lr = 6e-4f;
    c.run.min_lr = 6e-5f;
    c.run.data = "data/chat/sft";
    c.run.starts = "a file with spaces.txt";
    c.run.best_val = 2.718f;
    c.run.best_step = 4750;
    c.state = initial_state(32768.0f);
    c.state.step = 4750;
    c.state.adam_t = 4741;
    c.state.skipped = 9;
    c.state.good_steps = 1234;
    c.state.lr = 1.5e-4f;
    c.state.loss = 3.25f;
    c.state.n_valid = 4093;
    mt19937 rng(7);
    normal_distribution<float> nd(0.0f, 1.0f);
    for (vector<float> *a : {&c.params, &c.m, &c.v})
    {
        a->resize(1003);
        for (float &x : *a)
        {
            x = nd(rng);
        }
    }
    c.v[5] = -0.0f;                         // the sign of a zero, and odd values, survive
    c.m[6] = 1e-40f;

    const string path = "test_checkpoint.tmp.bin", path2 = "test_checkpoint2.tmp.bin";
    write_checkpoint(path, c);
    check(!exists(path + ".tmp"), "no .tmp file left behind after the rename");
    Checkpoint b = read_checkpoint(path);
    bool same = same_size(b.cfg, c.cfg) && b.run.B == c.run.B && b.run.max_steps == c.run.max_steps &&
                b.run.eval_every == c.run.eval_every && b.run.warmup_steps == c.run.warmup_steps &&
                b.run.sft == c.run.sft && b.run.lr == c.run.lr && b.run.min_lr == c.run.min_lr &&
                b.run.data == c.run.data && b.run.starts == c.run.starts && b.run.best_val == c.run.best_val &&
                b.run.best_step == c.run.best_step;
    check(same, "sizes and run settings read back the same");
    check(same_state(b.state, c.state), "the StepState reads back byte for byte");
    bool arrays = memcmp(b.params.data(), c.params.data(), c.params.size() * sizeof(float)) == 0 &&
                  memcmp(b.m.data(), c.m.data(), c.m.size() * sizeof(float)) == 0 &&
                  memcmp(b.v.data(), c.v.data(), c.v.size() * sizeof(float)) == 0;
    check(arrays, "params, m and v read back bit for bit");
    write_checkpoint(path2, b);
    check(file_bytes(path) == file_bytes(path2), "writing what was read gives the same bytes");
    write_checkpoint(path, b);              // over an existing file
    check(file_bytes(path) == file_bytes(path2), "a second write over the old file replaces it");

    vector<unsigned char> raw = file_bytes(path);
    string why;
    Checkpoint bad;
    {
        ofstream out(path2, ios::binary);
        out.write(reinterpret_cast<const char *>(raw.data()), static_cast<streamsize>(raw.size() - 5));
    }
    bool read_cut = try_read_checkpoint(path2, bad, why);
    check(!read_cut, "a cut-off file is refused (" + why + ")");
    raw[0] = 'X';
    {
        ofstream out(path2, ios::binary);
        out.write(reinterpret_cast<const char *>(raw.data()), static_cast<streamsize>(raw.size()));
    }
    bool read_foreign = try_read_checkpoint(path2, bad, why);
    check(!read_foreign, "a file that is not GPTC is refused (" + why + ")");
    remove(path.c_str());
    remove(path2.c_str());
}

// ---------------------------------------------------------------- resuming

struct Data
{
    DeviceArray<uint16_t> tokens;
    DeviceInts targets, starts;
};

// Tokens, SFT targets with about a third -1, and STEPS batches of starts.
static Data make_data(const ModelConfig &cfg, int B, int T)
{
    const int NTOK = 4096;
    mt19937 rng(11);
    vector<uint16_t> tok(NTOK);
    vector<int> tgt(NTOK), st(STEPS * B);
    for (int i = 0; i < NTOK; i++)
    {
        tok[i] = static_cast<uint16_t>(rng() % 40 == 0 ? rng() % cfg.V : rng() % 50);
    }
    for (int i = 0; i < NTOK; i++)
    {
        tgt[i] = (i + 1 < NTOK && rng() % 3 != 0) ? tok[i + 1] : -1;
    }
    for (int &v : st)
    {
        v = static_cast<int>(rng() % (NTOK - T - 1));
    }
    Data d{DeviceArray<uint16_t>(NTOK), DeviceInts(NTOK), DeviceInts(st.size())};
    d.tokens.upload(tok.data(), tok.size());
    d.targets.upload(tgt.data(), tgt.size());
    d.starts.upload(st.data(), st.size());
    return d;
}

static TrainConfig run_config(const Dump &d)
{
    TrainConfig tc;
    tc.beta1 = d.beta1;
    tc.beta2 = d.beta2;
    tc.eps = d.eps;
    tc.weight_decay = d.weight_decay;
    tc.lr = 1e-3f;
    tc.min_lr = 1e-4f;
    tc.warmup_steps = 4;                    // the lr changes every step, so a resume at the wrong step shows
    tc.max_steps = STEPS;
    return tc;
}

// steps [from, to) through a captured graph; the loss of each from the history ring
static vector<float> run_steps(FastGPT &model, const Data &data, int from, int to)
{
    int B = model.B, T = model.T;
    CudaGraph graph;
    graph.capture(model.stream, [&]()
                  {
                      load_batch(model.x.ptr, model.y.ptr, data.tokens.ptr, data.targets.ptr, data.starts.ptr, STEPS,
                                 model.state.ptr, B, T, model.stream);
                      model.train_step();
                  });
    for (int k = from; k < to; k++)
    {
        graph.launch(model.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(model.stream));
    vector<StepState> h(model.history_len);
    CUDA_CHECK(cudaMemcpy(h.data(), model.history.ptr, h.size() * sizeof(StepState), cudaMemcpyDeviceToHost));
    vector<float> losses;
    for (int k = from; k < to; k++)
    {
        losses.push_back(h[k % model.history_len].loss);
    }
    return losses;
}

static FastOptions options()
{
    FastOptions fo;
    fo.history_len = 32;
    return fo;
}

static double max_rel(const vector<float> &a, const vector<float> &b)
{
    double worst = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++)
    {
        double d = fabs(static_cast<double>(a[i]) - b[i]) / max(1e-30, fabs(static_cast<double>(a[i])));
        worst = max(worst, a[i] == b[i] ? 0.0 : d);
    }
    return worst;
}

// Largest difference over the largest value: a weight near 0 does not blow it up
static double max_scaled(const vector<float> &a, const vector<float> &b)
{
    double diff = 0.0, scale = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++)
    {
        diff = max(diff, fabs(static_cast<double>(a[i]) - b[i]));
        scale = max(scale, fabs(static_cast<double>(a[i])));
    }
    return diff / max(1e-30, scale);
}

static const double LOSS_BAR = 1e-4, PARAM_BAR = 1e-2;

static void resume_checks(const Dump &d, const ModelConfig &cfg)
{
    printf("\nresume: %d steps straight against %d, a checkpoint file, a new model, %d more\n", STEPS, HALF, HALF);
    int B = d.B, T = d.T;
    Data data = make_data(cfg, B, T);
    TrainConfig tc = run_config(d);

    vector<vector<float>> straight;
    vector<StepState> straight_end;
    vector<vector<float>> straight_params;
    for (int r = 0; r < 2; r++)
    {
        FastGPT m(cfg, B, T, tc, SCALE, options());
        m.load_params(d);
        straight.push_back(run_steps(m, data, 0, STEPS));
        straight_end.push_back(m.read_state());
        straight_params.push_back(m.params.download());
    }

    const string path = "test_checkpoint_resume.tmp.bin";
    vector<float> first_half;
    {
        FastGPT m(cfg, B, T, tc, SCALE, options());
        m.load_params(d);
        first_half = run_steps(m, data, 0, HALF);
        RunInfo run;
        run.B = B;
        run.max_steps = STEPS;
        write_checkpoint(path, checkpoint_of(m, run));
    }                                       // that model is gone: only the file carries the run on
    FastGPT resumed(cfg, B, T, tc, SCALE, options());
    Checkpoint c = read_checkpoint(path);
    restore(resumed, c);
    check(resumed.read_state().step == HALF && c.state.adam_t == HALF, "the restored state is at step 10, AdamW t 10");
    vector<float> second_half = run_steps(resumed, data, HALF, STEPS);
    StepState end = resumed.read_state();
    vector<float> params = resumed.params.download();
    remove(path.c_str());

    vector<float> resumed_all = first_half;
    resumed_all.insert(resumed_all.end(), second_half.begin(), second_half.end());
    printf("  %-6s %-16s %-16s %-16s\n", "step", "straight", "straight again", "resumed");
    for (int k = 0; k < STEPS; k++)
    {
        printf("  %-6d %-16.9g %-16.9g %-16.9g%s\n", k + 1, straight[0][k], straight[1][k], resumed_all[k],
               k == HALF ? "   <- after the checkpoint" : "");
    }
    double noise = max_rel(straight[0], straight[1]);
    double loss_gap = max_rel(straight[0], resumed_all);
    double param_noise = max_scaled(straight_params[0], straight_params[1]);
    double param_gap = max_scaled(straight_params[0], params);
    bool exact = noise == 0.0 && param_noise == 0.0;
    printf("  two straight runs differ by %.3g (losses), %.3g (params): %s\n", noise, param_noise,
           exact ? "deterministic, so the resumed run must match bit for bit"
                 : "the atomics' order, so the bars are fixed (losses 1e-4, params 1e-2)");
    bool losses_ok = exact ? loss_gap == 0.0 : loss_gap <= LOSS_BAR;
    bool params_ok = exact ? param_gap == 0.0 : param_gap <= PARAM_BAR;
    check(losses_ok, "losses of steps 11 to 20 after the resume (gap " + to_string(loss_gap) + ")");
    check(params_ok, "parameters after 20 steps (gap " + to_string(param_gap) + ")");
    check(end.step == STEPS && end.adam_t == straight_end[0].adam_t && end.skipped == straight_end[0].skipped &&
              end.loss_scale == straight_end[0].loss_scale && end.good_steps == straight_end[0].good_steps,
          "step, AdamW t, skipped steps and the loss scale end the same");
    check(end.lr == straight_end[0].lr, "the last step used the same learning rate");
}

int main(int argc, char *argv[])
{
    string path = argc > 1 ? argv[1] : "data/ref/debug.bin";
    printf("test_checkpoint, %s build, gemm %s\n", FLOATX_NAME, gemm_impl_name(gemm_impl()));
    Dump d = read_dump(path);
    ModelConfig cfg = config_from_dump(d);
    printf("==== %s: L %d H %d C %d B %u T %u\n", path.c_str(), cfg.L, cfg.H, cfg.C, d.B, d.T);
    file_checks(cfg);
    resume_checks(d, cfg);
    printf("\n%s\n", g_checks_failed == 0 ? "checkpoints and resuming work"
                                          : (to_string(g_checks_failed) + " failure(s)").c_str());
    return g_checks_failed == 0 ? 0 : 1;
}
