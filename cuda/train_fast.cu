// train_fast.cu: the stage 4 training loop (FastGPT), same data, schedule,
// eval batches and log line as train.cu and reference/train.py.
//
//   ./build/train_fast                         fp16, mma.sync GEMMs, flash attention, CUDA Graph
//   ./build/train_fast -n 1000 -s data/ref/starts.txt     PyTorch's exact batches (train.py --starts)
//   ./build/train_fast -G wmma -A unfused -g 0           turn optimizations off, one at a time
//   ./build/train_fast32                       the fp32 build (the correctness anchor)
//
// Stage 6, the chat model (sizes from the weights.bin header):
//   ./build/train_fast -w data/chat/init.bin -D data/chat/pretrain --warmup 2000 -n 30000 -c data/chat/pre
//   ./build/train_fast -w data/chat/pre/weights.bin -D data/chat/sft --sft -n 6000 -c data/chat/sft
//   ./build/train_fast -c data/chat/pre -r     pick a stopped run up from its last checkpoint
//
// The whole step runs on the GPU with no host decisions: the batch comes
// from every step's starts uploaded once, the loss scale, skip flag, lr,
// grad norm and bias corrections live in StepState. The host only reads
// them back at eval time (the StepState history ring), so a step never waits.

#include <getopt.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "checkpoint.h"
#include "common.h"
#include "config.h"
#include "data.h"
#include "dump.h"
#include "gpt_fast.h"
#include "graph.h"
#include "weights.h"

using namespace std;

#ifdef FP16
static const double PEAK_TFLOPS = 40.0;     // fp16 tensor cores, fp32 accumulate (same rule as reference/train.py)
#else
static const double PEAK_TFLOPS = 10.0;     // fp32 CUDA cores
#endif

// --sft: fine-tuning a pretrained model wants a smaller step (reference/config.py SFT_TRAIN)
static const float SFT_LR = 1e-4f, SFT_MIN_LR = 1e-5f;
static const int SFT_WARMUP = 100;

static void usage(const char *prog)
{
    printf("Usage: %s [OPTIONS ...]\n", prog);
    printf("  -d file       dump with the model sizes and starting weights (default data/ref/debug_full.bin)\n");
    printf("  -w file       start from these weights instead (weights.bin); without -d its header gives the sizes\n");
    printf("  -D prefix     data: prefix_train.bin and prefix_val.bin (default data/train.bin, data/val.bin)\n");
    printf("  --sft         train on the answers only: targets from prefix_train.tgt and prefix_val.tgt\n");
    printf("                (python.exe -m reference.chat), lr 1e-4, min lr 1e-5, warmup 100\n");
    printf("  -n #          optimizer steps (default 5000)\n");
    printf("  -b #          sequences per step (default 16)\n");
    printf("  -e #          steps between evals (default 250)\n");
    printf("  --lr x        peak learning rate (default 6e-4, --sft 1e-4)\n");
    printf("  --min-lr x    learning rate at the end (default 6e-5, --sft 1e-5)\n");
    printf("  --warmup #    warmup steps (default 200, --sft 100; 2000 for the chat pretrain)\n");
    printf("  -s file       batch starts from reference/train.py --starts (default: own random batches)\n");
    printf("  -c dir        checkpoint dir: dir/ckpt.bin after every eval, and the log and weights.bin go here\n");
    printf("  -r            resume from <-c dir>/ckpt.bin, with the settings saved in it\n");
    printf("  -g 0|1        capture the step as a CUDA Graph (default 1)\n");
    printf("  -G name       GEMM: mma (inline PTX), wmma, simt (fp32 cores) (default mma, fp32 build: simt)\n");
    printf("  -A name       attention: flash or unfused (the stage 3 kernels) (default flash, fp32 build: unfused)\n");
    printf("  -o dir        folder for the log and the weights (default data/ref, or the -c dir)\n");
    printf("  -l name       log file name inside -o (default fast_log.csv)\n");
    printf("  -S file       save the trained weights here (default <-o>/cuda_weights.bin, with -c <dir>/weights.bin,\n");
    printf("                \"\" for none)\n");
    printf("  -h            print this help message\n");
}

// One line per step, B numbers each (what reference/train.py --starts writes)
static vector<int> read_starts(const string &path, int steps, int B, size_t n_tokens, int T)
{
    ifstream in(path);
    if (!in)
    {
        fprintf(stderr, "error: cannot open %s\n", path.c_str());
        exit(1);
    }
    vector<int> starts;
    string line;
    int lines = 0;
    while (lines < steps && getline(in, line))
    {
        istringstream row(line);
        long long v;
        int count = 0;
        while (row >> v)
        {
            if (v < 0 || v + T >= static_cast<long long>(n_tokens))
            {
                fprintf(stderr, "error: %s line %d: start %lld does not fit %zu tokens\n", path.c_str(), lines + 1, v,
                        n_tokens);
                exit(1);
            }
            starts.push_back(static_cast<int>(v));
            count++;
        }
        if (count != B)
        {
            fprintf(stderr, "error: %s line %d has %d starts, the batch size is %d\n", path.c_str(), lines + 1, count,
                    B);
            exit(1);
        }
        lines++;
    }
    if (lines < steps)
    {
        fprintf(stderr, "error: %s has %d steps of starts, training wants %d\n", path.c_str(), lines, steps);
        exit(1);
    }
    return starts;
}

static GemmImpl parse_gemm(const string &name)
{
    if (name == "mma")
    {
        return GemmImpl::Mma;
    }
    if (name == "wmma")
    {
        return GemmImpl::Wmma;
    }
    if (name == "simt")
    {
        return GemmImpl::Simt;
    }
    fprintf(stderr, "error: -G takes mma, wmma or simt, not %s\n", name.c_str());
    exit(1);
}

// A resumed run appends to the log. Rows after the checkpoint's step came from
// steps that are about to run again (the run stopped between two checkpoints),
// so they go first.
static void trim_log(const string &path, int upto)
{
    ifstream in(path);
    vector<string> keep;
    string line;
    while (getline(in, line))
    {
        bool row = !line.empty() && line[0] >= '0' && line[0] <= '9';
        if (!row || atoi(line.c_str()) <= upto)
        {
            keep.push_back(line);
        }
    }
    in.close();
    ofstream out(path, ios::trunc);
    for (const string &k : keep)
    {
        out << k << "\n";
    }
}

// A flag given on the command line of a resumed run must agree with the checkpoint.
template <typename V>
static void same_as_saved(bool given, const V &now, const V &saved, const char *flag)
{
    if (given && !(now == saved))
    {
        fprintf(stderr, "error: the checkpoint was written with a different %s; -r resumes the same run, so leave "
                        "%s out (or start a new run without -r)\n", flag, flag);
        exit(1);
    }
}

int main(int argc, char *argv[])
{
    string dump_path = "data/ref/debug_full.bin";
    string weights_path, starts_path, data_prefix, ckpt_dir;
    string out_dir = "data/ref";
    string log_name = "fast_log.csv";
    string save_path = "-";
    bool use_graph = true, sft = false, resume = false;
    bool given_d = false, given_o = false, given_n = false, given_b = false, given_e = false;
    bool given_lr = false, given_min_lr = false, given_warmup = false, given_D = false, given_s = false;
    FastOptions fo;
    TrainConfig tc;
    enum
    {
        OPT_SFT = 1000,
        OPT_LR,
        OPT_MIN_LR,
        OPT_WARMUP,
    };
    static const option long_opts[] = {
        {"sft", no_argument, nullptr, OPT_SFT},
        {"lr", required_argument, nullptr, OPT_LR},
        {"min-lr", required_argument, nullptr, OPT_MIN_LR},
        {"warmup", required_argument, nullptr, OPT_WARMUP},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };
    int opt = 0;
    while ((opt = getopt_long(argc, argv, "d:w:D:n:b:e:s:c:rg:G:A:o:l:S:h", long_opts, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'd':
            dump_path = optarg;
            given_d = true;
            break;
        case 'w':
            weights_path = optarg;
            break;
        case 'D':
            data_prefix = optarg;
            given_D = true;
            break;
        case OPT_SFT:
            sft = true;
            break;
        case 'n':
            tc.max_steps = atoi(optarg);
            given_n = true;
            break;
        case 'b':
            tc.batch_size = atoi(optarg);
            given_b = true;
            break;
        case 'e':
            tc.eval_every = atoi(optarg);
            given_e = true;
            break;
        case OPT_LR:
            tc.lr = static_cast<float>(atof(optarg));
            given_lr = true;
            break;
        case OPT_MIN_LR:
            tc.min_lr = static_cast<float>(atof(optarg));
            given_min_lr = true;
            break;
        case OPT_WARMUP:
            tc.warmup_steps = atoi(optarg);
            given_warmup = true;
            break;
        case 's':
            starts_path = optarg;
            given_s = true;
            break;
        case 'c':
            ckpt_dir = optarg;
            break;
        case 'r':
            resume = true;
            break;
        case 'g':
            use_graph = atoi(optarg) != 0;
            break;
        case 'G':
            set_gemm_impl(parse_gemm(optarg));
            break;
        case 'A':
            if (string(optarg) != "flash" && string(optarg) != "unfused")
            {
                fprintf(stderr, "error: -A takes flash or unfused\n");
                return 1;
            }
            fo.flash = string(optarg) == "flash";
            break;
        case 'o':
            out_dir = optarg;
            given_o = true;
            break;
        case 'l':
            log_name = optarg;
            break;
        case 'S':
            save_path = optarg;
            break;
        case 'h':
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 1;
        }
    }
    if (resume && ckpt_dir.empty())
    {
        fprintf(stderr, "error: -r resumes from <-c dir>/ckpt.bin, so it needs -c\n");
        return 1;
    }
    if (!ckpt_dir.empty() && !given_o)
    {
        out_dir = ckpt_dir;
    }
    if (save_path == "-")
    {
        save_path = ckpt_dir.empty() ? out_dir + "/cuda_weights.bin" : ckpt_dir + "/weights.bin";
    }
    string ckpt_path = ckpt_dir.empty() ? "" : ckpt_dir + "/ckpt.bin";

    // ---- the model size and starting weights: a checkpoint, a weights.bin, or a dump
    ModelConfig cfg;
    Checkpoint ck;
    vector<float> start_params;             // empty: the dump's records
    string weights_from;
    if (resume)
    {
        ck = read_checkpoint(ckpt_path);
        cfg = ck.cfg;
        weights_from = ckpt_path;
        const RunInfo &r = ck.run;
        same_as_saved(given_n, tc.max_steps, r.max_steps, "-n");
        same_as_saved(given_b, tc.batch_size, r.B, "-b");
        same_as_saved(given_e, tc.eval_every, r.eval_every, "-e");
        same_as_saved(given_lr, tc.lr, r.lr, "--lr");
        same_as_saved(given_min_lr, tc.min_lr, r.min_lr, "--min-lr");
        same_as_saved(given_warmup, tc.warmup_steps, r.warmup_steps, "--warmup");
        same_as_saved(given_D, data_prefix, r.data, "-D");
        same_as_saved(given_s, starts_path, r.starts, "-s");
        same_as_saved(sft, 1, r.sft, "--sft");
        tc.max_steps = r.max_steps;
        tc.batch_size = r.B;
        tc.eval_every = r.eval_every;
        tc.lr = r.lr;
        tc.min_lr = r.min_lr;
        tc.warmup_steps = r.warmup_steps;
        data_prefix = r.data;
        starts_path = r.starts;
        sft = r.sft != 0;
    }
    else if (sft)
    {
        tc.lr = given_lr ? tc.lr : SFT_LR;
        tc.min_lr = given_min_lr ? tc.min_lr : SFT_MIN_LR;
        tc.warmup_steps = given_warmup ? tc.warmup_steps : SFT_WARMUP;
    }
    Dump d;
    bool use_dump = !resume && (given_d || weights_path.empty());
    if (use_dump || (resume && given_d))
    {
        d = read_dump(dump_path);
        ModelConfig dc = config_from_dump(d);
        if (resume && !same_size(dc, cfg))
        {
            fprintf(stderr, "error: %s is a different model size than %s\n", dump_path.c_str(), ckpt_path.c_str());
            return 1;
        }
        cfg = resume ? cfg : dc;
        weights_from = resume ? weights_from : dump_path;
    }
    if (!weights_path.empty())
    {
        Weights w = read_weights(weights_path);
        if ((use_dump || resume) && !same_size(w.cfg, cfg))
        {
            fprintf(stderr, "error: %s is a different model size than %s\n", weights_path.c_str(),
                    resume ? ckpt_path.c_str() : dump_path.c_str());
            return 1;
        }
        if (!resume)
        {
            cfg = w.cfg;
            start_params.swap(w.params);
            weights_from = weights_path;
        }
    }
    if (tc.max_steps <= 0 || tc.batch_size <= 0 || tc.eval_every <= 0)
    {
        fprintf(stderr, "error: -n, -b and -e must be positive\n");
        return 1;
    }
    if (sft && data_prefix.empty())
    {
        fprintf(stderr, "error: --sft needs -D prefix (the targets are prefix_train.tgt and prefix_val.tgt)\n");
        return 1;
    }

    // ---- data
    int B = tc.batch_size;
    int T = cfg.n_ctx;
    string train_path = data_prefix.empty() ? "data/train.bin" : data_prefix + "_train.bin";
    string val_path = data_prefix.empty() ? "data/val.bin" : data_prefix + "_val.bin";
    vector<uint16_t> train_tokens = load_tokens16(train_path, cfg.V);
    vector<uint16_t> val_tokens = load_tokens16(val_path, cfg.V);
    if (train_tokens.size() < static_cast<size_t>(T) + 1 || val_tokens.size() < static_cast<size_t>(T) + 1)
    {
        fprintf(stderr, "error: need at least %d tokens in %s and %s\n", T + 1, train_path.c_str(), val_path.c_str());
        return 1;
    }
    if (train_tokens.size() > static_cast<size_t>(INT_MAX))
    {
        fprintf(stderr, "error: %s has %zu tokens, batch starts are int (at most %d)\n", train_path.c_str(),
                train_tokens.size(), INT_MAX);
        return 1;
    }
    Targets train_tgt, val_tgt;
    if (sft)
    {
        train_tgt = load_targets(data_prefix + "_train.tgt", train_tokens.size(), T);
        val_tgt = load_targets(data_prefix + "_val.tgt", val_tokens.size(), T);
    }

    fo.history_len = tc.eval_every;         // the host reads the ring once per eval window
    FastGPT model(cfg, B, T, tc, 65536.0f, fo);
    if (resume)
    {
        restore(model, ck);
    }
    else if (start_params.empty())
    {
        model.load_params(d);
    }
    else
    {
        model.set_params(start_params);
    }
    int start_step = resume ? ck.state.step : 0;
    size_t n_params = model.params.count;
    cudaStream_t s = model.stream;
    printf("model  %s parameters | L %d H %d C %d n_ctx %d | V %d Vp %d | weights from %s\n",
           with_commas(static_cast<long long>(n_params)).c_str(), cfg.L, cfg.H, cfg.C, cfg.n_ctx, cfg.V, cfg.Vp,
           weights_from.c_str());
    printf("data   train %s tokens | val %s tokens | %s tokens per step | batches %s\n",
           with_commas(static_cast<long long>(train_tokens.size())).c_str(),
           with_commas(static_cast<long long>(val_tokens.size())).c_str(),
           with_commas(static_cast<long long>(B) * T).c_str(),
           starts_path.empty() ? "own (mt19937_64, seed 1337)" : starts_path.c_str());
    if (sft)
    {
        long long counted = 0;
        for (int y : train_tgt.y)
        {
            counted += y >= 0 ? 1 : 0;
        }
        printf("sft    %s conversation starts | %s counted targets (%.1f%% of the tokens) | val %s starts\n",
               with_commas(static_cast<long long>(train_tgt.starts.size())).c_str(), with_commas(counted).c_str(),
               100.0 * counted / train_tokens.size(), with_commas(static_cast<long long>(val_tgt.starts.size())).c_str());
    }
    printf("mode   %s | gemm %s | attention %s | graph %s | loss scale %s\n", FLOATX_NAME,
           gemm_impl_name(gemm_impl()), model.flash ? "flash" : "unfused (stage 3)", use_graph ? "on" : "off",
           model.sched.dynamic_scale ? "dynamic from 65,536" : "fixed 65,536");
    printf("lr     %.1e, warmup %d, cosine to %.1e at step %d%s\n", tc.lr, tc.warmup_steps, tc.min_lr, tc.max_steps,
           resume ? (", resuming at step " + to_string(start_step) + " from " + ckpt_path).c_str() : "");

    // ---- every step's batch starts, on the GPU once (a resumed run makes the same list)
    vector<int> starts;
    if (!starts_path.empty())
    {
        starts = read_starts(starts_path, tc.max_steps, B, train_tokens.size(), T);
    }
    else if (sft)
    {
        mt19937_64 rng(tc.seed);
        uniform_int_distribution<long long> pick(0, static_cast<long long>(train_tgt.starts.size()) - 1);
        starts.resize(static_cast<size_t>(tc.max_steps) * B);
        for (int &v : starts)
        {
            v = static_cast<int>(train_tgt.starts[pick(rng)]);
        }
    }
    else
    {
        mt19937_64 rng(tc.seed);
        uniform_int_distribution<long long> pick(0, static_cast<long long>(train_tokens.size()) - T - 1);
        starts.resize(static_cast<size_t>(tc.max_steps) * B);
        for (int &v : starts)
        {
            v = static_cast<int>(pick(rng));
        }
    }
    DeviceArray<uint16_t> d_train(train_tokens.size());
    DeviceInts d_targets(sft ? train_tgt.y.size() : 0), d_starts(starts.size());
    d_train.upload(train_tokens.data(), train_tokens.size());
    if (sft)
    {
        d_targets.upload(train_tgt.y.data(), train_tgt.y.size());
        vector<int>().swap(train_tgt.y);    // the GPU has them now
    }
    d_starts.upload(starts.data(), starts.size());
    const int *targets_ptr = sft ? d_targets.ptr : nullptr;

    // ---- val: fixed batches, eager forward passes, losses summed on the host in double.
    // SFT: each batch's mean is over its counted targets, so the batches are
    // weighted by their counts and the result is the mean over every counted target.
    vector<long long> val_start = sft ? val_starts_from(val_tgt.starts, tc.eval_batches * B)
                                      : val_starts(val_tokens.size(), T, tc.eval_batches * B);
    DeviceBuffer val_losses(tc.eval_batches);
    vector<int> hx(static_cast<size_t>(B) * T), hy(static_cast<size_t>(B) * T);
    vector<long long> val_counts(tc.eval_batches, static_cast<long long>(B) * T);
    auto evaluate = [&]()
    {
        for (int c = 0; c < tc.eval_batches; c++)
        {
            long long counted = 0;
            for (int b = 0; b < B; b++)
            {
                long long st = val_start[static_cast<size_t>(c) * B + b];
                for (int t = 0; t < T; t++)
                {
                    hx[b * T + t] = val_tokens[st + t];
                    hy[b * T + t] = sft ? val_tgt.y[st + t] : val_tokens[st + t + 1];
                    counted += hy[b * T + t] >= 0 ? 1 : 0;
                }
            }
            val_counts[c] = counted;
            model.x.upload(hx.data(), hx.size());
            model.y.upload(hy.data(), hy.size());
            model.forward(false, val_losses.ptr + c);
        }
        vector<float> l = val_losses.download();
        double sum = 0.0, weight = 0.0;
        for (int c = 0; c < tc.eval_batches; c++)
        {
            sum += sft ? l[c] * static_cast<double>(val_counts[c]) : l[c];
            weight += sft ? static_cast<double>(val_counts[c]) : 1.0;
        }
        return static_cast<float>(sum / max(1.0, weight));
    };

    auto one_step = [&]()
    {
        load_batch(model.x.ptr, model.y.ptr, d_train.ptr, targets_ptr, d_starts.ptr, tc.max_steps, model.state.ptr, B,
                   T, s);
        model.train_step();
    };
    CudaGraph graph;
    if (use_graph)
    {
        graph.capture(s, one_step);
    }

    RunInfo run;
    run.B = B;
    run.max_steps = tc.max_steps;
    run.eval_every = tc.eval_every;
    run.warmup_steps = tc.warmup_steps;
    run.sft = sft ? 1 : 0;
    run.lr = tc.lr;
    run.min_lr = tc.min_lr;
    run.data = data_prefix;
    run.starts = starts_path;

    float val = evaluate();
    printf("step %4d | val %.3f | ln(V) = %.3f\n", start_step, val, log(static_cast<double>(cfg.V)));
    fflush(stdout);

    filesystem::create_directories(out_dir);
    if (!ckpt_dir.empty())
    {
        filesystem::create_directories(ckpt_dir);
    }
    string log_path = out_dir + "/" + log_name;
    if (resume)
    {
        trim_log(log_path, start_step);
    }
    FILE *log = fopen(log_path.c_str(), resume ? "a" : "w");
    if (!log)
    {
        fprintf(stderr, "error: cannot write %s\n", log_path.c_str());
        return 1;
    }
    if (!resume)
    {
        fprintf(log, "step,train,val,lr,norm,ms,scale,skipped,counted\n0,,%.6f,,,,,,\n", val);
    }

    // one event before every step of the window: step k takes ev[k] -> ev[k + 1]
    vector<cudaEvent_t> ev(tc.eval_every + 1);
    for (cudaEvent_t &e : ev)
    {
        CUDA_CHECK(cudaEventCreate(&e));
    }
    vector<StepState> window(tc.eval_every);
    double total_ms = 0.0;
    long long total_timed = 0;
    float best_val = resume ? ck.run.best_val : val;
    int best_step = resume ? ck.run.best_step : 0;
    int first = start_step;                 // first step of the current window
    vector<float>().swap(ck.params);        // restored into the model, not needed on the host any more
    vector<float>().swap(ck.m);
    vector<float>().swap(ck.v);
    CUDA_CHECK(cudaEventRecord(ev[0], s));
    for (int step = start_step; step < tc.max_steps; step++)
    {
        if (use_graph)
        {
            graph.launch(s);
        }
        else
        {
            one_step();
        }
        CUDA_CHECK(cudaEventRecord(ev[step - first + 1], s));

        int done = step + 1;
        if (done % tc.eval_every != 0 && done != tc.max_steps)
        {
            continue;
        }
        int count = done - first;
        CUDA_CHECK(cudaStreamSynchronize(s));
        for (int k = 0; k < count; k++)     // the ring holds step i at i % eval_every
        {
            CUDA_CHECK(cudaMemcpy(&window[k], model.history.ptr + (first + k) % tc.eval_every, sizeof(StepState),
                                  cudaMemcpyDeviceToHost));
        }
        Checkpoint snap;
        if (!ckpt_dir.empty())
        {
            snap = checkpoint_of(model, run); // before eval, so the StepState is the training step's
        }
        val = evaluate();
        double window_loss = 0.0, window_ms = 0.0;
        int window_timed = 0;
        for (int k = 0; k < count; k++)
        {
            const StepState &h = window[k];
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev[k], ev[k + 1]));
            if (first + k - start_step >= tc.timing_skip)
            {
                window_ms += ms;
                window_timed++;
            }
            window_loss += h.loss;
            bool last = k == count - 1;
            fprintf(log, "%d,%.6f,%s,%.6e,%.4f,%.2f,%g,%d,%d\n", h.step + 1, h.loss,
                    last ? to_string(val).c_str() : "", h.lr, h.grad_norm, ms, h.loss_scale, h.found_inf, h.n_valid);
            if (h.found_inf)
            {
                printf("       step %d skipped: inf or NaN in the gradients, loss scale now %g\n", h.step + 1,
                       h.loss_scale);
            }
            else if (k > 0 && h.loss_scale != window[k - 1].loss_scale)
            {
                printf("       step %d: %d finite steps in a row, loss scale up to %g\n", h.step + 1,
                       model.sched.growth_interval, h.loss_scale);
            }
        }
        fflush(log);
        const StepState &h = window[count - 1];
        string speed = "warming up";
        if (window_timed > 0)
        {
            double tok_s = window_timed * static_cast<double>(B) * T / (window_ms / 1000.0);
            double mfu = 6.0 * n_params * tok_s / (PEAK_TFLOPS * 1e12);
            char buf[96];
            snprintf(buf, sizeof(buf), "%s tok/s | mfu %.1f%%", with_commas(llround(tok_s)).c_str(), 100.0 * mfu);
            speed = buf;
        }
        if (val < best_val)
        {
            best_val = val;
            best_step = done;
        }
        string saved;
        if (!ckpt_dir.empty())
        {
            snap.run.best_val = best_val;
            snap.run.best_step = best_step;
            write_checkpoint(ckpt_path, snap);
            saved = " | ckpt";
        }
        printf("step %4d | train %.3f | val %.3f | lr %.1e | norm %.2f | S %g | skipped %d | %s%s\n", done,
               window_loss / count, val, h.lr, h.grad_norm, h.loss_scale, h.skipped, speed.c_str(), saved.c_str());
        fflush(stdout);
        total_ms += window_ms;
        total_timed += window_timed;
        first = done;
        CUDA_CHECK(cudaEventRecord(ev[0], s));
    }
    fclose(log);
    for (cudaEvent_t &e : ev)
    {
        CUDA_CHECK(cudaEventDestroy(e));
    }

    StepState fin = model.read_state();
    if (total_timed > 0)
    {
        double tok_s = total_timed * static_cast<double>(B) * T / (total_ms / 1000.0);
        printf("speed  %s tok/s | MFU %.1f%% of %g TFLOPS over %lld timed steps -> %s\n",
               with_commas(llround(tok_s)).c_str(), 100.0 * 6.0 * n_params * tok_s / (PEAK_TFLOPS * 1e12),
               PEAK_TFLOPS, total_timed, log_path.c_str());
    }
    printf("done   best val %.3f at step %d | %d of %d steps skipped | final loss scale %g\n", best_val, best_step,
           fin.skipped, fin.step, fin.loss_scale);
    if (!save_path.empty())
    {
        write_weights(save_path, cfg, model.params.download());
        printf("saved  %s\n", save_path.c_str());
    }
    return 0;
}
