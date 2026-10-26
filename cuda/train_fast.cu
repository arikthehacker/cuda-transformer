// train_fast.cu: the stage 4 training loop (FastGPT), same data, schedule,
// eval batches and log line as train.cu and reference/train.py.
//
//   ./build/train_fast                         fp16, mma.sync GEMMs, flash attention, CUDA Graph
//   ./build/train_fast -n 1000 -s data/ref/starts.txt     PyTorch's exact batches (train.py --starts)
//   ./build/train_fast -G wmma -A unfused -g 0           turn optimizations off, one at a time
//   ./build/train_fast32                       the fp32 build (the correctness anchor)
//
// The whole step runs on the GPU with no host decisions: the batch comes
// from every step's starts uploaded once, the loss scale, skip flag, lr,
// grad norm and bias corrections live in StepState. The host only reads
// them back at eval time (the StepState history ring), so a step never waits.

#include <getopt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

static void usage(const char *prog)
{
    printf("Usage: %s [OPTIONS ...]\n", prog);
    printf("  -d file       dump with the model sizes and starting weights (default data/ref/debug_full.bin)\n");
    printf("  -w file       start from these weights instead (weights.bin, same sizes as the dump)\n");
    printf("  -n #          optimizer steps (default 5000)\n");
    printf("  -b #          sequences per step (default 16)\n");
    printf("  -e #          steps between evals (default 250)\n");
    printf("  -s file       batch starts from reference/train.py --starts (default: own random batches)\n");
    printf("  -g 0|1        capture the step as a CUDA Graph (default 1)\n");
    printf("  -G name       GEMM: mma (inline PTX), wmma, simt (fp32 cores) (default mma, fp32 build: simt)\n");
    printf("  -A name       attention: flash or unfused (the stage 3 kernels) (default flash, fp32 build: unfused)\n");
    printf("  -o dir        folder for the log and the weights (default data/ref)\n");
    printf("  -l name       log file name inside -o (default fast_log.csv)\n");
    printf("  -S file       save the trained weights here (default <-o>/cuda_weights.bin, \"\" for none)\n");
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

int main(int argc, char *argv[])
{
    string dump_path = "data/ref/debug_full.bin";
    string weights_path, starts_path;
    string out_dir = "data/ref";
    string log_name = "fast_log.csv";
    string save_path = "-";
    bool use_graph = true;
    FastOptions fo;
    TrainConfig tc;
    int opt = 0;
    while ((opt = getopt(argc, argv, "d:w:n:b:e:s:g:G:A:o:l:S:h")) != -1)
    {
        switch (opt)
        {
        case 'd':
            dump_path = optarg;
            break;
        case 'w':
            weights_path = optarg;
            break;
        case 'n':
            tc.max_steps = atoi(optarg);
            break;
        case 'b':
            tc.batch_size = atoi(optarg);
            break;
        case 'e':
            tc.eval_every = atoi(optarg);
            break;
        case 's':
            starts_path = optarg;
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
    if (tc.max_steps <= 0 || tc.batch_size <= 0 || tc.eval_every <= 0)
    {
        fprintf(stderr, "error: -n, -b and -e must be positive\n");
        return 1;
    }
    if (save_path == "-")
    {
        save_path = out_dir + "/cuda_weights.bin";
    }

    Dump d = read_dump(dump_path);
    ModelConfig cfg = config_from_dump(d);
    int B = tc.batch_size;
    int T = cfg.n_ctx;
    vector<int> train_tokens = load_tokens("data/train.bin", cfg.V);
    vector<int> val_tokens = load_tokens("data/val.bin", cfg.V);
    if (train_tokens.size() < static_cast<size_t>(T) + 1 || val_tokens.size() < static_cast<size_t>(T) + 1)
    {
        fprintf(stderr, "error: need at least %d tokens in train.bin and val.bin\n", T + 1);
        return 1;
    }

    fo.history_len = tc.eval_every;         // the host reads the ring once per eval window
    FastGPT model(cfg, B, T, tc, 65536.0f, fo);
    model.load_params(d);
    if (!weights_path.empty())
    {
        Weights w = read_weights(weights_path);
        if (w.cfg.V != cfg.V || w.cfg.Vp != cfg.Vp || w.cfg.n_ctx != cfg.n_ctx || w.cfg.L != cfg.L ||
            w.cfg.H != cfg.H || w.cfg.C != cfg.C)
        {
            fprintf(stderr, "error: %s is a different model size than %s\n", weights_path.c_str(), dump_path.c_str());
            return 1;
        }
        model.set_params(w.params);
    }
    size_t n_params = model.params.count;
    cudaStream_t s = model.stream;
    printf("model  %s parameters | L %d H %d C %d n_ctx %d | V %d Vp %d | weights from %s\n",
           with_commas(static_cast<long long>(n_params)).c_str(), cfg.L, cfg.H, cfg.C, cfg.n_ctx, cfg.V, cfg.Vp,
           weights_path.empty() ? dump_path.c_str() : weights_path.c_str());
    printf("data   train %s tokens | val %s tokens | %s tokens per step | batches %s\n",
           with_commas(static_cast<long long>(train_tokens.size())).c_str(),
           with_commas(static_cast<long long>(val_tokens.size())).c_str(),
           with_commas(static_cast<long long>(B) * T).c_str(),
           starts_path.empty() ? "own (mt19937_64, seed 1337)" : starts_path.c_str());
    printf("mode   %s | gemm %s | attention %s | graph %s | loss scale %s\n", FLOATX_NAME,
           gemm_impl_name(gemm_impl()), model.flash ? "flash" : "unfused (stage 3)", use_graph ? "on" : "off",
           model.sched.dynamic_scale ? "dynamic from 65,536" : "fixed 65,536");

    // ---- every step's batch starts, on the GPU once
    vector<int> starts;
    if (!starts_path.empty())
    {
        starts = read_starts(starts_path, tc.max_steps, B, train_tokens.size(), T);
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
    DeviceInts d_train(train_tokens.size()), d_starts(starts.size());
    d_train.upload(train_tokens.data(), train_tokens.size());
    d_starts.upload(starts.data(), starts.size());

    // ---- val: fixed batches, eager forward passes, losses summed on the host in double
    vector<long long> val_start = val_starts(val_tokens.size(), T, tc.eval_batches * B);
    DeviceBuffer val_losses(tc.eval_batches);
    vector<int> hx(static_cast<size_t>(B) * T), hy(static_cast<size_t>(B) * T);
    auto evaluate = [&]()
    {
        for (int c = 0; c < tc.eval_batches; c++)
        {
            for (int b = 0; b < B; b++)
            {
                long long st = val_start[static_cast<size_t>(c) * B + b];
                for (int t = 0; t < T; t++)
                {
                    hx[b * T + t] = val_tokens[st + t];
                    hy[b * T + t] = val_tokens[st + t + 1];
                }
            }
            model.x.upload(hx.data(), hx.size());
            model.y.upload(hy.data(), hy.size());
            model.forward(false, val_losses.ptr + c);
        }
        vector<float> l = val_losses.download();
        double sum = 0.0;
        for (float v : l)
        {
            sum += v;
        }
        return static_cast<float>(sum / tc.eval_batches);
    };

    auto one_step = [&]()
    {
        load_batch(model.x.ptr, model.y.ptr, d_train.ptr, d_starts.ptr, tc.max_steps, model.state.ptr, B, T, s);
        model.train_step();
    };
    CudaGraph graph;
    if (use_graph)
    {
        graph.capture(s, one_step);
    }

    float val = evaluate();
    printf("step    0 | val %.3f | ln(V) = %.3f\n", val, log(static_cast<double>(cfg.V)));
    fflush(stdout);

    filesystem::create_directories(out_dir);
    string log_path = out_dir + "/" + log_name;
    FILE *log = fopen(log_path.c_str(), "w");
    if (!log)
    {
        fprintf(stderr, "error: cannot write %s\n", log_path.c_str());
        return 1;
    }
    fprintf(log, "step,train,val,lr,norm,ms,scale,skipped\n0,,%.6f,,,,,\n", val);

    // one event before every step of the window: step k takes ev[k] -> ev[k + 1]
    vector<cudaEvent_t> ev(tc.eval_every + 1);
    for (cudaEvent_t &e : ev)
    {
        CUDA_CHECK(cudaEventCreate(&e));
    }
    vector<StepState> window(tc.eval_every);
    double total_ms = 0.0;
    long long total_timed = 0;
    float best_val = val;
    int best_step = 0;
    int first = 0;                          // first step of the current window
    CUDA_CHECK(cudaEventRecord(ev[0], s));
    for (int step = 0; step < tc.max_steps; step++)
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
        val = evaluate();
        double window_loss = 0.0, window_ms = 0.0;
        int window_timed = 0;
        for (int k = 0; k < count; k++)
        {
            const StepState &h = window[k];
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev[k], ev[k + 1]));
            if (first + k >= tc.timing_skip)
            {
                window_ms += ms;
                window_timed++;
            }
            window_loss += h.loss;
            bool last = k == count - 1;
            fprintf(log, "%d,%.6f,%s,%.6e,%.4f,%.2f,%g,%d\n", h.step + 1, h.loss, last ? to_string(val).c_str() : "",
                    h.lr, h.grad_norm, ms, h.loss_scale, h.found_inf);
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
        printf("step %4d | train %.3f | val %.3f | lr %.1e | norm %.2f | S %g | skipped %d | %s\n", done,
               window_loss / count, val, h.lr, h.grad_norm, h.loss_scale, h.skipped, speed.c_str());
        fflush(stdout);
        if (val < best_val)
        {
            best_val = val;
            best_step = done;
        }
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
