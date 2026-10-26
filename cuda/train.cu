// train.cu: the training loop on data/train.bin, in fp32 CUDA.
//
//   ./build/train_gpt                          full size, weights from data/ref/debug_full.bin
//   ./build/train_gpt -n 300                   300 steps
//
// The model sizes AND the starting weights come from the dump. dump.py --full
// builds the model with the same seed as reference/train.py, so both trainers
// start from identical weights, and the step 0 val loss should match the
// PyTorch log to about 5 digits. The schedule, eval batches and log line are
// the same as reference/train.py; only the random training batches differ.

#include <getopt.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "config.h"
#include "data.h"
#include "dump.h"
#include "gpt.h"

using namespace std;

static void usage(const char *prog)
{
    printf("Usage: %s [OPTIONS ...]\n", prog);
    printf("  -d file       dump with the model sizes and starting weights (default data/ref/debug_full.bin)\n");
    printf("  -w file       start from these weights instead (weights.bin, same sizes as the dump)\n");
    printf("  -n #          optimizer steps (default 5000)\n");
    printf("  -b #          sequences per step (default 16)\n");
    printf("  -e #          steps between evals (default 250)\n");
    printf("  -o dir        folder for cuda_log.csv (default data/ref)\n");
    printf("  -h            print this help message\n");
}

// x = tokens[s : s + T], y = tokens[s + 1 : s + T + 1], for each start s
static void fill_batch(vector<int> &x, vector<int> &y, const vector<int> &tokens,
                       const vector<long long> &starts, size_t first, int B, int T)
{
    for (int b = 0; b < B; b++)
    {
        long long s = starts[first + b];
        for (int t = 0; t < T; t++)
        {
            x[b * T + t] = tokens[s + t];
            y[b * T + t] = tokens[s + t + 1];
        }
    }
}

int main(int argc, char *argv[])
{
    string dump_path = "data/ref/debug_full.bin";
    string weights_path;
    string out_dir = "data/ref";
    TrainConfig tc;
    int opt = 0;
    while ((opt = getopt(argc, argv, "d:w:n:b:e:o:h")) != -1)
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
        case 'o':
            out_dir = optarg;
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

    GPT model(cfg, B, T);
    model.load_params(d);
    if (!weights_path.empty())
    {
        model.load(weights_path);
    }
    size_t n_params = model.params.count;
    printf("model  %s parameters | L %d H %d C %d n_ctx %d | V %d Vp %d | weights from %s\n",
           with_commas(static_cast<long long>(n_params)).c_str(), cfg.L, cfg.H, cfg.C, cfg.n_ctx, cfg.V, cfg.Vp,
           weights_path.empty() ? dump_path.c_str() : weights_path.c_str());
    printf("data   train %s tokens | val %s tokens | %s tokens per step\n",
           with_commas(static_cast<long long>(train_tokens.size())).c_str(),
           with_commas(static_cast<long long>(val_tokens.size())).c_str(),
           with_commas(static_cast<long long>(B) * T).c_str());

    // Val: the same evenly spaced starts as fixed_batches() in reference/data.py
    vector<long long> val_start = val_starts(val_tokens.size(), T, tc.eval_batches * B);

    vector<int> hx(static_cast<size_t>(B) * T);
    vector<int> hy(static_cast<size_t>(B) * T);
    DeviceInts x(static_cast<size_t>(B) * T);
    DeviceInts y(static_cast<size_t>(B) * T);

    auto evaluate = [&]()
    {
        double sum = 0.0;
        for (int c = 0; c < tc.eval_batches; c++)
        {
            fill_batch(hx, hy, val_tokens, val_start, static_cast<size_t>(c) * B, B, T);
            x.upload(hx.data(), hx.size());
            y.upload(hy.data(), hy.size());
            sum += model.forward(x.ptr, y.ptr, B, T);
        }
        return static_cast<float>(sum / tc.eval_batches);
    };

    float val = evaluate();
    printf("step    0 | val %.3f | ln(V) = %.3f\n", val, log(static_cast<double>(cfg.V)));
    fflush(stdout);

    string log_path = out_dir + "/cuda_log.csv";
    FILE *log = fopen(log_path.c_str(), "w");
    if (!log)
    {
        fprintf(stderr, "error: cannot write %s\n", log_path.c_str());
        return 1;
    }
    fprintf(log, "step,train,val,lr,norm,ms\n0,,%.6f,,,\n", val);

    mt19937_64 rng(tc.seed);
    uniform_int_distribution<long long> pick(0, static_cast<long long>(train_tokens.size()) - T - 1);
    vector<long long> starts(B);
    Timer timer;
    double window_loss = 0.0;
    int window_count = 0;
    double window_ms = 0.0;
    int window_timed = 0;
    double total_ms = 0.0;
    int total_timed = 0;

    for (int step = 0; step < tc.max_steps; step++)
    {
        float lr = lr_at(step, tc);
        for (int b = 0; b < B; b++)
        {
            starts[b] = pick(rng);
        }
        fill_batch(hx, hy, train_tokens, starts, 0, B, T);
        x.upload(hx.data(), hx.size());
        y.upload(hy.data(), hy.size());

        timer.start();
        model.zero_grad();
        float loss = model.forward(x.ptr, y.ptr, B, T);
        model.backward(x.ptr, y.ptr, B, T);
        float norm = model.clip(tc.grad_clip);
        model.update(lr, tc);
        timer.stop();
        float ms = timer.ms();

        window_loss += loss;
        window_count++;
        if (step >= tc.timing_skip)
        {
            window_ms += ms;
            window_timed++;
        }

        int done = step + 1;
        bool eval_now = done % tc.eval_every == 0 || done == tc.max_steps;
        if (eval_now)
        {
            val = evaluate();
        }
        fprintf(log, "%d,%.6f,%s,%.6e,%.4f,%.2f\n", done, loss, eval_now ? to_string(val).c_str() : "", lr, norm, ms);

        if (eval_now)
        {
            fflush(log);
            string speed = "warming up";
            if (window_timed > 0)
            {
                double tok_s = window_timed * static_cast<double>(B) * T / (window_ms / 1000.0);
                speed = with_commas(llround(tok_s)) + " tok/s";
            }
            printf("step %4d | train %.3f | val %.3f | lr %.1e | norm %.2f | %s\n",
                   done, window_loss / window_count, val, lr, norm, speed.c_str());
            fflush(stdout);
            total_ms += window_ms;
            total_timed += window_timed;
            window_loss = 0.0;
            window_count = 0;
            window_ms = 0.0;
            window_timed = 0;
        }
    }
    fclose(log);

    if (total_timed > 0)
    {
        double tok_s = total_timed * static_cast<double>(B) * T / (total_ms / 1000.0);
        printf("speed  %s tok/s over %d timed steps -> %s\n", with_commas(llround(tok_s)).c_str(), total_timed,
               log_path.c_str());
    }
    return 0;
}
