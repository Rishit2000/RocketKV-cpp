#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <sys/resource.h>
#include <vector>

static ggml_type get_kv_type(const std::string & type_str) {
    if (type_str == "q8_0") return GGML_TYPE_Q8_0;
    if (type_str == "q4_0") return GGML_TYPE_Q4_0;
    return GGML_TYPE_F16;
}

struct InferenceOutput {
    std::string text;
    double ttft_ms = 0.0;                // User-facing TTFT: prefill + eviction
    double prefill_ms = 0.0;             // Pure Metal GPU prefill latency
    double evict_ms = 0.0;               // Native Metal GPU RocketKV Stage 1 compaction latency
    double decode_time_ms = 0.0;         // Pure decode phase latency
    double compute_time_ms = 0.0;        // Total inference compute time: prefill + eviction + decode
    double ms_per_tok = 0.0;             // Pure decode step latency
    double tok_per_sec = 0.0;            // Generation throughput
    double peak_memory_mb = 0.0;         // Peak process memory footprint (macOS getrusage)
    double kv_traffic_mb_per_tok = 0.0;  // KV (+ kt summary) traffic transferred per decode step
    double effective_bandwidth_gbs = 0.0;// Total effective decode memory bandwidth: weights + KV (GB/s)
    int prompt_tokens = 0;               // Prompt length after truncation
    int kept_tokens = 0;                 // Surviving prompt tokens after Stage 1 compaction
    int decode_tokens = 0;               // Number of autoregressive decode forward passes
    double prompt_kv_mb = 0.0;           // Physical KV (+ active kt summary) right after prefill/eviction
    double kt_cache_mb = 0.0;            // Stage 2 transposed [k_min, k_max] summary footprint (MB)
    double uncompressed_mb = 0.0;        // Uncompressed FP16 full KV baseline
};

static void run_warmup(
    llama_context * ctx,
    const llama_vocab * vocab,
    bool enable_rocket,
    bool enable_stage2
) {
    llama_memory_t mem = llama_get_memory(ctx);

    llama_token bos = llama_vocab_bos(vocab);
    if (bos < 0) bos = 1;

    // Standard dense prefill + decode warmup
    llama_rocket_config disabled_cfg = { false, 0, 0, 32, 63, false };
    llama_rocket_configure(ctx, disabled_cfg, 0);

    llama_token dummy_tokens[2] = { bos, 1 };
    llama_batch batch_prefill = llama_batch_get_one(&dummy_tokens[0], 1);
    llama_decode(ctx, batch_prefill);
    llama_synchronize(ctx);

    struct llama_sampler * smpl = llama_sampler_init_greedy();
    llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_free(smpl);

    llama_batch batch_decode = llama_batch_init(1, 0, 1);
    batch_decode.token[0] = dummy_tokens[1];
    batch_decode.pos[0] = 1;
    batch_decode.n_seq_id[0] = 1;
    batch_decode.seq_id[0][0] = 0;
    batch_decode.logits[0] = true;
    batch_decode.n_tokens = 1;
    llama_decode(ctx, batch_decode);
    llama_synchronize(ctx);

    llama_memory_seq_rm(mem, -1, -1, -1);

    // Native Metal RocketKV Stage 1 + Stage 2 pipeline warmup (pre-compiles PSOs)
    if (enable_rocket && llama_n_ctx(ctx) >= 256) {
        const int n_warm = 128;
        llama_rocket_config warm_cfg = {
            /*.enabled        =*/ true,
            /*.token_budget   =*/ 32,
            /*.max_new_tokens =*/ 4,
            /*.window_size    =*/ 8,
            /*.kernel_size    =*/ 7,
            /*.enable_stage2  =*/ enable_stage2,
        };
        llama_rocket_configure(ctx, warm_cfg, n_warm);

        llama_batch wb = llama_batch_init(n_warm, 0, 1);
        for (int i = 0; i < n_warm; ++i) {
            wb.token[i] = (i == 0) ? bos : 1;
            wb.pos[i] = i;
            wb.n_seq_id[i] = 1;
            wb.seq_id[i][0] = 0;
            wb.logits[i] = (i == n_warm - 1);
        }
        wb.n_tokens = n_warm;
        llama_decode(ctx, wb);
        llama_synchronize(ctx);
        llama_batch_free(wb);

        llama_rocket_compress_prefill(ctx);

        batch_decode.token[0] = 1;
        batch_decode.pos[0] = n_warm;
        batch_decode.n_seq_id[0] = 1;
        batch_decode.seq_id[0][0] = 0;
        batch_decode.logits[0] = true;
        batch_decode.n_tokens = 1;
        llama_decode(ctx, batch_decode);
        llama_synchronize(ctx);

        llama_memory_seq_rm(mem, -1, -1, -1);
        llama_rocket_configure(ctx, disabled_cfg, 0);
    }

    llama_batch_free(batch_decode);
}

static InferenceOutput run_inference(
    llama_context * ctx,
    llama_model * model,
    const llama_vocab * vocab,
    const llama_context_params & ctx_params,
    const std::string & prompt,
    int max_tokens,
    int token_budget,
    bool enable_rocket,
    bool enable_stage2,
    int window_size,
    int kernel_size,
    int context_len_limit,
    float bytes_per_elem
) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, -1, -1, -1);

    // Tokenize prompt
    std::vector<llama_token> tokens(prompt.length() + 32);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    // Official LongBench middle-truncation convention
    if (context_len_limit > 0 && n_tokens > context_len_limit) {
        int half = context_len_limit / 2;
        std::vector<llama_token> truncated;
        truncated.reserve(context_len_limit);
        truncated.insert(truncated.end(), tokens.begin(), tokens.begin() + half);
        truncated.insert(truncated.end(), tokens.end() - half, tokens.end());
        tokens = std::move(truncated);
        n_tokens = (int)tokens.size();
    }

    const bool use_rocket = enable_rocket && token_budget > 0;
    llama_rocket_config rcfg = {
        /*.enabled        =*/ use_rocket,
        /*.token_budget   =*/ token_budget,
        /*.max_new_tokens =*/ max_tokens,
        /*.window_size    =*/ window_size,
        /*.kernel_size    =*/ kernel_size,
        /*.enable_stage2  =*/ enable_stage2,
    };
    llama_rocket_configure(ctx, rcfg, n_tokens);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1);
    }
    batch.n_tokens = n_tokens;

    // --- PREFILL PHASE ---
    int64_t t_start_prefill = ggml_time_us();
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "Prefill failed\n";
        llama_batch_free(batch);
        return {};
    }
    llama_synchronize(ctx);
    int64_t t_end_prefill = ggml_time_us();
    llama_batch_free(batch);
    double prefill_ms = (t_end_prefill - t_start_prefill) / 1000.0;

    // --- NATIVE METAL ROCKETKV STAGE 1 COMPACTION ---
    int final_kept = n_tokens;
    double evict_ms = 0.0;
    if (use_rocket) {
        int64_t t_evict_start = ggml_time_us();
        final_kept = llama_rocket_compress_prefill(ctx);
        int64_t t_evict_end = ggml_time_us();
        evict_ms = (t_evict_end - t_evict_start) / 1000.0;
    }

    llama_rocket_state_info rstate = llama_rocket_get_state(ctx);
    double ttft_ms = prefill_ms + evict_ms;

    // --- FUSED METAL STAGE 2 SPARSE FLASH ATTENTION ---
    struct llama_sampler * smpl = llama_sampler_init_greedy();
    int n_cur = n_tokens;
    int n_decode_steps = 0;
    int n_decode_calls = 0;
    std::string gen_text = "";
    llama_batch batch_decode = llama_batch_init(1, 0, 1);

    // Sample the initial token generated from the prefill prompt
    llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

    int64_t t_start_decode = ggml_time_us();

    while (n_cur < (int)ctx_params.n_ctx && n_decode_steps < max_tokens) {
        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char piece_buf[256];
        int n_piece = llama_token_to_piece(vocab, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (n_piece > 0) {
            gen_text.append(piece_buf, n_piece);
        }

        n_decode_steps++;
        if (n_decode_steps >= max_tokens) break;

        batch_decode.token[0] = new_token_id;
        batch_decode.pos[0] = n_cur;
        batch_decode.n_seq_id[0] = 1;
        batch_decode.seq_id[0][0] = 0;
        batch_decode.logits[0] = true;
        batch_decode.n_tokens = 1;

        if (llama_decode(ctx, batch_decode) != 0) break;

        n_decode_calls++;
        n_cur++;
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
    }

    llama_synchronize(ctx);
    int64_t t_end_decode = ggml_time_us();

    llama_batch_free(batch_decode);
    llama_sampler_free(smpl);

    double decode_time_ms = (n_decode_calls > 0) ? ((t_end_decode - t_start_decode) / 1000.0) : 0.0;
    double ms_per_tok = (n_decode_calls > 0) ? (decode_time_ms / n_decode_calls) : 0.0;
    double tok_per_sec = (n_decode_calls > 0 && decode_time_ms > 0.0) ? (n_decode_calls / (decode_time_ms / 1000.0)) : 0.0;
    double compute_time_ms = prefill_ms + evict_ms + decode_time_ms;

    const int model_n_layer = llama_model_n_layer(model);
    const int model_n_head = llama_model_n_head(model);
    const int model_n_head_kv = llama_model_n_head_kv(model);
    const int model_dim_head = llama_model_n_embd(model) / model_n_head;

    const double bytes_per_tok = 2.0 * model_n_layer * model_n_head_kv * model_dim_head * bytes_per_elem;
    const double fp16_bytes_per_tok = 2.0 * model_n_layer * model_n_head_kv * model_dim_head * 2.0;
    const double uncompressed_mb = (n_tokens * fp16_bytes_per_tok) / (1024.0 * 1024.0);

    const bool stage2_active_prefill = use_rocket && enable_stage2 && rstate.topk_decode > 0 && final_kept > rstate.topk_decode;
    double kt_cache_bytes = 0.0;
    if (stage2_active_prefill && rstate.chunk_size > 0) {
        const int num_chunks = (final_kept + rstate.chunk_size - 1) / rstate.chunk_size;
        const int kt_rows = (rstate.chunk_size >= 2) ? (2 * model_dim_head) : model_dim_head;
        kt_cache_bytes = (double)model_n_layer * model_n_head_kv * kt_rows * num_chunks * sizeof(ggml_fp16_t);
    }
    const double kt_cache_mb = kt_cache_bytes / (1024.0 * 1024.0);
    const double prompt_kv_mb = (final_kept * bytes_per_tok + kt_cache_bytes) / (1024.0 * 1024.0);

    // Clean KV cache and RocketKV per-request flags
    llama_memory_seq_rm(mem, -1, -1, -1);

    // Peak memory footprint queried via macOS getrusage (ru_maxrss in bytes)
    struct rusage usage;
    double peak_memory_mb = 0.0;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        peak_memory_mb = (double)usage.ru_maxrss / (1024.0 * 1024.0);
    }

    // Exact per-step KV (+ Stage 2 kt summary) memory traffic averaged across decode forward passes
    double kv_traffic_bytes = 0.0;
    if (n_decode_calls > 0) {
        double total_kv_traffic = 0.0;
        for (int s = 0; s < n_decode_calls; ++s) {
            const int cur_kv = final_kept + s + 1;
            if (use_rocket && enable_stage2 && rstate.topk_decode > 0 && rstate.chunk_size > 0 && cur_kv > rstate.topk_decode) {
                const int num_chunks = (cur_kv + rstate.chunk_size - 1) / rstate.chunk_size;
                const double kt_scan_bytes = (double)model_n_layer * model_n_head_kv * rstate.r_dim * num_chunks * sizeof(ggml_fp16_t);
                const double sparse_kv_bytes = (double)rstate.topk_decode * bytes_per_tok;
                total_kv_traffic += kt_scan_bytes + sparse_kv_bytes;
            } else {
                total_kv_traffic += (double)cur_kv * bytes_per_tok;
            }
        }
        kv_traffic_bytes = total_kv_traffic / (double)n_decode_calls;
    }
    const double kv_traffic_mb_per_tok = kv_traffic_bytes / (1024.0 * 1024.0);
    const double total_traffic_bytes = (double)llama_model_size(model) + kv_traffic_bytes;
    const double effective_bandwidth_gbs = (ms_per_tok > 0.0) ?
        (total_traffic_bytes / ((ms_per_tok / 1000.0) * 1e9)) : 0.0;

    InferenceOutput out{};
    out.text = gen_text;
    out.ttft_ms = ttft_ms;
    out.prefill_ms = prefill_ms;
    out.evict_ms = evict_ms;
    out.decode_time_ms = decode_time_ms;
    out.compute_time_ms = compute_time_ms;
    out.ms_per_tok = ms_per_tok;
    out.tok_per_sec = tok_per_sec;
    out.peak_memory_mb = peak_memory_mb;
    out.kv_traffic_mb_per_tok = kv_traffic_mb_per_tok;
    out.effective_bandwidth_gbs = effective_bandwidth_gbs;
    out.prompt_tokens = n_tokens;
    out.kept_tokens = final_kept;
    out.decode_tokens = n_decode_calls;
    out.prompt_kv_mb = prompt_kv_mb;
    out.kt_cache_mb = kt_cache_mb;
    out.uncompressed_mb = uncompressed_mb;
    return out;
}

static int safe_stoi(const std::string & s, int default_val = 0) {
    try {
        return std::stoi(s);
    } catch (...) {
        return default_val;
    }
}

static nlohmann::json metrics_to_json(
    const InferenceOutput & res,
    const std::string & kv_type_str,
    bool enable_rocket,
    int token_budget
) {
    return {
        {"ttft_ms", res.ttft_ms},
        {"prefill_ms", res.prefill_ms},
        {"evict_ms", res.evict_ms},
        {"decode_time_ms", res.decode_time_ms},
        {"compute_time_ms", res.compute_time_ms},
        {"ms_per_token", res.ms_per_tok},
        {"tokens_per_sec", res.tok_per_sec},
        {"peak_memory_mb", res.peak_memory_mb},
        {"kv_traffic_mb_per_token", res.kv_traffic_mb_per_tok},
        {"effective_bandwidth_gbs", res.effective_bandwidth_gbs},
        {"prompt_tokens", res.prompt_tokens},
        {"kept_tokens", res.kept_tokens},
        {"decode_tokens", res.decode_tokens},
        {"kv_type", kv_type_str},
        {"rocket_enabled", enable_rocket},
        {"token_budget", token_budget},
        {"kv_cache_mb", res.prompt_kv_mb},
        {"kt_cache_mb", res.kt_cache_mb},
        {"uncompressed_kv_cache_mb", res.uncompressed_mb}
    };
}

int main(int argc, char ** argv) {
    // Suppress verbose internal llama/ggml/metal logs unless it's an error
    llama_log_set([](enum ggml_log_level level, const char * text, void * user_data) {
        (void)user_data;
        if (level == GGML_LOG_LEVEL_ERROR) {
            std::cerr << "[LLAMA ERROR] " << text;
        }
    }, nullptr);

    std::string model_path = "";
    std::string kv_type_str = "f16";
    int max_tokens = 128;
    int token_budget = 1024;
    bool enable_rocket = true;
    bool enable_stage2 = true;
    int n_ctx_override = 9216;
    int context_len_limit = 0;
    int window_size = 32;
    int kernel_size = 63;
    std::string batch_file = "";
    std::string output_file = "";

    // Argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--kv-type" && i + 1 < argc) kv_type_str = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = safe_stoi(argv[++i], max_tokens);
        else if ((arg == "--budget" || arg == "--token-budget") && i + 1 < argc) token_budget = safe_stoi(argv[++i], token_budget);
        else if (arg == "--rocket" && i + 1 < argc) {
            std::string v = argv[++i];
            enable_rocket = (v == "1" || v == "true");
        }
        else if (arg == "--stage2" && i + 1 < argc) {
            std::string v = argv[++i];
            enable_stage2 = (v == "1" || v == "true");
        }
        else if (arg == "--n-ctx" && i + 1 < argc) n_ctx_override = safe_stoi(argv[++i], n_ctx_override);
        else if (arg == "--context-len" && i + 1 < argc) context_len_limit = safe_stoi(argv[++i], 0);
        else if (arg == "--window-size" && i + 1 < argc) window_size = safe_stoi(argv[++i], 32);
        else if (arg == "--kernel-size" && i + 1 < argc) kernel_size = safe_stoi(argv[++i], 63);
        else if (arg == "--batch" && i + 1 < argc) batch_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg.rfind("--", 0) != 0 && model_path.empty()) {
            model_path = arg;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: Model path must be specified.\n";
        return 1;
    }

    if (batch_file.empty()) {
        std::cerr << "Error: Batch file must be specified with --batch <file>.\n";
        return 1;
    }

    if (enable_rocket) {
        std::cerr << "[RocketKV] In-graph ggml-metal backend active (Stage 1 GPU Compaction"
                  << (enable_stage2 ? " + Stage 2 Sparse FlashAttention" : " only") << ").\n";
    }

    ggml_type kv_type = get_kv_type(kv_type_str);

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Error: Failed to load model from " << model_path << "\n";
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_override;
    ctx_params.n_batch = n_ctx_override;
    ctx_params.n_ubatch = std::min(1024u, (uint32_t)ctx_params.n_batch);
    ctx_params.type_k = kv_type;
    ctx_params.type_v = kv_type;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Error: Failed to create context\n";
        llama_model_free(model);
        return 1;
    }

    // Metal pipeline pre-warmup: ensures Sample 1 TTFT & decode are NOT skewed by lazy shader compilation
    run_warmup(ctx, vocab, enable_rocket, enable_stage2);

    float bytes_per_elem = 2.0f;
    if (kv_type == GGML_TYPE_Q8_0) bytes_per_elem = 34.0f / 32.0f;
    else if (kv_type == GGML_TYPE_Q4_0) bytes_per_elem = 18.0f / 32.0f;
    else if (kv_type == GGML_TYPE_F32) bytes_per_elem = 4.0f;

    std::ifstream f(batch_file);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open batch file: " << batch_file << "\n";
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    nlohmann::json in_data;
    f >> in_data;
    nlohmann::json out_data = nlohmann::json::array();

    int total = in_data.size();

    for (int i = 0; i < total; ++i) {
        const auto & item = in_data[i];
        std::string prompt = item["prompt"];
        int req_max_tokens = item.value("max_tokens", max_tokens);

        InferenceOutput res = run_inference(
            ctx, model, vocab, ctx_params, prompt,
            req_max_tokens, token_budget, enable_rocket, enable_stage2,
            window_size, kernel_size, context_len_limit, bytes_per_elem
        );

        nlohmann::json item_out;
        item_out["sample_id"] = item.value("sample_id", i + 1);
        item_out["generated"] = res.text;
        item_out["metrics"] = metrics_to_json(res, kv_type_str, enable_rocket, token_budget);
        out_data.push_back(item_out);

        std::cerr << "    [" << (i + 1) << "/" << total << "] TTFT: "
                  << (int)res.ttft_ms << "ms (prefill: " << (int)res.prefill_ms
                  << "ms, evict: " << (int)res.evict_ms << "ms) | Compute: "
                  << (int)res.compute_time_ms << "ms | Decode: "
                  << res.tok_per_sec << " tok/s ("
                  << (int)res.effective_bandwidth_gbs << " GB/s) | Kept: "
                  << res.kept_tokens << "/" << res.prompt_tokens
                  << " | Peak Mem: " << (int)res.peak_memory_mb << " MB\n";
    }

    std::ofstream out_f(output_file.empty() ? "batch_results.json" : output_file);
    out_f << out_data.dump(2) << std::endl;
    std::cerr << "Batch completed successfully!" << "\n";

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
