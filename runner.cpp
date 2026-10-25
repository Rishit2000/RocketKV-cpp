#include "llama.h"
#include "rocket.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <nlohmann/json.hpp>

// Captured Q and K activations of the target layer during prefill
struct PrefillActivations {
    int total_prompt_tokens = 0;
    int target_layer = -1;
    int q_offset = 0;
    int k_offset = 0;
    std::vector<float> q; // [num_heads, total_prompt_tokens, dim_head]
    std::vector<float> k; // [num_heads_kv, total_prompt_tokens, dim_head]
    std::vector<float> raw_buf;
    std::vector<ggml_fp16_t> fp16_buf;
    bool enabled = false;
};

static PrefillActivations g_acts;

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;
    if (!g_acts.enabled) return false;

    const char * name = t->name;
    if (!name) return false;

    // Intercept post-RoPE Q and K tensors of the target layer
    bool is_q = (std::strncmp(name, "Qcur-", 5) == 0);
    bool is_k = (std::strncmp(name, "Kcur-", 5) == 0);
    if (!is_q && !is_k) return false;
    if (t->op != GGML_OP_ROPE) return false;

    int layer_idx = std::atoi(name + 5);
    if (g_acts.target_layer >= 0 && layer_idx != g_acts.target_layer) {
        return false;
    }

    if (ask) return true;

    int dim_head = t->ne[0];
    int n_heads = t->ne[1];
    int n_tokens_ubatch = t->ne[2];
    if (n_tokens_ubatch <= 1) return true; // Prefill only

    int total_tokens = g_acts.total_prompt_tokens;
    int n_elem = ggml_nelements(t);

    // Extract raw data from GGML tensor
    const float * src_data = nullptr;
    if (t->type == GGML_TYPE_F32) {
        if (g_acts.raw_buf.size() < (size_t)n_elem) g_acts.raw_buf.resize(n_elem);
        ggml_backend_tensor_get(t, g_acts.raw_buf.data(), 0, ggml_nbytes(t));
        src_data = g_acts.raw_buf.data();
    } else if (t->type == GGML_TYPE_F16) {
        if (g_acts.fp16_buf.size() < (size_t)n_elem) g_acts.fp16_buf.resize(n_elem);
        if (g_acts.raw_buf.size() < (size_t)n_elem) g_acts.raw_buf.resize(n_elem);
        ggml_backend_tensor_get(t, g_acts.fp16_buf.data(), 0, ggml_nbytes(t));
        ggml_fp16_to_fp32_row(g_acts.fp16_buf.data(), g_acts.raw_buf.data(), n_elem);
        src_data = g_acts.raw_buf.data();
    } else if (t->type == GGML_TYPE_BF16) {
        if (g_acts.fp16_buf.size() < (size_t)n_elem) g_acts.fp16_buf.resize(n_elem);
        if (g_acts.raw_buf.size() < (size_t)n_elem) g_acts.raw_buf.resize(n_elem);
        ggml_backend_tensor_get(t, g_acts.fp16_buf.data(), 0, ggml_nbytes(t));
        ggml_bf16_to_fp32_row((const ggml_bf16_t *)g_acts.fp16_buf.data(), g_acts.raw_buf.data(), n_elem);
        src_data = g_acts.raw_buf.data();
    } else {
        return false;
    }

    if (!src_data) return false;

    // Copy into contiguous layout [num_heads, total_prompt_tokens, dim_head]
    auto & dst_buf = is_q ? g_acts.q : g_acts.k;
    int & offset = is_q ? g_acts.q_offset : g_acts.k_offset;

    for (int tok = 0; tok < n_tokens_ubatch; ++tok) {
        int target_tok = offset + tok;
        if (target_tok >= total_tokens) break;
        for (int h = 0; h < n_heads; ++h) {
            const float * src = src_data + (tok * n_heads + h) * dim_head;
            float * dst = dst_buf.data() + (h * total_tokens + target_tok) * dim_head;
            std::memcpy(dst, src, dim_head * sizeof(float));
        }
    }
    offset += n_tokens_ubatch;
    return true;
}

static ggml_type get_kv_type(const std::string & type_str) {
    if (type_str == "q8_0") return GGML_TYPE_Q8_0;
    if (type_str == "q4_0") return GGML_TYPE_Q4_0;
    return GGML_TYPE_F16; 
}

struct InferenceOutput {
    std::string text;
    double ttft_ms = 0.0;       // User-facing TTFT: prefill + eviction
    double prefill_ms = 0.0;    // Pure Metal GPU prefill latency
    double evict_ms = 0.0;      // RocketKV Stage 1 eviction latency
    double ms_per_tok = 0.0;    // Pure decode step latency
    double tok_per_sec = 0.0;   // Generation throughput
    int prompt_tokens = 0;      // Prompt length after truncation
    int kept_tokens = 0;        // Surviving prompt tokens
    int decode_tokens = 0;      // Number of generated tokens
    double prompt_kv_mb = 0.0;  // Physical KV cache right after prefill/eviction
    double uncompressed_mb = 0.0; // Uncompressed full KV baseline
};

static void run_warmup(llama_context * ctx, const llama_vocab * vocab) {
    llama_token dummy_tokens[2] = { llama_vocab_bos(vocab), 1 };
    if (dummy_tokens[0] < 0) dummy_tokens[0] = 1;
    
    // Warm up prefill Metal kernel
    llama_batch batch_prefill = llama_batch_get_one(&dummy_tokens[0], 1);
    llama_decode(ctx, batch_prefill);
    llama_synchronize(ctx);

    // Warm up sampler pipeline
    struct llama_sampler * smpl = llama_sampler_init_greedy();
    llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_free(smpl);

    // Warm up autoregressive single-token decode Metal kernel
    llama_batch batch_decode = llama_batch_init(1, 0, 1);
    batch_decode.token[0] = dummy_tokens[1];
    batch_decode.pos[0] = 1;
    batch_decode.n_seq_id[0] = 1;
    batch_decode.seq_id[0][0] = 0;
    batch_decode.logits[0] = true;
    batch_decode.n_tokens = 1;
    llama_decode(ctx, batch_decode);
    llama_synchronize(ctx);
    llama_batch_free(batch_decode);

    // Reset KV cache to pristine empty state
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, -1, -1, -1);
}

static InferenceOutput run_inference(
    llama_context * ctx,
    llama_model * model,
    const llama_vocab * vocab,
    const llama_context_params & ctx_params,
    RocketConfig config,
    const std::string & prompt,
    int max_tokens,
    int token_budget,
    bool enable_rocket,
    int context_len_limit,
    float bytes_per_elem,
    bool stream_stdout
) {
    // Strict Isolation: ensure cache and activation buffers are completely clean
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, -1, -1, -1);
    g_acts.q.clear();
    g_acts.k.clear();
    g_acts.q_offset = 0;
    g_acts.k_offset = 0;

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

    // Compute budget parameters
    if (enable_rocket && token_budget > 0) {
        RocketBudgetParams budget_params = compute_rocket_params(token_budget, n_tokens, max_tokens, config.dim_head, config.window_size);
        config.prompt_budget = budget_params.prompt_budget;
        config.topk_decode = budget_params.k;
        config.compression_ratio = budget_params.compression_ratio;
        config.chunk_size = budget_params.chunk_size;
        config.r_dim = budget_params.r;
    } else {
        config.prompt_budget = n_tokens;
        config.topk_decode = n_tokens;
        config.compression_ratio = 1.0f;
        config.chunk_size = 0;
        config.r_dim = 0;
    }

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
    g_acts.target_layer = llama_model_n_layer(model) - 1;
    g_acts.total_prompt_tokens = n_tokens;
    g_acts.q_offset = 0;
    g_acts.k_offset = 0;
    if (enable_rocket) {
        g_acts.q.assign(config.num_heads * n_tokens * config.dim_head, 0.0f);
        g_acts.k.assign(config.num_heads_kv * n_tokens * config.dim_head, 0.0f);
    }
    g_acts.enabled = enable_rocket;

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

    // Disable activation interception during decode
    g_acts.enabled = false;

    // --- ROCKETKV STAGE 1 EVICTION ---
    int evicted_count = 0;
    double evict_ms = 0.0;
    if (enable_rocket && n_tokens > config.prompt_budget && !g_acts.k.empty()) {
        int64_t t_evict_start = ggml_time_us();

        std::vector<int> kept_indices = compute_prefill_eviction(
            g_acts.q.data(), g_acts.k.data(), n_tokens, n_tokens, config);

        std::vector<bool> keep_mask(n_tokens, false);
        for (int idx : kept_indices) {
            if (idx >= 0 && idx < n_tokens) {
                keep_mask[idx] = true;
            }
        }

        // Fast grouped contiguous range deletion
        int range_start = -1;
        for (int pos = 0; pos < n_tokens; ++pos) {
            if (!keep_mask[pos]) {
                evicted_count++;
                if (range_start < 0) range_start = pos;
            } else {
                if (range_start >= 0) {
                    llama_memory_seq_rm(mem, 0, range_start, pos);
                    range_start = -1;
                }
            }
        }
        if (range_start >= 0) {
            llama_memory_seq_rm(mem, 0, range_start, n_tokens);
        }

        int64_t t_evict_end = ggml_time_us();
        evict_ms = (t_evict_end - t_evict_start) / 1000.0;
    }

    // Release prefill activation buffers
    g_acts.q.clear();
    g_acts.k.clear();

    double ttft_ms = prefill_ms + evict_ms;

    // --- DECODE PHASE ---
    struct llama_sampler * smpl = llama_sampler_init_greedy();
    int n_cur = n_tokens;
    int n_decode_steps = 0;
    std::string gen_text = "";
    llama_batch batch_decode = llama_batch_init(1, 0, 1);

    // Sample the initial token generated from the prefill prompt
    llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

    int64_t t_start_decode = ggml_time_us();

    while (n_cur < ctx_params.n_ctx && n_decode_steps < max_tokens) {
        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char piece_buf[256];
        int n_piece = llama_token_to_piece(vocab, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (n_piece > 0) {
            gen_text.append(piece_buf, n_piece);
            if (stream_stdout) {
                std::cout.write(piece_buf, n_piece);
            }
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
        llama_synchronize(ctx);

        n_cur++;
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
    }

    int64_t t_end_decode = ggml_time_us();

    if (stream_stdout) {
        std::cout.flush();
    }

    llama_batch_free(batch_decode);
    llama_sampler_free(smpl);

    double decode_time_ms = (t_end_decode - t_start_decode) / 1000.0;
    double ms_per_tok = (n_decode_steps > 0) ? (decode_time_ms / n_decode_steps) : 0.0;
    double tok_per_sec = (decode_time_ms > 0) ? (n_decode_steps / (decode_time_ms / 1000.0)) : 0.0;

    int model_n_layer = llama_model_n_layer(model);
    int model_n_head_kv = llama_model_n_head_kv(model);
    double bytes_per_tok = 2.0 * model_n_layer * model_n_head_kv * config.dim_head * bytes_per_elem;
    int final_kept = n_tokens - evicted_count;
    double uncompressed_mb = (n_tokens * bytes_per_tok) / (1024.0 * 1024.0);
    double prompt_kv_mb = (final_kept * bytes_per_tok) / (1024.0 * 1024.0);

    // Clean KV cache
    llama_memory_seq_rm(mem, -1, -1, -1);

    InferenceOutput out{};
    out.text = gen_text;
    out.ttft_ms = ttft_ms;
    out.prefill_ms = prefill_ms;
    out.evict_ms = evict_ms;
    out.ms_per_tok = ms_per_tok;
    out.tok_per_sec = tok_per_sec;
    out.prompt_tokens = n_tokens;
    out.kept_tokens = final_kept;
    out.decode_tokens = n_decode_steps;
    out.prompt_kv_mb = prompt_kv_mb;
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
        {"ms_per_token", res.ms_per_tok},
        {"tokens_per_sec", res.tok_per_sec},
        {"prompt_tokens", res.prompt_tokens},
        {"kept_tokens", res.kept_tokens},
        {"decode_tokens", res.decode_tokens},
        {"kv_type", kv_type_str},
        {"rocket_enabled", enable_rocket},
        {"token_budget", token_budget},
        {"kv_cache_mb", res.prompt_kv_mb},
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

    if (enable_rocket) {
        ctx_params.cb_eval = eval_callback;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Error: Failed to create context\n";
        llama_model_free(model);
        return 1;
    }

    // Metal pipeline pre-warmup: ensures Sample 1 TTFT is NOT skewed by lazy shader compilation
    run_warmup(ctx, vocab);

    int model_n_head = llama_model_n_head(model);
    int model_n_head_kv = llama_model_n_head_kv(model);
    int model_n_embd = llama_model_n_embd(model);
    int model_dim_head = model_n_embd / model_n_head;

    RocketConfig config{};
    config.num_heads = model_n_head;
    config.num_heads_kv = model_n_head_kv;
    config.dim_head = model_dim_head;
    config.window_size = window_size;
    config.kernel_size = kernel_size;

    float bytes_per_elem = 2.0f;
    if (kv_type == GGML_TYPE_Q8_0) bytes_per_elem = 34.0f / 32.0f;
    else if (kv_type == GGML_TYPE_Q4_0) bytes_per_elem = 18.0f / 32.0f;
    else if (kv_type == GGML_TYPE_F32) bytes_per_elem = 4.0f;

    if (!batch_file.empty()) {
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
        //std::cerr << "\n>>> Processing batch of " << total << " samples (" << kv_type_str << ", rocket=" << (enable_rocket ? "1" : "0") << ") <<<\n";

        for (int i = 0; i < total; ++i) {
            const auto & item = in_data[i];
            std::string prompt = item["prompt"];
            int req_max_tokens = item.value("max_tokens", max_tokens);

            InferenceOutput res = run_inference(
                ctx, model, vocab, ctx_params, config, prompt,
                req_max_tokens, token_budget, enable_rocket,
                context_len_limit, bytes_per_elem, false
            );

            nlohmann::json item_out;
            item_out["sample_id"] = item.value("sample_id", i + 1);
            item_out["generated"] = res.text;
            item_out["metrics"] = metrics_to_json(res, kv_type_str, enable_rocket, token_budget);
            out_data.push_back(item_out);

            std::cerr << "    [" << (i + 1) << "/" << total << "] TTFT: "
                      << (int)res.ttft_ms << "ms (prefill: " << (int)res.prefill_ms
                      << "ms, evict: " << (int)res.evict_ms << "ms) | Decode: "
                      << res.tok_per_sec << " tok/s | Kept: "
                      << res.kept_tokens << "/" << res.prompt_tokens << "\n";
        }

        std::ofstream out_f(output_file.empty() ? "batch_results.json" : output_file);
        out_f << out_data.dump(2) << std::endl;
        std::cerr << "Batch completed successfully!" << "\n";
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
