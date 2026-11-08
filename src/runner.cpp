#include "llama.h"
#include "rocket.h"
#include "rocket_adapter.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
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
    std::vector<unsigned char> raw_buf;
    int dim_head = 0;
    int num_heads = 0;
    int num_heads_kv = 0;
    std::string error;
    bool enabled = false;
    bool capture_decode = false;
    std::vector<std::vector<float>> attention;
};

static PrefillActivations g_acts;

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;
    if (!g_acts.enabled && !g_acts.capture_decode) return false;

    const char * name = t->name;
    if (!name) return false;
    if (g_acts.capture_decode) {
        if (std::strncmp(name, "kqv_out-", 8) != 0) return false;
        if (ask) return true;
        const int layer = std::atoi(name + 8);
        if (layer < 0 || layer >= (int)g_acts.attention.size() || t->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(t) || ggml_nelements(t) != (int64_t)g_acts.num_heads * g_acts.dim_head) {
            g_acts.error = "Unsupported decode attention trace tensor";
            return false;
        }
        try {
            auto& output = g_acts.attention[layer];
            output.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, output.data(), 0, ggml_nbytes(t));
        } catch (const std::exception& error) {
            g_acts.error = error.what();
            return false;
        }
        return true;
    }

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

    const int dim_head = g_acts.dim_head;
    const int n_heads = is_q ? g_acts.num_heads : g_acts.num_heads_kv;
    const int total_tokens = g_acts.total_prompt_tokens;
    auto & dst_buf = is_q ? g_acts.q : g_acts.k;
    int & offset = is_q ? g_acts.q_offset : g_acts.k_offset;
    if (t->ne[0] != dim_head || t->ne[1] != n_heads || t->ne[3] != 1 ||
        t->ne[2] <= 0 || t->ne[2] > total_tokens - offset ||
        (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16)) {
        g_acts.error = "Unsupported or incomplete prefill Q/K capture";
        return false;
    }
    try {
        g_acts.raw_buf.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, g_acts.raw_buf.data(), 0, g_acts.raw_buf.size());
        for (int tok = 0; tok < t->ne[2]; ++tok) {
            for (int h = 0; h < n_heads; ++h) {
                for (int d = 0; d < dim_head; ++d) {
                    const auto* src = g_acts.raw_buf.data() + tok * t->nb[2] + h * t->nb[1] + d * t->nb[0];
                    float value;
                    if (t->type == GGML_TYPE_F32) {
                        std::memcpy(&value, src, sizeof(value));
                    } else if (t->type == GGML_TYPE_F16) {
                        ggml_fp16_t half;
                        std::memcpy(&half, src, sizeof(half));
                        value = ggml_fp16_to_fp32(half);
                    } else {
                        ggml_bf16_t half;
                        std::memcpy(&half, src, sizeof(half));
                        value = ggml_bf16_to_fp32(half);
                    }
                    dst_buf[((size_t)h * total_tokens + offset + tok) * dim_head + d] = value;
                }
            }
        }
    } catch (const std::exception& error) {
        g_acts.error = error.what();
        return false;
    }
    offset += (int)t->ne[2];
    return true;
}

static ggml_type get_kv_type(const std::string & type_str) {
    if (type_str == "q8_0") return GGML_TYPE_Q8_0;
    if (type_str == "q4_0") return GGML_TYPE_Q4_0;
    if (type_str == "f16") return GGML_TYPE_F16;
    throw std::invalid_argument("Unsupported --kv-type: " + type_str);
}

enum class RocketMode { Off, Stage1, Hsa };

static const char* mode_name(RocketMode mode) {
    if (mode == RocketMode::Hsa) return "hsa";
    return mode == RocketMode::Stage1 ? "stage1" : "off";
}

struct BatchGuard {
    llama_batch batch;
    explicit BatchGuard(int count) : batch(llama_batch_init(count, 0, 1)) {}
    ~BatchGuard() { llama_batch_free(batch); }
    BatchGuard(const BatchGuard&) = delete;
    BatchGuard& operator=(const BatchGuard&) = delete;
};

struct RequestGuard {
    llama_context* ctx;
    bool finished = false;
    ~RequestGuard() {
        if (finished) return;
        g_acts.enabled = false;
        g_acts.capture_decode = false;
        llama_set_rocket_attention(ctx, nullptr, nullptr);
        llama_rocket_set_f32_reference(ctx, false);
        if (!llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1)) {
            std::cerr << "Error: failed to clear request KV cache\n";
        }
    }
};

static void clear_kv(llama_context* ctx) {
    if (!llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1)) {
        throw std::runtime_error("Failed to clear KV cache");
    }
}

struct InferenceOutput {
    std::string text;
    double ttft_ms = 0.0;       // User-facing TTFT: prefill + eviction
    double prefill_ms = 0.0;    // Prefill including activation capture
    double evict_ms = 0.0;      // RocketKV Stage 1 eviction latency
    double ms_per_tok = 0.0;    // Legacy generation-wall / output-token metric
    double tok_per_sec = 0.0;   // Generation throughput
    int prompt_tokens = 0;      // Prompt length after truncation
    int kept_tokens = 0;        // Surviving prompt tokens
    int decode_tokens = 0;      // Number of generated tokens
    double prompt_kv_mb = 0.0;  // Logical prompt KV estimate, not allocated memory
    double uncompressed_mb = 0.0; // Uncompressed full KV baseline
    double stage2_init_ms = 0.0;
    double decode_forward_ms = 0.0;
    int decode_forward_calls = 0;
    int model_layers = 0;
    RocketConfig config{};
    RocketStage2Stats stage2;
    std::vector<llama_token> token_ids;
    std::vector<std::vector<float>> logits;
    std::vector<std::vector<std::vector<float>>> attention;
    std::vector<int> kept_positions;
    bool native_f32_reference = false;
};

static void run_warmup(llama_context * ctx, const llama_vocab * vocab) {
    llama_token dummy_tokens[2] = { llama_vocab_bos(vocab), 1 };
    if (dummy_tokens[0] < 0) dummy_tokens[0] = 1;
    
    // Warm up prefill Metal kernel
    llama_batch batch_prefill = llama_batch_get_one(&dummy_tokens[0], 1);
    if (llama_decode(ctx, batch_prefill) != 0) throw std::runtime_error("Prefill warmup failed");
    llama_synchronize(ctx);

    // Warm up sampler pipeline
    struct llama_sampler * smpl = llama_sampler_init_greedy();
    llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_free(smpl);

    // Warm up autoregressive single-token decode Metal kernel
    BatchGuard batch_owner(1);
    auto& batch_decode = batch_owner.batch;
    batch_decode.token[0] = dummy_tokens[1];
    batch_decode.pos[0] = 1;
    batch_decode.n_seq_id[0] = 1;
    batch_decode.seq_id[0][0] = 0;
    batch_decode.logits[0] = true;
    batch_decode.n_tokens = 1;
    if (llama_decode(ctx, batch_decode) != 0) throw std::runtime_error("Decode warmup failed");
    llama_synchronize(ctx);

    // Reset KV cache to pristine empty state
    clear_kv(ctx);
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
    RocketMode mode,
    RocketAdapter* adapter,
    bool full_attention,
    bool trace_logits,
    bool native_f32_reference,
    const std::vector<llama_token>& forced_tokens,
    int context_len_limit,
    float bytes_per_elem,
    bool stream_stdout
) {
    const bool enable_rocket = mode != RocketMode::Off;
    const bool enable_hsa = mode == RocketMode::Hsa;
    if (max_tokens < 1 || (enable_hsa && (!adapter || token_budget <= 0))) {
        throw std::invalid_argument("Invalid generation length or HSA configuration");
    }
    if (!forced_tokens.empty()) {
        if (!trace_logits || forced_tokens.size() != (size_t)max_tokens) {
            throw std::invalid_argument("forced_tokens requires --trace-logits and max_tokens entries");
        }
        for (auto token : forced_tokens) {
            if (token < 0 || token >= llama_vocab_n_tokens(vocab) || llama_vocab_is_eog(vocab, token)) {
                throw std::invalid_argument("Invalid or EOG forced token");
            }
        }
    }
    RequestGuard request_guard{ctx};
    InferenceOutput out;
    out.native_f32_reference = native_f32_reference;
    if (adapter) adapter->reset();
    llama_rocket_set_f32_reference(ctx, false);
    // Strict Isolation: ensure cache and activation buffers are completely clean
    llama_memory_t mem = llama_get_memory(ctx);
    clear_kv(ctx);
    g_acts.enabled = false;
    g_acts.capture_decode = false;
    g_acts.error.clear();
    g_acts.q.clear();
    g_acts.k.clear();
    g_acts.q_offset = 0;
    g_acts.k_offset = 0;

    // Tokenize prompt
    if (prompt.size() > INT_MAX - 32) throw std::invalid_argument("Prompt is too large");
    std::vector<llama_token> tokens(prompt.length() + 32);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    }
    if (n_tokens <= 0) throw std::runtime_error("Tokenization failed or produced no tokens");
    tokens.resize(n_tokens);

    // Official LongBench middle-truncation convention
    if (context_len_limit > 0 && n_tokens > context_len_limit) {
        int half = context_len_limit / 2;
        std::vector<llama_token> truncated;
        truncated.reserve(context_len_limit);
        truncated.insert(truncated.end(), tokens.begin(), tokens.begin() + half);
        truncated.insert(truncated.end(), tokens.end() - (context_len_limit - half), tokens.end());
        tokens = std::move(truncated);
        n_tokens = (int)tokens.size();
    }
    if ((int64_t)n_tokens + max_tokens > ctx_params.n_ctx) {
        throw std::invalid_argument("Prompt plus max_tokens exceeds --n-ctx; set --context-len or increase --n-ctx");
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

    BatchGuard prefill_owner(n_tokens);
    auto& batch = prefill_owner.batch;
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
    g_acts.dim_head = config.dim_head;
    g_acts.num_heads = config.num_heads;
    g_acts.num_heads_kv = config.num_heads_kv;
    if (enable_rocket) {
        g_acts.q.assign((size_t)config.num_heads * n_tokens * config.dim_head, 0.0f);
        g_acts.k.assign((size_t)config.num_heads_kv * n_tokens * config.dim_head, 0.0f);
    }
    g_acts.enabled = enable_rocket;

    int64_t t_start_prefill = ggml_time_us();
    if (llama_decode(ctx, batch) != 0) {
        throw std::runtime_error("Prefill failed: " + g_acts.error);
    }
    llama_synchronize(ctx);
    int64_t t_end_prefill = ggml_time_us();
    double prefill_ms = (t_end_prefill - t_start_prefill) / 1000.0;

    // Disable activation interception during decode
    g_acts.enabled = false;
    if (enable_rocket && (!g_acts.error.empty() || g_acts.q_offset != n_tokens || g_acts.k_offset != n_tokens)) {
        throw std::runtime_error("Incomplete prefill Q/K capture: " + g_acts.error);
    }

    // --- ROCKETKV STAGE 1 EVICTION ---
    int evicted_count = 0;
    double evict_ms = 0.0;
    std::vector<int> kept_positions(n_tokens);
    std::iota(kept_positions.begin(), kept_positions.end(), 0);
    if (enable_rocket && n_tokens > config.prompt_budget) {
        int64_t t_evict_start = ggml_time_us();

        kept_positions = compute_prefill_eviction(
            g_acts.q.data(), g_acts.k.data(), n_tokens, n_tokens, config);

        std::vector<bool> keep_mask(n_tokens, false);
        for (int idx : kept_positions) {
            if (idx < 0 || idx >= n_tokens || keep_mask[idx]) throw std::runtime_error("Invalid Stage 1 selection");
            keep_mask[idx] = true;
        }

        // Fast grouped contiguous range deletion
        int range_start = -1;
        for (int pos = 0; pos < n_tokens; ++pos) {
            if (!keep_mask[pos]) {
                evicted_count++;
                if (range_start < 0) range_start = pos;
            } else {
                if (range_start >= 0) {
                    if (!llama_memory_seq_rm(mem, 0, range_start, pos)) throw std::runtime_error("Stage 1 deletion failed");
                    range_start = -1;
                }
            }
        }
        if (range_start >= 0) {
            if (!llama_memory_seq_rm(mem, 0, range_start, n_tokens)) throw std::runtime_error("Stage 1 deletion failed");
        }

        int64_t t_evict_end = ggml_time_us();
        evict_ms = (t_evict_end - t_evict_start) / 1000.0;
    }

    if (enable_hsa) {
        const auto start = ggml_time_us();
        adapter->initialize(config, kept_positions, full_attention);
        llama_synchronize(ctx);
        out.stage2_init_ms = (ggml_time_us() - start) / 1000.0;
    }
    if (trace_logits) out.kept_positions = kept_positions;
    llama_rocket_set_f32_reference(ctx, native_f32_reference);

    // Release prefill activation buffers
    g_acts.q.clear();
    g_acts.k.clear();

    double ttft_ms = prefill_ms + evict_ms + out.stage2_init_ms;

    // --- DECODE PHASE ---
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
    auto* smpl = sampler.get();
    int n_cur = n_tokens;
    int n_decode_steps = 0;
    std::string gen_text = "";
    BatchGuard decode_owner(1);
    auto& batch_decode = decode_owner.batch;

    auto record_logits = [&]() {
        if (!trace_logits) return;
        const float* logits = llama_get_logits_ith(ctx, -1);
        const int count = llama_vocab_n_tokens(vocab);
        if (!logits) throw std::runtime_error("Missing logits");
        for (int i = 0; i < count; ++i) {
            if (!std::isfinite(logits[i])) throw std::runtime_error("Non-finite model logits");
        }
        out.logits.emplace_back(logits, logits + count);
    };

    // Sample the initial token generated from the prefill prompt
    record_logits();
    llama_token new_token_id = forced_tokens.empty() ? llama_sampler_sample(smpl, ctx, -1) : forced_tokens[0];

    int64_t t_start_decode = ggml_time_us();

    while ((int64_t)n_cur < ctx_params.n_ctx && n_decode_steps < max_tokens) {
        if (llama_vocab_is_eog(vocab, new_token_id)) break;
        if (trace_logits) out.token_ids.push_back(new_token_id);

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

        const auto forward_start = ggml_time_us();
        g_acts.capture_decode = trace_logits;
        if (trace_logits) g_acts.attention.assign(llama_model_n_layer(model), {});
        if (enable_hsa) adapter->begin_step(n_cur);
        if (llama_decode(ctx, batch_decode) != 0) throw std::runtime_error("Decode failed at position " + std::to_string(n_cur));
        llama_synchronize(ctx);
        if (enable_hsa) adapter->finish_step();
        if (!g_acts.error.empty()) throw std::runtime_error(g_acts.error);
        out.decode_forward_ms += (ggml_time_us() - forward_start) / 1000.0;
        ++out.decode_forward_calls;
        if (trace_logits) {
            for (const auto& layer_output : g_acts.attention) {
                if (layer_output.empty()) throw std::runtime_error("Missing decode attention trace");
            }
            out.attention.push_back(g_acts.attention);
        }

        n_cur++;
        record_logits();
        new_token_id = forced_tokens.empty() ? llama_sampler_sample(smpl, ctx, -1) : forced_tokens[n_decode_steps];
    }

    int64_t t_end_decode = ggml_time_us();

    if (stream_stdout) {
        std::cout.flush();
    }

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
    if (enable_hsa) {
        out.stage2 = adapter->stats();
        adapter->reset();
    }
    clear_kv(ctx);
    g_acts.capture_decode = false;
    llama_rocket_set_f32_reference(ctx, false);
    request_guard.finished = true;

    out.config = config;
    out.model_layers = model_n_layer;
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

static int parse_integer(const std::string& value) {
    size_t consumed = 0;
    const int result = std::stoi(value, &consumed);
    if (consumed != value.size()) throw std::invalid_argument("Invalid integer: " + value);
    return result;
}

static nlohmann::json metrics_to_json(
    const InferenceOutput & res,
    const std::string & kv_type_str,
    RocketMode mode,
    int token_budget,
    bool full_attention,
    const std::string& stage2_backend
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
        {"rocket_enabled", mode != RocketMode::Off},
        {"token_budget", token_budget},
        {"kv_cache_mb", res.prompt_kv_mb},
        {"uncompressed_kv_cache_mb", res.uncompressed_mb},
        {"kv_cache_measurement", "logical-estimate"},
        {"requested_mode", mode_name(mode)},
        {"effective_mode", mode_name(mode)},
        {"stage1_policy", mode == RocketMode::Off ? "none" : "legacy-global"},
        {"stage2_backend", mode == RocketMode::Hsa ? stage2_backend : "none"},
        {"hsa_full_attention", full_attention},
        {"native_f32_reference", res.native_f32_reference},
        {"stage2_init_ms", res.stage2_init_ms},
        {"stage2_layer_calls", res.stage2.layer_calls},
        {"stage2_calls_per_layer", res.stage2.calls_per_layer},
        {"stage2_candidate_min", res.stage2.candidate_min},
        {"stage2_candidate_max", res.stage2.candidate_max},
        {"stage2_selected_min", res.stage2.selected_min},
        {"stage2_selected_max", res.stage2.selected_max},
        {"stage2_k", res.config.topk_decode},
        {"stage2_r", res.config.r_dim},
        {"stage2_chunk", res.config.chunk_size},
        {"model_layers", res.model_layers},
        {"decode_forward_calls", res.decode_forward_calls},
        {"decode_forward_ms", res.decode_forward_ms},
        {"decode_ms_per_forward", res.decode_forward_calls > 0
            ? nlohmann::json(res.decode_forward_ms / res.decode_forward_calls) : nlohmann::json(nullptr)}
    };
}

static int run_main(int argc, char ** argv) {
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
    RocketMode mode = RocketMode::Stage1;
    bool legacy_mode_seen = false;
    bool mode_seen = false;
    bool full_attention = false;
    bool trace_logits = false;
    bool native_f32_reference = false;
    int gpu_layers = -2;
    int threads = 4;
    int ubatch = 1024;
    std::string stage2_backend = "cpu-reference";
    int n_ctx_override = 9216;
    int context_len_limit = 0;
    int window_size = 32;
    int kernel_size = 63;
    std::string batch_file = "";
    std::string output_file = "";

    // Argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--help") {
            std::cout << "Usage: runner --model MODEL --batch INPUT.json [--output OUTPUT.json]\n"
                      << "  --rocket-mode off|stage1|hsa  (default: stage1)\n"
                      << "  --rocket 0|1                 legacy mode; do not combine with --rocket-mode\n"
                      << "  --budget N --kv-type f16|q8_0|q4_0 --max-tokens N\n"
                      << "  --n-ctx N --context-len N --window-size N --kernel-size N\n"
                      << "  --threads N --ubatch N --gpu-layers N\n"
                      << "  HSA model placement: --gpu-layers 0 for CPU; positive values or -1 for Metal\n"
                      << "  --stage2-backend cpu-reference|metal (F16 KV; default cpu-reference)\n"
                      << "  --hsa-full                   diagnostic: select all live candidates\n"
                      << "  --reference-f32              diagnostic native F32 decode, requires --trace-logits\n"
                      << "  --trace-logits               include token IDs/logits; allows batch forced_tokens\n";
            return 0;
        }
        if (arg == "--model") model_path = value();
        else if (arg == "--kv-type") kv_type_str = value();
        else if (arg == "--max-tokens") max_tokens = parse_integer(value());
        else if (arg == "--budget" || arg == "--token-budget") token_budget = parse_integer(value());
        else if (arg == "--rocket") {
            const auto v = value();
            if (v != "0" && v != "1" && v != "false" && v != "true") throw std::invalid_argument("Invalid --rocket value");
            mode = (v == "1" || v == "true") ? RocketMode::Stage1 : RocketMode::Off;
            legacy_mode_seen = true;
        }
        else if (arg == "--rocket-mode") {
            const auto v = value();
            if (v == "off") mode = RocketMode::Off;
            else if (v == "stage1") mode = RocketMode::Stage1;
            else if (v == "hsa") mode = RocketMode::Hsa;
            else throw std::invalid_argument("Invalid --rocket-mode: " + v);
            mode_seen = true;
        }
        else if (arg == "--n-ctx") n_ctx_override = parse_integer(value());
        else if (arg == "--context-len") context_len_limit = parse_integer(value());
        else if (arg == "--window-size") window_size = parse_integer(value());
        else if (arg == "--kernel-size") kernel_size = parse_integer(value());
        else if (arg == "--batch") batch_file = value();
        else if (arg == "--output") output_file = value();
        else if (arg == "--gpu-layers") gpu_layers = parse_integer(value());
        else if (arg == "--threads") threads = parse_integer(value());
        else if (arg == "--ubatch") ubatch = parse_integer(value());
        else if (arg == "--stage2-backend") stage2_backend = value();
        else if (arg == "--hsa-full") full_attention = true;
        else if (arg == "--trace-logits") trace_logits = true;
        else if (arg == "--reference-f32") native_f32_reference = true;
        else if (arg.rfind("--", 0) != 0 && model_path.empty()) {
            model_path = arg;
        }
        else throw std::invalid_argument("Unknown argument: " + arg);
    }

    if (legacy_mode_seen && mode_seen) throw std::invalid_argument("Do not combine --rocket and --rocket-mode");
    if (model_path.empty() || batch_file.empty()) throw std::invalid_argument("--model and --batch are required");
    if (max_tokens < 1 || token_budget < 0 || n_ctx_override < 1 || context_len_limit < 0 ||
        context_len_limit > n_ctx_override || window_size < 1 || kernel_size < 1 ||
        kernel_size % 2 == 0 || threads < 1 || ubatch < 1 || gpu_layers < -2) {
        throw std::invalid_argument("Invalid budget, context, window, thread, or batch size");
    }
    if (stage2_backend != "cpu-reference" && stage2_backend != "metal") throw std::invalid_argument("Unknown stage2 backend");
    if (stage2_backend == "metal" && mode != RocketMode::Hsa) throw std::invalid_argument("Metal stage2 backend requires --rocket-mode hsa");
    if (full_attention && mode != RocketMode::Hsa) throw std::invalid_argument("--hsa-full requires --rocket-mode hsa");
    if (mode == RocketMode::Hsa) {
        if (kv_type_str != "f16" || token_budget < 1) {
            throw std::invalid_argument("HSA requires F16 KV and positive budget");
        }
        if (gpu_layers == -2) gpu_layers = stage2_backend == "metal" ? -1 : 0;
    }
    if (native_f32_reference) {
        if (!trace_logits || mode == RocketMode::Hsa || kv_type_str != "f16" ||
            (gpu_layers != -2 && gpu_layers != 0)) {
            throw std::invalid_argument("--reference-f32 requires traced native attention, F16 KV, and CPU");
        }
        gpu_layers = 0;
    }
    const bool enable_rocket = mode != RocketMode::Off;

    ggml_type kv_type = get_kv_type(kv_type_str);

    struct BackendGuard {
        BackendGuard() { llama_backend_init(); }
        ~BackendGuard() { llama_backend_free(); }
    } backend_guard;
    llama_model_params model_params = llama_model_default_params();
    ggml_backend_dev_t metal_devices[] = {nullptr, nullptr};
    const bool offloaded_hsa = mode == RocketMode::Hsa && gpu_layers != 0;
    if (offloaded_hsa) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto device = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(device);
            if ((type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
                std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "MTL") == 0) {
                metal_devices[0] = device;
                break;
            }
        }
        if (!metal_devices[0]) {
            throw std::invalid_argument("HSA model offload requires an available Metal GPU; use --gpu-layers 0 for CPU");
        }
        model_params.devices = metal_devices;
        std::cerr << "HSA model offload on " << ggml_backend_dev_description(metal_devices[0])
                  << " (Metal), HSA backend: " << stage2_backend << "\n";
    }
    if (gpu_layers != -2) model_params.n_gpu_layers = gpu_layers;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_owner(
        llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);
    auto* model = model_owner.get();
    if (!model) throw std::runtime_error("Failed to load model: " + model_path);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_override; 
    ctx_params.n_batch = n_ctx_override;
    ctx_params.n_ubatch = std::min((uint32_t)ubatch, ctx_params.n_batch);
    ctx_params.n_threads = threads;
    ctx_params.n_threads_batch = threads;
    ctx_params.type_k = kv_type;
    ctx_params.type_v = kv_type;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (gpu_layers == 0) {
        ctx_params.op_offload = false;
        ctx_params.offload_kqv = false;
    }

    ctx_params.cb_eval = (enable_rocket || trace_logits) ? eval_callback : nullptr;

    std::unique_ptr<llama_context, decltype(&llama_free)> ctx_owner(llama_init_from_model(model, ctx_params), llama_free);
    auto* ctx = ctx_owner.get();
    if (!ctx) throw std::runtime_error("Failed to create context");

    // Metal pipeline pre-warmup: ensures Sample 1 TTFT is NOT skewed by lazy shader compilation
    run_warmup(ctx, vocab);

    int model_n_head = llama_model_n_head(model);
    int model_n_head_kv = llama_model_n_head_kv(model);
    int model_n_embd = llama_model_n_embd(model);
    if (model_n_head <= 0 || model_n_head_kv <= 0 || model_n_embd % model_n_head != 0) {
        throw std::runtime_error("Unsupported model head dimensions");
    }
    int model_dim_head = model_n_embd / model_n_head;

    RocketConfig config{};
    config.num_heads = model_n_head;
    config.num_heads_kv = model_n_head_kv;
    config.dim_head = model_dim_head;
    config.window_size = window_size;
    config.kernel_size = kernel_size;
    std::unique_ptr<RocketAdapter> adapter;
    if (mode == RocketMode::Hsa) adapter = std::make_unique<RocketAdapter>(ctx, llama_model_n_layer(model), stage2_backend == "metal");

    float bytes_per_elem = 2.0f;
    if (kv_type == GGML_TYPE_Q8_0) bytes_per_elem = 34.0f / 32.0f;
    else if (kv_type == GGML_TYPE_Q4_0) bytes_per_elem = 18.0f / 32.0f;
    else if (kv_type == GGML_TYPE_F32) bytes_per_elem = 4.0f;

    if (!batch_file.empty()) {
        std::ifstream f(batch_file);
        if (!f.is_open()) throw std::runtime_error("Could not open batch file: " + batch_file);
        nlohmann::json in_data;
        f >> in_data;
        if (!in_data.is_array() || in_data.empty()) throw std::invalid_argument("Batch input must be a non-empty JSON array");
        nlohmann::json out_data = nlohmann::json::array();

        int total = in_data.size();
        //std::cerr << "\n>>> Processing batch of " << total << " samples (" << kv_type_str << ", rocket=" << (enable_rocket ? "1" : "0") << ") <<<\n";

        for (int i = 0; i < total; ++i) {
            const auto & item = in_data[i];
            std::string prompt = item["prompt"];
            int req_max_tokens = item.value("max_tokens", max_tokens);
            const auto forced_tokens = item.value("forced_tokens", std::vector<llama_token>{});

            InferenceOutput res = run_inference(
                ctx, model, vocab, ctx_params, config, prompt,
                req_max_tokens, token_budget, mode, adapter.get(), full_attention, trace_logits, native_f32_reference, forced_tokens,
                context_len_limit, bytes_per_elem, false
            );

            nlohmann::json item_out;
            item_out["sample_id"] = item.value("sample_id", i + 1);
            item_out["generated"] = res.text;
            item_out["status"] = "ok";
            item_out["metrics"] = metrics_to_json(res, kv_type_str, mode, token_budget, full_attention, stage2_backend);
            item_out["metrics"]["requested_gpu_layers"] = gpu_layers == -2
                ? nlohmann::json(nullptr) : nlohmann::json(gpu_layers);
            item_out["metrics"]["model_backend"] = offloaded_hsa ? "metal" : (gpu_layers == 0 ? "cpu" : "auto");
            item_out["metrics"]["model_gpu_device"] = offloaded_hsa
                ? nlohmann::json(ggml_backend_dev_description(metal_devices[0])) : nlohmann::json(nullptr);
            if (trace_logits) {
                item_out["token_ids"] = res.token_ids;
                item_out["logits"] = res.logits;
                item_out["attention"] = res.attention;
                item_out["kept_positions"] = res.kept_positions;
            }
            out_data.push_back(item_out);

            std::cerr << "    [" << (i + 1) << "/" << total << "] TTFT: "
                      << (int)res.ttft_ms << "ms (prefill: " << (int)res.prefill_ms
                      << "ms, evict: " << (int)res.evict_ms << "ms) | Decode: "
                      << res.tok_per_sec << " tok/s | Kept: "
                      << res.kept_tokens << "/" << res.prompt_tokens << "\n";
        }

        std::ofstream out_f(output_file.empty() ? "batch_results.json" : output_file);
        if (!out_f.is_open()) throw std::runtime_error("Could not open output file");
        out_f << out_data.dump(2) << std::endl;
        if (!out_f) throw std::runtime_error("Could not write output file");
        std::cerr << "Batch completed successfully!" << "\n";
    }

    return 0;
}

int main(int argc, char** argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
