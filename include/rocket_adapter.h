#pragma once

#include "llama-rocket.h"
#include "rocket.h"
#include "rocket_metal.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct RocketStage2Stats {
    int64_t layer_calls = 0;
    int candidate_min = 0;
    int candidate_max = 0;
    int selected_min = 0;
    int selected_max = 0;
    std::vector<int64_t> calls_per_layer;
};

class RocketAdapter {
public:
    RocketAdapter(llama_context* ctx, int layer_count, bool metal = false);
    ~RocketAdapter();
    RocketAdapter(const RocketAdapter&) = delete;
    RocketAdapter& operator=(const RocketAdapter&) = delete;

    void reset();
    void initialize(const RocketConfig& config, const std::vector<int>& kept_positions, bool full_attention);
    void begin_step(int position);
    void finish_step() const;
    const RocketStage2Stats& stats() const { return stats_; }

private:
    struct Layer {
        RocketAdapter* owner;
        int index;
        // Original KV views in the active graph, before scheduler input copies.
        const ggml_tensor* k = nullptr;
        const ggml_tensor* v = nullptr;
    };

    static ggml_tensor* build_attention(
        ggml_context* graph_ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
        ggml_tensor* mask, float scale, int32_t layer, void* user_data);
    static void compute(ggml_tensor* dst, int ith, int nth, void* user_data);
    static void encode(ggml_tensor* dst, void* encoder, void* user_data);
    void execute(ggml_tensor* dst, int layer, void* encoder);
    void compute_layer(ggml_tensor* dst, int layer, void* encoder);
    void refresh_slots();

    llama_context* ctx_;
    std::unique_ptr<RocketMetal> metal_;
    RocketConfig config_{};
    RocketStage2Stats stats_;
    std::vector<Layer> layers_;
    std::vector<int> candidates_;
    std::vector<llama_pos> positions_;
    std::vector<int> slots_;
    std::vector<float> q_, k_, v_;
    std::string error_;
    std::mutex encode_mutex_;
    int next_layer_ = 0;
    bool active_ = false;
    bool step_pending_ = false;
    bool full_attention_ = false;
};
