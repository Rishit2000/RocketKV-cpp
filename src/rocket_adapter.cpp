#include "rocket_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

float tensor_value(const ggml_tensor* tensor, int64_t x, int64_t y = 0, int64_t z = 0) {
    if (!tensor || !tensor->data || x < 0 || x >= tensor->ne[0] ||
        y < 0 || y >= tensor->ne[1] || z < 0 || z >= tensor->ne[2]) {
        throw std::runtime_error("Invalid HSA tensor access");
    }
    const auto* data = static_cast<const char*>(tensor->data) +
                       x * tensor->nb[0] + y * tensor->nb[1] + z * tensor->nb[2];
    if (tensor->type == GGML_TYPE_F32) {
        float value;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t value;
        std::memcpy(&value, data, sizeof(value));
        return ggml_fp16_to_fp32(value);
    }
    throw std::runtime_error("CPU HSA supports only F32 activations and F16/F32 tensor reads");
}

void update_range(int value, int& minimum, int& maximum) {
    minimum = minimum == 0 ? value : std::min(minimum, value);
    maximum = std::max(maximum, value);
}

} // namespace

RocketAdapter::RocketAdapter(llama_context* ctx, int layer_count, bool metal) : ctx_(ctx) {
    if (!ctx || layer_count <= 0 || llama_rocket_abi_version() != LLAMA_ROCKET_ABI_VERSION) {
        throw std::runtime_error("Missing or incompatible llama.cpp Rocket extension");
    }
    if (metal) metal_ = std::make_unique<RocketMetal>();
    layers_.reserve(layer_count);
    for (int i = 0; i < layer_count; ++i) layers_.push_back({this, i});
    reset();
}

RocketAdapter::~RocketAdapter() {
    llama_set_rocket_attention(ctx_, nullptr, nullptr);
}

void RocketAdapter::reset() {
    if (!llama_set_rocket_attention(ctx_, nullptr, nullptr)) {
        throw std::runtime_error("Could not disable HSA");
    }
    if (metal_) metal_->reset();
    for (auto& layer : layers_) layer.k = layer.v = nullptr;
    active_ = false;
    step_pending_ = false;
    next_layer_ = 0;
    error_.clear();
    candidates_.clear();
    slots_.clear();
    stats_ = {};
    stats_.calls_per_layer.assign(layers_.size(), 0);
}

void RocketAdapter::initialize(
    const RocketConfig& config, const std::vector<int>& kept_positions, bool full_attention) {
    if (active_) throw std::logic_error("Reset HSA before initialization");
    if (kept_positions.empty() || !std::is_sorted(kept_positions.begin(), kept_positions.end()) ||
        std::adjacent_find(kept_positions.begin(), kept_positions.end()) != kept_positions.end() ||
        kept_positions.front() < 0 || kept_positions.back() >= (int)llama_n_ctx(ctx_) ||
        config.num_heads <= 0 || config.num_heads_kv <= 0 || config.dim_head <= 0 ||
        config.num_heads % config.num_heads_kv != 0 || config.topk_decode <= 0 ||
        config.r_dim <= 0 || config.chunk_size <= 0) {
        throw std::invalid_argument("Invalid HSA candidates or configuration");
    }
    config_ = config;
    candidates_ = kept_positions;
    full_attention_ = full_attention;
    const int capacity = llama_rocket_kv_positions(ctx_, nullptr, 0);
    if (capacity <= 0) throw std::runtime_error("HSA requires a single-sequence KV cache");
    positions_.resize(capacity);
    refresh_slots();
    if (!llama_set_rocket_attention(ctx_, build_attention, this)) {
        throw std::runtime_error("HSA requires the supported Llama/F16/single-sequence configuration");
    }
    active_ = true;
}

void RocketAdapter::refresh_slots() {
    if (llama_rocket_kv_positions(ctx_, positions_.data(), positions_.size()) != (int)positions_.size()) {
        throw std::runtime_error("Could not read HSA KV positions");
    }
    std::vector<std::pair<int, int>> live;
    live.reserve(candidates_.size());
    for (size_t slot = 0; slot < positions_.size(); ++slot) {
        if (positions_[slot] >= 0) live.emplace_back(positions_[slot], (int)slot);
    }
    std::sort(live.begin(), live.end());
    if (live.size() != candidates_.size()) {
        throw std::runtime_error("HSA candidate count does not match the live KV cache");
    }
    slots_.resize(live.size());
    for (size_t i = 0; i < live.size(); ++i) {
        if (live[i].first != candidates_[i]) {
            throw std::runtime_error("HSA candidate positions do not match the live KV cache");
        }
        slots_[i] = live[i].second;
    }
}

void RocketAdapter::begin_step(int position) {
    if (!active_ || step_pending_ || !error_.empty() ||
        position <= candidates_.back() || position >= (int)llama_n_ctx(ctx_)) {
        throw std::runtime_error("Invalid HSA decode step");
    }
    candidates_.push_back(position);
    next_layer_ = 0;
    step_pending_ = true;
}

void RocketAdapter::finish_step() const {
    if (!error_.empty()) throw std::runtime_error(error_);
    if (step_pending_ || next_layer_ != 0) {
        throw std::runtime_error("HSA did not execute every model layer");
    }
    if (metal_) metal_->check_errors();
}

ggml_tensor* RocketAdapter::build_attention(
    ggml_context* graph_ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
    ggml_tensor* mask, float scale, int32_t layer, void* user_data) {
    auto& self = *static_cast<RocketAdapter*>(user_data);
    const auto& cfg = self.config_;
    if (layer < 0 || layer >= (int)self.layers_.size() || !mask ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        q->ne[0] != cfg.dim_head || q->ne[1] != 1 || q->ne[2] != cfg.num_heads || q->ne[3] != 1 ||
        k->ne[0] != cfg.dim_head || k->ne[2] != cfg.num_heads_kv || k->ne[3] != 1 ||
        !ggml_are_same_shape(k, v) || mask->ne[0] != k->ne[1] ||
        std::abs(scale - 1.0f / std::sqrt((float)cfg.dim_head)) > 1e-6f) {
        throw std::runtime_error("Unsupported tensor shape, type, or attention scale for HSA");
    }
    ggml_tensor* args[] = {q, k, v, mask};
    auto& state = self.layers_[layer];
    state.k = state.v = nullptr;
    auto* out = ggml_custom_4d(graph_ctx, GGML_TYPE_F32, cfg.dim_head, cfg.num_heads, 1, 1,
                               args, 4, compute, 1, &state);
    if (self.metal_ && !self.metal_->attach(out, encode, &state) &&
        self.metal_->can_bind(k) && self.metal_->can_bind(v)) {
        state.k = k;
        state.v = v;
        // Keep scheduler dependencies without copying the full KV tensors to CPU.
        out->src[1] = ggml_view_1d(graph_ctx, k, 1, 0);
        out->src[2] = ggml_view_1d(graph_ctx, v, 1, 0);
    }
    ggml_format_name(out, "rocket_hsa-%d", layer);
    return out;
}

void RocketAdapter::compute(ggml_tensor* dst, int ith, int nth, void* user_data) {
    (void)nth;
    if (ith != 0) return;
    auto& layer = *static_cast<Layer*>(user_data);
    layer.owner->execute(dst, layer.index, nullptr);
}

void RocketAdapter::encode(ggml_tensor* dst, void* encoder, void* user_data) {
    auto& layer = *static_cast<Layer*>(user_data);
    layer.owner->execute(dst, layer.index, encoder);
}

void RocketAdapter::execute(ggml_tensor* dst, int layer, void* encoder) {
    // GGML may encode different command buffers on concurrent host threads.
    std::lock_guard<std::mutex> lock(encode_mutex_);
    try {
        if (!error_.empty()) throw std::runtime_error(error_);
        compute_layer(dst, layer, encoder);
    } catch (const std::exception& error) {
        // Exceptions must not cross GGML's worker boundary.
        if (error_.empty()) error_ = "HSA layer " + std::to_string(layer) + ": " + error.what();
        if (!encoder) std::fill_n(static_cast<float*>(dst->data), ggml_nelements(dst),
                                 std::numeric_limits<float>::quiet_NaN());
    }
}

void RocketAdapter::compute_layer(ggml_tensor* dst, int layer, void* encoder) {
    if (!active_ || !step_pending_ || stats_.calls_per_layer[layer] != stats_.layer_calls / int64_t(layers_.size())) {
        throw std::runtime_error("Unexpected or duplicate HSA layer execution");
    }
    if (next_layer_ == 0) refresh_slots();
    const int count = (int)slots_.size();
    const int dim = config_.dim_head;
    for (int i = 0; !encoder && i < count; ++i) {
        if (tensor_value(dst->src[3], slots_[i]) != 0.0f) {
            throw std::runtime_error("Candidate is masked or uses unsupported attention bias");
        }
    }
    auto cfg = config_;
    if (full_attention_) cfg.topk_decode = count;
    if (encoder) {
        metal_->encode(dst, slots_, cfg, encoder, layer);
    } else if (metal_) {
        const auto& state = layers_[layer];
        metal_->compute(dst->src[0], state.k ? state.k : dst->src[1], state.v ? state.v : dst->src[2], slots_, cfg,
                        static_cast<float*>(dst->data), nullptr, layer);
    } else {
        q_.resize((size_t)config_.num_heads * dim);
        k_.resize((size_t)config_.num_heads_kv * count * dim);
        v_.resize(k_.size());
        for (int h = 0; h < config_.num_heads; ++h) {
            for (int d = 0; d < dim; ++d) {
                q_[(size_t)h * dim + d] = tensor_value(dst->src[0], d, 0, h);
            }
        }
        for (int h = 0; h < config_.num_heads_kv; ++h) {
            for (int i = 0; i < count; ++i) {
                for (int d = 0; d < dim; ++d) {
                    const size_t index = ((size_t)h * count + i) * dim + d;
                    k_[index] = tensor_value(dst->src[1], d, slots_[i], h);
                    v_[index] = tensor_value(dst->src[2], d, slots_[i], h);
                }
            }
        }
        const auto output = compute_hybrid_sparse_decode(q_.data(), k_.data(), v_.data(), count, cfg);
        std::memcpy(dst->data, output.data(), output.size() * sizeof(float));
    }
    ++stats_.layer_calls;
    ++stats_.calls_per_layer[layer];
    update_range(count, stats_.candidate_min, stats_.candidate_max);
    update_range(std::min(cfg.topk_decode, count), stats_.selected_min, stats_.selected_max);
    if (++next_layer_ == (int)layers_.size()) {
        next_layer_ = 0;
        step_pending_ = false;
    }
}
