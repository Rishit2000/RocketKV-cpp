#pragma once
#include "rocket.h"
#include "ggml.h"
#include <memory>
#include <vector>

struct RocketCaptureStats {
    int metal_calls = 0;
    int host_calls = 0;
    size_t host_bytes = 0;
};

// compute() is synchronous; encode() borrows the scheduler's Metal encoder.
// Keep inputs/state alive until GPU completion. Unbound compute() inputs must be CPU-readable.
// Serialize host calls; synchronize the previous step before reusing layer buffers.
class RocketMetal {
  public:
    RocketMetal();
    ~RocketMetal();
    RocketMetal(const RocketMetal &) = delete;
    RocketMetal &operator=(const RocketMetal &) = delete;
    void reset();
    // Host-readable F32 inputs, laid out [head, token, dim]. Top-K stays on CPU.
    std::vector<int> prefill(const float *q, const float *k, int len_q, int len_k,
                            const RocketConfig &config, std::vector<float> *importance = nullptr);
    // Capture ordered microbatches after their producer has synchronized.
    // Each call finishes its GPU copy before returning; source storage may then be reused.
    void begin_prefill(int len_q, int len_k, const RocketConfig &config);
    void capture_prefill(const ggml_tensor *tensor, bool is_query, int offset);
    std::vector<int> finish_prefill(std::vector<float> *importance = nullptr);
    RocketCaptureStats capture_stats() const;
    bool can_bind(const ggml_tensor *tensor) const;
    using EncodeCallback = void (*)(ggml_tensor *, void *, void *);
    bool attach(ggml_tensor *tensor, EncodeCallback callback, void *userdata) const;
    void encode(ggml_tensor *dst, const std::vector<int> &slots, const RocketConfig &config,
                void *encoder, int layer);
    // Call only after the scheduler has synchronized all encoded work.
    void check_errors() const;
    // A nonnegative layer caches immutable KV prefixes until reset().
    // Changed slots/configuration rebuild the cache; -1 computes without reuse.
    void compute(const ggml_tensor *q, const ggml_tensor *k, const ggml_tensor *v,
                 const std::vector<int> &slots, const RocketConfig &config, float *output,
                 std::vector<std::vector<int>> *selected = nullptr, int layer = -1);

  private:
    std::vector<int> score_prefill(bool pack_keys, std::vector<float> *importance);
    void run(const ggml_tensor *q, const ggml_tensor *k, const ggml_tensor *v,
             const std::vector<int> &slots, const RocketConfig &config, float *output,
             std::vector<std::vector<int>> *selected, int layer, void *encoder, ggml_tensor *dst);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
