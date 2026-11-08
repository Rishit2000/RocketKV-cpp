#pragma once
#include "rocket.h"
#include "ggml.h"
#include <memory>
#include <vector>

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
    void run(const ggml_tensor *q, const ggml_tensor *k, const ggml_tensor *v,
             const std::vector<int> &slots, const RocketConfig &config, float *output,
             std::vector<std::vector<int>> *selected, int layer, void *encoder, ggml_tensor *dst);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
