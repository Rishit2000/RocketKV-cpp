#pragma once
#include "rocket.h"
#include "ggml.h"
#include <memory>
#include <vector>

// One synchronous Metal command buffer per attention layer. The GGML scheduler
// makes inputs CPU-readable before this call; results are ready when it returns.
class RocketMetal {
  public:
    RocketMetal();
    ~RocketMetal();
    RocketMetal(const RocketMetal &) = delete;
    RocketMetal &operator=(const RocketMetal &) = delete;
    void compute(const ggml_tensor *q, const ggml_tensor *k, const ggml_tensor *v,
                 const std::vector<int> &slots, const RocketConfig &config, float *output,
                 std::vector<std::vector<int>> *selected = nullptr);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
