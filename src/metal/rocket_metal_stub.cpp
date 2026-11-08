#include "rocket_metal.h"
#include <stdexcept>
struct RocketMetal::Impl {};
RocketMetal::RocketMetal() {
    throw std::runtime_error("Metal HSA is available only on macOS");
}
RocketMetal::~RocketMetal() = default;
void RocketMetal::compute(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
                          const std::vector<int> &, const RocketConfig &, float *,
                          std::vector<std::vector<int>> *) {
    throw std::runtime_error("Metal HSA is available only on macOS");
}
