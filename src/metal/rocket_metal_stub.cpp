#include "rocket_metal.h"
#include <stdexcept>
struct RocketMetal::Impl {};
RocketMetal::RocketMetal() {
    throw std::runtime_error("Metal HSA is available only on macOS");
}
RocketMetal::~RocketMetal() = default;
void RocketMetal::reset() {}
bool RocketMetal::can_bind(const ggml_tensor *) const { return false; }
bool RocketMetal::attach(ggml_tensor *, EncodeCallback, void *) const { return false; }
void RocketMetal::check_errors() const {}
void RocketMetal::encode(ggml_tensor *, const std::vector<int> &, const RocketConfig &, void *, int) {
    throw std::runtime_error("Metal HSA is available only on macOS");
}
void RocketMetal::compute(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
                          const std::vector<int> &, const RocketConfig &, float *,
                          std::vector<std::vector<int>> *, int) {
    throw std::runtime_error("Metal HSA is available only on macOS");
}
