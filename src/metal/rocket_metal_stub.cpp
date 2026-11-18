#include "rocket_metal.h"
#include <stdexcept>
struct RocketMetal::Impl {};
RocketMetal::RocketMetal() {
    throw std::runtime_error("Metal HSA is available only on macOS");
}
RocketMetal::~RocketMetal() = default;
void RocketMetal::reset() {}
void RocketMetal::begin_prefill(int, int, const RocketConfig &) {
    throw std::runtime_error("Metal Stage 1 is available only on macOS");
}
void RocketMetal::capture_prefill(const ggml_tensor *, bool, int) {
    throw std::runtime_error("Metal Stage 1 is available only on macOS");
}
std::vector<int> RocketMetal::finish_prefill(std::vector<float> *) {
    throw std::runtime_error("Metal Stage 1 is available only on macOS");
}
RocketCaptureStats RocketMetal::capture_stats() const { return {}; }
std::vector<int> RocketMetal::prefill(const float *, const float *, int, int,
                                     const RocketConfig &, std::vector<float> *) {
    throw std::runtime_error("Metal Stage 1 is available only on macOS");
}
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
