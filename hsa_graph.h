#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "rocket.h"

// q: [D, n_heads, 1], k: [D, n_kv, 1] (the new token only).
void run_hsa_head_dim_graph(
    ggml_backend_t backend,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    const RocketConfig & config);