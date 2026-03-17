#include "expert.h"
#include "tensor.h"
#include "utils.h"

#include <stdlib.h>

Expert expert_create(int hidden_dim, int intermediate_dim) {
    Expert expert;
    expert.hidden_dim = hidden_dim;
    expert.intermediate_dim = intermediate_dim;
    expert.up_proj = tensor_alloc((size_t)hidden_dim * (size_t)intermediate_dim);
    expert.gate_proj = tensor_alloc((size_t)hidden_dim * (size_t)intermediate_dim);
    expert.down_proj = tensor_alloc((size_t)intermediate_dim * (size_t)hidden_dim);
    return expert;
}

void expert_free(Expert *expert) {
    if (!expert) {
        return;
    }
    tensor_free(expert->up_proj);
    tensor_free(expert->gate_proj);
    tensor_free(expert->down_proj);
    expert->up_proj = NULL;
    expert->gate_proj = NULL;
    expert->down_proj = NULL;
}

void expert_forward(float *output, const float *input, const Expert *expert) {
    int h = expert->hidden_dim;
    int i = expert->intermediate_dim;

    float *up = tensor_alloc((size_t)i);
    float *gate = tensor_alloc((size_t)i);
    float *activated = tensor_alloc((size_t)i);
    float *mixed = tensor_alloc((size_t)i);

    matvec(input, expert->up_proj, up, h, i);
    matvec(input, expert->gate_proj, gate, h, i);

    for (int j = 0; j < i; ++j) {
        activated[j] = silu(gate[j]);
    }

    hadamard(mixed, up, activated, i);
    matvec(mixed, expert->down_proj, output, i, h);

    tensor_free(up);
    tensor_free(gate);
    tensor_free(activated);
    tensor_free(mixed);
}
