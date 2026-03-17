#include "moe.h"
#include "tensor.h"

#include <stdio.h>

static void init_demo_moe(MoE *moe) {
    int h = moe->hidden_dim;
    int e = moe->num_experts;
    int i = moe->intermediate_dim;

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < e; ++col) {
            moe->router_weight[row * e + col] = 0.05f * (float)(row + 1) + 0.02f * (float)(col + 1);
        }
    }

    for (int ex = 0; ex < e; ++ex) {
        Expert *expert = &moe->experts[ex];
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < i; ++col) {
                expert->up_proj[row * i + col] = 0.01f * (float)(1 + ex + row + col);
                expert->gate_proj[row * i + col] = 0.02f * (float)(1 + ex + row + col);
            }
        }
        for (int row = 0; row < i; ++row) {
            for (int col = 0; col < h; ++col) {
                expert->down_proj[row * h + col] = 0.015f * (float)(1 + ex + row + col);
            }
        }
    }
}

int main(void) {
    const int num_tokens = 2;
    const int hidden_dim = 4;
    const int num_experts = 3;
    const int top_k = 2;
    const int intermediate_dim = 6;

    float input[num_tokens * hidden_dim];
    float output[num_tokens * hidden_dim];

    input[0] = 1.0f; input[1] = 0.5f; input[2] = -0.25f; input[3] = 2.0f;
    input[4] = -1.0f; input[5] = 0.25f; input[6] = 1.5f; input[7] = 0.75f;

    MoE moe = moe_create(num_experts, top_k, hidden_dim, intermediate_dim);
    if (!moe.experts || !moe.router_weight) {
        fprintf(stderr, "Failed to create MoE\n");
        return 1;
    }

    init_demo_moe(&moe);
    moe_forward(output, input, num_tokens, &moe);

    tensor_print_2d(input, num_tokens, hidden_dim, "input");
    tensor_print_2d(output, num_tokens, hidden_dim, "output");

    moe_free(&moe);
    return 0;
}
