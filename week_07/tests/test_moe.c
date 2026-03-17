#include "moe.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    MoE moe = moe_create(2, 1, 2, 2);
    if (!moe.experts || !moe.router_weight) {
        fprintf(stderr, "moe allocation failed\n");
        return EXIT_FAILURE;
    }

    moe.router_weight[0] = 1.0f; moe.router_weight[1] = 0.0f;
    moe.router_weight[2] = 0.0f; moe.router_weight[3] = 1.0f;

    for (int e = 0; e < 2; ++e) {
        Expert *ex = &moe.experts[e];
        ex->up_proj[0] = 1.0f; ex->up_proj[1] = 0.0f;
        ex->up_proj[2] = 0.0f; ex->up_proj[3] = 1.0f;

        ex->gate_proj[0] = 1.0f; ex->gate_proj[1] = 0.0f;
        ex->gate_proj[2] = 0.0f; ex->gate_proj[3] = 1.0f;

        ex->down_proj[0] = 1.0f; ex->down_proj[1] = 0.0f;
        ex->down_proj[2] = 0.0f; ex->down_proj[3] = 1.0f;
    }

    const float input[2] = {2.0f, 1.0f};
    float output[2];
    moe_forward(output, input, 1, &moe);

    if (!(output[0] > 0.0f && output[1] > 0.0f)) {
        fprintf(stderr, "moe forward failed\n");
        moe_free(&moe);
        return EXIT_FAILURE;
    }

    moe_free(&moe);
    printf("test_moe passed\n");
    return EXIT_SUCCESS;
}
