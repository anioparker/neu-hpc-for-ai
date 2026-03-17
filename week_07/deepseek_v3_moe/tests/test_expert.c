#include "expert.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    Expert ex = expert_create(2, 2);
    if (!ex.up_proj || !ex.gate_proj || !ex.down_proj) {
        fprintf(stderr, "expert allocation failed\n");
        return EXIT_FAILURE;
    }

    ex.up_proj[0] = 1.0f; ex.up_proj[1] = 0.0f;
    ex.up_proj[2] = 0.0f; ex.up_proj[3] = 1.0f;

    ex.gate_proj[0] = 1.0f; ex.gate_proj[1] = 0.0f;
    ex.gate_proj[2] = 0.0f; ex.gate_proj[3] = 1.0f;

    ex.down_proj[0] = 1.0f; ex.down_proj[1] = 0.0f;
    ex.down_proj[2] = 0.0f; ex.down_proj[3] = 1.0f;

    const float input[2] = {1.0f, 2.0f};
    float output[2];
    expert_forward(output, input, &ex);

    if (!(output[0] > 0.0f && output[1] > 0.0f)) {
        fprintf(stderr, "expert forward failed\n");
        expert_free(&ex);
        return EXIT_FAILURE;
    }

    expert_free(&ex);
    printf("test_expert passed\n");
    return EXIT_SUCCESS;
}
