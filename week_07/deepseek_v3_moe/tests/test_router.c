#include "router.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const float logits[3] = {1.0f, 2.0f, 3.0f};
    float probs[3];
    softmax_stable(logits, probs, 3);

    float sum = probs[0] + probs[1] + probs[2];
    if (!nearly_equal(sum, 1.0f, 1e-5f)) {
        fprintf(stderr, "softmax sum test failed: got %f\n", sum);
        return EXIT_FAILURE;
    }

    printf("test_router passed\n");
    return EXIT_SUCCESS;
}
