#include "router.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const float x[5] = {0.2f, 0.9f, 0.1f, 0.7f, 0.8f};
    int idx[2];
    float vals[2];

    topk_select(x, 5, 2, idx, vals);

    if (!(idx[0] == 1 && idx[1] == 4)) {
        fprintf(stderr, "topk indices failed: got [%d, %d]\n", idx[0], idx[1]);
        return EXIT_FAILURE;
    }

    printf("test_topk passed\n");
    return EXIT_SUCCESS;
}
