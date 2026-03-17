#include <stdlib.h>
#include "moe.h"
#include "router.h"
#include "tensor.h"
#include "utils.h"

#include <stddef.h>

MoE moe_create(int num_experts, int top_k, int hidden_dim, int intermediate_dim) {
    MoE moe;
    moe.num_experts = num_experts;
    moe.top_k = top_k;
    moe.hidden_dim = hidden_dim;
    moe.intermediate_dim = intermediate_dim;
    moe.router_weight = tensor_alloc((size_t)hidden_dim * (size_t)num_experts);
    moe.experts = (Expert *)tensor_alloc((size_t)num_experts * sizeof(Expert) / sizeof(float));

    /* tensor_alloc returns float*, so allocate Experts separately below in a portable way. */
    tensor_free((float *)moe.experts);
    moe.experts = NULL;
    moe.experts = (Expert *)calloc((size_t)num_experts, sizeof(Expert));
    if (!moe.experts) {
        tensor_free(moe.router_weight);
        moe.router_weight = NULL;
        return moe;
    }

    for (int e = 0; e < num_experts; ++e) {
        moe.experts[e] = expert_create(hidden_dim, intermediate_dim);
    }
    return moe;
}

void moe_free(MoE *moe) {
    if (!moe) {
        return;
    }
    if (moe->experts) {
        for (int e = 0; e < moe->num_experts; ++e) {
            expert_free(&moe->experts[e]);
        }
        free(moe->experts);
        moe->experts = NULL;
    }
    tensor_free(moe->router_weight);
    moe->router_weight = NULL;
}

void moe_forward(float *output, const float *input, int num_tokens, const MoE *moe) {
    int h = moe->hidden_dim;
    int e = moe->num_experts;
    int k = moe->top_k;

    float *logits = tensor_alloc((size_t)num_tokens * (size_t)e);
    float *probs = tensor_alloc((size_t)e);
    int *topk_ids = (int *)calloc((size_t)k, sizeof(int));
    float *topk_weights = tensor_alloc((size_t)k);
    float *expert_out = tensor_alloc((size_t)h);

    router_logits(input, moe->router_weight, logits, num_tokens, h, e);

    for (int t = 0; t < num_tokens; ++t) {
        const float *token_x = input + t * h;
        const float *token_logits = logits + t * e;
        float *token_y = output + t * h;

        zero_vector(token_y, h);
        softmax_stable(token_logits, probs, e);
        topk_select(probs, e, k, topk_ids, topk_weights);

        for (int j = 0; j < k; ++j) {
            int expert_id = topk_ids[j];
            float weight = topk_weights[j];
            expert_forward(expert_out, token_x, &moe->experts[expert_id]);
            vec_add_scaled(token_y, expert_out, weight, h);
        }
    }

    tensor_free(logits);
    tensor_free(probs);
    free(topk_ids);
    tensor_free(topk_weights);
    tensor_free(expert_out);
}
