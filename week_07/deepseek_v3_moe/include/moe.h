#ifndef MOE_H
#define MOE_H

#include "expert.h"

typedef struct {
    int num_experts;
    int top_k;
    int hidden_dim;
    int intermediate_dim;
    float *router_weight; /* [hidden_dim, num_experts] */
    Expert *experts;
} MoE;

MoE moe_create(int num_experts, int top_k, int hidden_dim, int intermediate_dim);
void moe_free(MoE *moe);
void moe_forward(float *output, const float *input, int num_tokens, const MoE *moe);

#endif
