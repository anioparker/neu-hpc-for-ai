#ifndef ROUTER_H
#define ROUTER_H

void router_logits(
    const float *input,
    const float *router_weight,
    float *logits,
    int num_tokens,
    int hidden_dim,
    int num_experts
);

void softmax_stable(const float *logits, float *probs, int n);
void topk_select(const float *values, int n, int k, int *indices, float *topk_values);

#endif
