#ifndef EXPERT_H
#define EXPERT_H

typedef struct {
    int hidden_dim;
    int intermediate_dim;
    float *up_proj;     /* [hidden_dim, intermediate_dim] */
    float *gate_proj;   /* [hidden_dim, intermediate_dim] */
    float *down_proj;   /* [intermediate_dim, hidden_dim] */
} Expert;

Expert expert_create(int hidden_dim, int intermediate_dim);
void expert_free(Expert *expert);
void expert_forward(float *output, const float *input, const Expert *expert);

#endif
