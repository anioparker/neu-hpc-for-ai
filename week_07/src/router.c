#include "router.h"
#include "utils.h"

#include <float.h>
#include <math.h>

void router_logits(
    const float *input,
    const float *router_weight,
    float *logits,
    int num_tokens,
    int hidden_dim,
    int num_experts
) {
    for (int t = 0; t < num_tokens; ++t) {
        const float *x = input + t * hidden_dim;
        float *y = logits + t * num_experts;
        matvec(x, router_weight, y, hidden_dim, num_experts);
    }
}

void softmax_stable(const float *logits, float *probs, int n) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }

    if (sum == 0.0f) {
        float uniform = 1.0f / (float)n;
        for (int i = 0; i < n; ++i) {
            probs[i] = uniform;
        }
        return;
    }

    for (int i = 0; i < n; ++i) {
        probs[i] /= sum;
    }
}

void topk_select(const float *values, int n, int k, int *indices, float *topk_values) {
    for (int i = 0; i < k; ++i) {
        indices[i] = -1;
        topk_values[i] = -FLT_MAX;
    }

    for (int i = 0; i < n; ++i) {
        float v = values[i];
        int pos = -1;
        for (int j = 0; j < k; ++j) {
            if (v > topk_values[j] ||
                (v == topk_values[j] && (indices[j] == -1 || i < indices[j]))) {
                pos = j;
                break;
            }
        }
        if (pos == -1) {
            continue;
        }
        for (int j = k - 1; j > pos; --j) {
            topk_values[j] = topk_values[j - 1];
            indices[j] = indices[j - 1];
        }
        topk_values[pos] = v;
        indices[pos] = i;
    }
}
