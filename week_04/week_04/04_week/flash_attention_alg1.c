// flash_attention_alg1.c  (CPU reference for Algorithm 1: FlashAttention-2 forward)

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>

#include "assertc.h"
#include "trace.h"
#include "float_eq.h"
#include "mat_util.h"
#include "div_ceil.h"
#include "tile.h"
#include "min.h"

/* ---------------- helpers (file scope) ---------------- */

static inline void tile_fill(tile t, float v) {
    for (unsigned int r = 0; r < t.size.x; ++r)
        for (unsigned int c = 0; c < t.size.y; ++c)
            tile_set(t, r, c, v);
}

// O[r,:] *= alpha[r]
static inline void row_scale_inplace(tile O, tile alpha /*(Br,1)*/) {
    assertc(alpha.size.x == O.size.x && alpha.size.y == 1);
    for (unsigned int r = 0; r < O.size.x; ++r) {
        float a = tile_at(alpha, r, 0);
        for (unsigned int c = 0; c < O.size.y; ++c) {
            tile_set(O, r, c, tile_at(O, r, c) * a);
        }
    }
}

// O[r,:] += sum_{c} P[r,c] * V[c,:]   where P is (Br, bc) and V is (bc, d)
// Here P is stored in tile Ptilde (Br, Bc) but only first bc cols are valid.
static inline void add_PtildeV_inplace(tile O, tile Ptilde, tile V, unsigned int bc) {
    assertc(O.size.x == Ptilde.size.x);
    assertc(O.size.y == V.size.y);
    assertc(V.size.x >= bc);
    for (unsigned int r = 0; r < O.size.x; ++r) {
        for (unsigned int outc = 0; outc < O.size.y; ++outc) {
            float acc = tile_at(O, r, outc);
            float sum = 0.0f;
            for (unsigned int c = 0; c < bc; ++c) {
                sum += tile_at(Ptilde, r, c) * tile_at(V, c, outc);
            }
            tile_set(O, r, outc, acc + sum);
        }
    }
}

// O[r,:] /= l[r]
static inline void row_div_inplace(tile O, tile l /*(Br,1)*/) {
    assertc(l.size.x == O.size.x && l.size.y == 1);
    for (unsigned int r = 0; r < O.size.x; ++r) {
        float denom = tile_at(l, r, 0);
        // denom should be > 0
        for (unsigned int c = 0; c < O.size.y; ++c) {
            tile_set(O, r, c, tile_at(O, r, c) / denom);
        }
    }
}

/* ---------------- Algorithm 1 block (per i) ---------------- */

void flash_attention_block_alg1(tile Q, tile K, tile V,
                                tile O, tile L,                 // outputs: O (N,d), L (N,1)
                                unsigned int N, unsigned int d,
                                unsigned int i, unsigned int Br, unsigned int Bc) {

    // (Alg 1, line 1–2): pick the i-th block of Q and output O/L
    tile Qi_g = sub_tile(Q, idx(i, 0), dim(Br, d));
    unsigned int Br_eff = Qi_g.size.x; // edge clip

    tile Oi_g = sub_tile(O, idx(i, 0), dim(Br_eff, d));
    tile Li_g = sub_tile(L, idx(i, 0), dim(Br_eff, 1));

    unsigned int Tc = div_ceil(N, Bc);

    // On-chip buffers (CPU heap here)
    tile Qi  = tile_alloc(Br_eff, d);
    tile Kj  = tile_alloc(Bc, d);
    tile Vj  = tile_alloc(Bc, d);

    tile S   = tile_alloc(Br_eff, Bc); // S_i^(j)  (Alg 1, line 8)
    tile P   = tile_alloc(Br_eff, Bc); // \tilde{P}_i^(j) (Alg 1, line 9)
    tile Oi  = tile_alloc(Br_eff, d);  // O_i^(j)
    tile li  = tile_alloc(Br_eff, 1);  // \ell_i^(j)
    tile mi  = tile_alloc(Br_eff, 1);  // m_i^(j)

    tile mij = tile_alloc(Br_eff, 1);  // m_ij (Alg 1, line 9)
    tile lij = tile_alloc(Br_eff, 1);  // \ell_ij (Alg 1, line 9)
    tile alpha = tile_alloc(Br_eff, 1);// exp(m^{j-1}-m^{j}) (used in line 10)

    // (Alg 1, line 4): Load Qi
    tilecpy(Qi, Qi_g);

    // (Alg 1, line 5): init O, l, m
    tile_fill(Oi, 0.0f);
    tile_fill(li, 0.0f);
    tile_fill(mi, -INFINITY);

    // (Alg 1, line 6): for 1 <= j <= Tc
    for (unsigned int j = 0; j < Tc; ++j) {

        // (Alg 1, line 7): Load Kj, Vj
        tile Kj_g = sub_tile(K, idx(j, 0), dim(Bc, d));
        tile Vj_g = sub_tile(V, idx(j, 0), dim(Bc, d));
        unsigned int bc = Kj_g.size.x; // effective rows in this block (<= Bc)

        // zero-pad then copy only bc rows
        tile_fill(Kj, 0.0f);
        tile_fill(Vj, 0.0f);
        tilecpy(sub_tile(Kj, idx(0,0), dim(bc, d)), Kj_g);
        tilecpy(sub_tile(Vj, idx(0,0), dim(bc, d)), Vj_g);

        // (Alg 1, line 8): S = Qi * Kj^T  (Br_eff x d) * (Bc x d)^T = (Br_eff x Bc)
        // Only columns 0..bc-1 are valid; cols bc..Bc-1 correspond to padding and should not contribute.
        for (unsigned int r = 0; r < Br_eff; ++r) {
            for (unsigned int c = 0; c < Bc; ++c) {
                if (c >= bc) { // masked padded keys
                    tile_set(S, r, c, -INFINITY);
                    continue;
                }
                float sum = 0.0f;
                for (unsigned int k = 0; k < d; ++k) {
                    sum += tile_at(Qi, r, k) * tile_at(Kj, c, k);
                }
                tile_set(S, r, c, sum);
            }
        }

        // (Alg 1, line 9): compute m_ij, Ptilde, l_ij (online softmax)
        for (unsigned int r = 0; r < Br_eff; ++r) {
            float m_prev = tile_at(mi, r, 0);

            // rowmax(S[r,:]) over valid cols (but masked cols are -inf already)
            float rowmax = -INFINITY;
            for (unsigned int c = 0; c < Bc; ++c) {
                float v = tile_at(S, r, c);
                if (v > rowmax) rowmax = v;
            }

            float m_new = (m_prev > rowmax) ? m_prev : rowmax;
            tile_set(mij, r, 0, m_new);

            // Ptilde = exp(S - m_new), and rowsum(Ptilde)
            float rowsum = 0.0f;
            for (unsigned int c = 0; c < Bc; ++c) {
                float s_rc = tile_at(S, r, c);
                float p = (isinf(s_rc) && s_rc < 0) ? 0.0f : expf(s_rc - m_new);
                tile_set(P, r, c, p);
                rowsum += p;
            }

            float l_prev = tile_at(li, r, 0);
            float l_new = expf(m_prev - m_new) * l_prev + rowsum;
            tile_set(lij, r, 0, l_new);

            // alpha = exp(m_prev - m_new) used in line 10 scaling of old O
            tile_set(alpha, r, 0, expf(m_prev - m_new));
        }

        // (Alg 1, line 10): O_i^(j) = diag(exp(m^{j-1}-m^j)) O_i^(j-1) + Ptilde Vj
        row_scale_inplace(Oi, alpha);
        add_PtildeV_inplace(Oi, P, Vj, bc);

        // Commit updates: mi <- mij, li <- lij
        tilecpy(mi, mij);
        tilecpy(li, lij);
    }

    // (Alg 1, line 12): O_i = diag(1/l_i) O_i
    row_div_inplace(Oi, li);

    // (Alg 1, line 13): L_i = m_i + log(l_i)
    for (unsigned int r = 0; r < Br_eff; ++r) {
        float m_final = tile_at(mi, r, 0);
        float l_final = tile_at(li, r, 0);
        tile_set(Li_g, r, 0, m_final + logf(l_final));
    }

    // (Alg 1, line 14): write O_i to HBM
    tilecpy(Oi_g, Oi);

    // (Alg 1, line 15): write L_i to HBM
    // already stored into Li_g above

    // free
    free(Qi.data);
    free(Kj.data);
    free(Vj.data);
    free(S.data);
    free(P.data);
    free(Oi.data);
    free(li.data);
    free(mi.data);
    free(mij.data);
    free(lij.data);
    free(alpha.data);
}

/* ---------------- full forward ---------------- */

void flash_attention_alg1(tile Q, tile K, tile V, tile O, tile L,
                          unsigned int N, unsigned int d,
                          unsigned int Br, unsigned int Bc) {

    unsigned int Tr = div_ceil(N, Br);

    for (unsigned int i = 0; i < Tr; ++i) {
        flash_attention_block_alg1(Q, K, V, O, L, N, d, i, Br, Bc);
    }
}

/* ---------------- demo main (tiny test) ---------------- */

int main(void) {
    signal(SIGABRT, handler);

    unsigned int N = 2;
    unsigned int d = 4;

    float Qdata[] = {
        1, 0, 1, 0,
        0, 1, 0, 1
    };
    float Kdata[] = {
        1, 0, 1, 0,
        0, 1, 0, 1
    };
    float Vdata[] = {
        10, 20, 30, 40,
        50, 60, 70, 80
    };

    float O_expected[] = {
        20.75766f, 30.75766f, 40.75766f, 50.75766f,
        39.24234f, 49.24235f, 59.24235f, 69.24235f,
    };

    tile Q = tile_alloc(N, d);
    tile K = tile_alloc(N, d);
    tile V = tile_alloc(N, d);
    tile O = tile_alloc(N, d);
    tile L = tile_alloc(N, 1);

    memcpy(Q.data, Qdata, N * d * sizeof(float));
    memcpy(K.data, Kdata, N * d * sizeof(float));
    memcpy(V.data, Vdata, N * d * sizeof(float));

    // Choose blocks. For N=2, setting Bc=Br=2 makes one block each.
    unsigned int Br = 2;
    unsigned int Bc = 2;

    flash_attention_alg1(Q, K, V, O, L, N, d, Br, Bc);

    check_float_array_eq(O.data, O_expected, (size_t)(N * d));

    printf("O:\n");
    tile_print(O);
    printf("L (logsumexp per row):\n");
    tile_print(L);

    free(Q.data);
    free(K.data);
    free(V.data);
    free(O.data);
    free(L.data);

    return 0;
}
