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
#include "mat_mul.h"
#include "transpose.h"
#include "softmax.h"





typedef struct {
    tile Q;  // (N, d)
    tile K;  // (N, d)
    tile V;  // (N, d)
    tile O;  // (N, d)
    tile l;  // (N, 1)
    tile m;  // (N, 1)

    tile Kj; // (B_c, d), reference to jth sub tile of K
    tile Vj; // (B_c, d), reference to jth sub tile of V
    tile Qi; // (B_r, d), reference to ith sub tile of Q
    tile Oi; // (B_r, d), reference to ith sub tile of O
    tile li; // (B_r, 1), reference to ith sub tile of l
    tile mi; // (B_r, 1), reference to ith sub tile of m
} GlobalMem;


/*
 *
 */
// shm copies of the corresponding sub tiles in GlobalMem
typedef struct {
    tile Kj; // (B_c, d)
    tile Vj; // (B_c, d)

    tile Qi; // (B_r, d),
    tile Oi; // (B_r, d)
    tile li; // (B_r, 1)
    tile mi; // (B_r, 1)

    tile Sij; // (B_r, B_c)
    tile mij; // (B_r, 1)
    tile Pij; // (B_r, B_c)
    tile lij; // (B_r, 1)
    tile mi_new; // (B_r, 1)
    tile li_new; // (B_r, 1)
} SharedMem;

// Runs per threadblock. Each threadblock solves one pair of Qi, Oi.
// Think of flash attention as Qi is the input and Oi is the output of the threadblock.
//  and then all the Kjs, Vjs are iterated over within the threadblock

// different than matmul in that matmul blocks had exactly one write per thread
// for FA, we don't have enough threads in a threadblock to do only one write
// each thread must write several locations
void flash_attention_block(tile Q, tile K, tile V, tile O, unsigned int N, unsigned int d, unsigned int M, unsigned int i, unsigned int Br, unsigned int Bc, unsigned int Tc) {

    GlobalMem g;
    g.Q = Q;
    g.K = K;
    g.V = V;
    g.O = O;

    // think of this conceptually as the input to this flash attention block
    g.Qi = sub_tile(g.Q, idx(i, 0), dim(Br, d)); // not a copy, not an alloc
    Br = g.Qi.size.x; // need to adjust Br for this threadblock due to edge clipping

    SharedMem s;
    s.Kj = tile_alloc(Bc, d);
    s.Vj = tile_alloc(Bc, d);
    s.Qi = tile_alloc(Br, d);
    s.Oi = tile_alloc(Br, d);
    s.li = tile_alloc(Br, 1);
    s.mi = tile_alloc(Br, 1);
    s.Sij = tile_alloc(Br, Bc);
    s.mij = tile_alloc(Br, 1);
    s.Pij = tile_alloc(Br, Bc);
    s.lij = tile_alloc(Br, 1);
    s.mi_new = tile_alloc(Br, 1);
    s.li_new = tile_alloc(Br, 1);

    tilecpy(s.Qi, g.Qi); // copying Qi tile from global memory to shared memory (no allocation)

    static inline void tile_fill(tile t, float v) {
    for (unsigned int r = 0; r < t.size.x; ++r)
        for (unsigned int c = 0; c < t.size.y; ++c)
            tile_set(t, r, c, v);
}

    // O_i^(0) = 0, l_i^(0) = 0, m_i^(0) = -inf   (Algorithm 1, line 4–5)
    tile_fill(s.Oi, 0.0f);
    tile_fill(s.li, 0.0f);
    tile_fill(s.mi, -INFINITY);

    for (unsigned int j = 0; j < Tc; j++) {
        g.Kj = sub_tile(g.K, idx(j, 0), dim(Bc, d)); // note: sub_tile returned may not be same dimensions as requested, but definitely at least as small.
        tile_fill(s.Kj, 0.0f);
        tilecpy(s.Kj, g.Kj);

        g.Vj = sub_tile(g.V, idx(j, 0), dim(Bc, d)); // note: sub_tile returned may not be same dimensions as requested, but definitely at least as small.
        tile_fill(s.Vj, 0.0f);
        tilecpy(s.Vj, g.Vj);

        static inline void tile_matmul(tile C, tile A, tile Bt /*actually B transposed tile, same layout*/) {
            // C: (M,N), A: (M,K), Bt: (N,K) where Bt is B transposed
            for (unsigned int i = 0; i < C.size.x; ++i) {
                for (unsigned int j = 0; j < C.size.y; ++j) {
                    float sum = 0.0f;
                    for (unsigned int k = 0; k < A.size.y; ++k) {
                        sum += tile_at(A, i, k) * tile_at(Bt, j, k);
                    }
                    tile_set(C, i, j, sum);
                }
            }
}

        // implement lines 8.
        tile KjT = tile_alloc(d, Bc);   // KjT is (d, Bc)
        // 1) KjT = transpose(Kj)  where Kj is (Bc, d) and KjT is (d, Bc)
        transpose(s.Kj.data, KjT.data, (size_t)Bc, (size_t)d);

        // 2) Sij = Qi * KjT   (Br x d) * (d x Bc) = (Br x Bc)
        mat_mul(s.Qi.data, KjT.data, s.Sij.data, (size_t)Br, (size_t)d, (size_t)Bc);

        // Lines 9-10 
       softmax(s.Sij.data, s.Pij.data, (size_t)Br, (size_t)Bc);
        // Oi = Pij * Vj  (Br x Bc) * (Bc x d) = (Br x d)
        // IMPLEMENT a matmul that takes tiles or raw pointers (contiguous).
        mat_mul(s.Pij.data, s.Vj.data, s.Oi.data, (size_t)Br, (size_t)Bc, (size_t)d);


        size_t bc = g.Kj.size.x;
        if (bc < (size_t)Bc) {
        printf("Padding happens at i=%u, j=%u: bc=%zu < Bc=%u\n",
        i, j, bc, Bc);
        }

}
    // line 12: O_i = O_i / l_i  (row-wise divide / broadcast)
    static inline void tile_broadcast_div_inplace(tile A, tile v /*(rows,1)*/) {
    for (unsigned int r = 0; r < A.size.x; ++r) {
        float denom = tile_at(v, r, 0);
        for (unsigned int c = 0; c < A.size.y; ++c) {
            tile_set(A, r, c, tile_at(A, r, c) / denom);
            }
        }
    }
    tile_broadcast_div_inplace(s.Oi, s.li);

    // write Oi back to global memory(algorithm 1 line 15)
    g.Oi = sub_tile(g.O, idx(i, 0), dim(Br, d));
    tilecpy(g.Oi, s.Oi);

    // Algorithm 1 line 13 wants: L_i = m_i^(Tc) + log(ℓ_i^(Tc))
    // but your code is saving m and l (classic FA backward stats), not L.
    // To match Alg 1 exactly, compute L and store it instead of saving (m, l).
    // The following would correspond to "save stats to HBM" but not the exact FA-2 line 13/15.
    // g.li = sub_tile(g.l, idx(i, 0), dim(Br, 1));
    // g.mi = sub_tile(g.m, idx(i, 0), dim(Br, 1));
    // tilecpy(g.li, s.li);
    // tilecpy(g.mi, s.mi);


    }
    

// Q, K, V are (N x d) and are in global memory
void flash_attention(tile Q, tile K, tile V, tile O, unsigned int N, unsigned int d) {

    // A100 will have 164e-[] KB of shared memory per chip (SM)
    // not all of it is available to us, because used for other things as well.
    // Each threadblock can only access 82 KB of shared memory
    // A100 has a max 1024 threads per thread block
    // SM has 2048 threads possible.

    unsigned int shmSize = 32 * 1024; // 32 KB.
    unsigned int M = shmSize / sizeof(float); // number of floats in shared memory

    // allows us to be a little less efficient with how we're using shared memory
    unsigned int Bc = umin(div_ceil(M, 8 * d), N); // # rows of K, V we can a process in a single thread block
    unsigned int Br = umin(div_ceil(M, 8 * d), N); // # rows of Q  we can a process in a single thread block
    unsigned int Tr = div_ceil(N, Br);
    unsigned int Tc = div_ceil(N, Bc);
    printf("Br=%d Bc=%d d=%d\n", Br, Bc, d);

    // Each threadblock solves one pair of Qi, Oi.
    // Think of flash attention as Qi is the input and Oi is the output of the threadblock.
    //  and then all the Kjs, Vjs are iterated over within the threadblock
    for (unsigned int i = 0; i < Tr; i++) {
        flash_attention_block(Q, K, V, O, N, d, M, i, Br, Bc, Tc);
    }
}

int main(void) {
    signal(SIGABRT, handler);

    printf("entry\n");

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

    float Odata[] = {
        20.75766f, 30.75766f, 40.75766f, 50.75766f,
        39.24234f, 49.24235f, 59.24235f, 69.24235f,
    };

    unsigned int size = N * d * sizeof(float);

    tile Q = tile_alloc(N, d);
    tile K = tile_alloc(N, d);
    tile V = tile_alloc(N, d);
    tile O = tile_alloc(N, d);

    memcpy(Q.data, Qdata, size);
    memcpy(K.data, Kdata, size);
    memcpy(V.data, Vdata, size);


    flash_attention(Q, K, V, O, N, d);

    check_float_array_eq(O.data, Odata, N * d);

    free(Q.data);
    free(K.data);
    free(V.data);
    free(O.data);
    free(KjT.data);


    return 0;
}


