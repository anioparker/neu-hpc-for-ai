#include <string.h>

#include "tile.h"

int main(void) {
    float data[] = {
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120,
        130, 140, 150, 160,
        170, 180, 190, 200,
    };

    unsigned int N = 5;
    unsigned int d = 4;

    tile t = tile_alloc(N, d);
    memcpy(t.data, data, N * d * sizeof(float));

    tile_print(t);

    tile s0 = sub_tile(t, idx(0, 0), dim(3, 3));
    printf("\ns0:\n"); // 10-110
    tile_print(s0);

    tile s1 = sub_tile(t, idx(1, 0), dim(3, 3));
    printf("\ns1:\n"); //130-190
    tile_print(s1);

    tile s2 = sub_tile(s1, idx(0, 0), dim(2, 2));
    printf("\ns2:\n"); // 130, 180
    tile_print(s2);

}
