#pragma once

#include <string.h>
#include "assertc.h"
#include <stdio.h>
#include <stdlib.h>
#include "min.h"
#include <limits.h>

typedef struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} dim3;

dim3 dim(unsigned int x, unsigned int y) {
    dim3 d = {x, y, 1};
    return d;
}

dim3 idx(unsigned int x, unsigned int y) {
    dim3 d = {x, y, 0};
    return d;
}

typedef struct {
    float* data;   // underlying matrix buffer
    dim3  idx;     // start coordinate in the underlying matrix
    dim3  size;    // extents of THIS tile (x = rows, y = cols)
    unsigned int ld; // leading dimension (row stride) of underlying matrix
} tile;

tile tile_alloc(unsigned int rows, unsigned int cols) {
    tile t;
    t.data = (float*)malloc(rows * cols * sizeof(float));
    t.idx  = (dim3){0,0,0};
    t.size = (dim3){rows, cols, 1};
    t.ld   = cols;                 // IMPORTANT: stride = full matrix width
    return t;
}

unsigned int tile_data_idx(tile t, unsigned int i, unsigned int j) {
    assertc(i < t.size.x);
    assertc(j < t.size.y);
    unsigned int x = t.idx.x + i;
    unsigned int y = t.idx.y + j;
    return x * t.ld + y;           // use ld, not size.y
}

float tile_at(tile t, unsigned int i, unsigned int j) {
    return t.data[tile_data_idx(t, i, j)];
}

void tile_set(tile t, unsigned int i, unsigned int j, float v) {
    t.data[tile_data_idx(t, i, j)] = v;
}

void tilecpy(tile dst, tile src) {
    assertc(dst.size.x == src.size.x);
    assertc(dst.size.y == src.size.y);
    for (unsigned int i = 0; i < src.size.x; ++i)
        for (unsigned int j = 0; j < src.size.y; ++j)
            tile_set(dst, i, j, tile_at(src, i, j));
}

void tile_print(tile t) {
    for (unsigned int i = 0; i < t.size.x; ++i) {
        for (unsigned int j = 0; j < t.size.y; ++j)
            printf("%.02f\t", tile_at(t, i, j));
        printf("\n");
    }
}

tile sub_tile(tile a, dim3 subIdx, dim3 subDim) {
    // Rule: requested tile must have non-zero extents
    assertc(subDim.x > 0);
    assertc(subDim.y > 0);

    tile t = a; // shares data and ld

    // WARNING: overflow risk with unsigned int math.
    // At minimum, assert that multiplication/addition won't overflow.
    // (Better: switch all dims/idx/ld to size_t and do checked ops.)

    assertc(subIdx.x == 0 || subDim.x <= UINT_MAX / subIdx.x);
    assertc(subIdx.y == 0 || subDim.y <= UINT_MAX / subIdx.y);

    unsigned int offx = subIdx.x * subDim.x;
    unsigned int offy = subIdx.y * subDim.y;

    assertc(a.idx.x <= UINT_MAX - offx);
    assertc(a.idx.y <= UINT_MAX - offy);

    unsigned int sx = a.idx.x + offx;
    unsigned int sy = a.idx.y + offy;

    t.idx.x = sx;
    t.idx.y = sy;

    // Parent end (may overflow without asserts)
    assertc(a.idx.x <= UINT_MAX - a.size.x);
    assertc(a.idx.y <= UINT_MAX - a.size.y);
    unsigned int ex = a.idx.x + a.size.x;
    unsigned int ey = a.idx.y + a.size.y;

    // Optional: if starting point is outside parent, abort instead of returning 0-sized tile
    // assertc(sx < ex);
    // assertc(sy < ey);

    unsigned int rx = (ex > sx) ? (ex - sx) : 0;
    unsigned int ry = (ey > sy) ? (ey - sy) : 0;

    t.size.x = umin(subDim.x, rx);
    t.size.y = umin(subDim.y, ry);

    return t;
}
