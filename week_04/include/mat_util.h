#pragma once

size_t at(size_t r, size_t c, size_t rows, size_t cols) {
    assertc(r < rows);
    assertc(c < cols);
    return r * cols + c;
}
