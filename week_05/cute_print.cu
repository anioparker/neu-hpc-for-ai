#include <cstdio>
#include <cute/tensor.hpp>

using namespace cute;

__global__ void k_print_stuff() {
  // Print from exactly one thread to avoid spam.
  if (thread0()) {
    int x = 42;
    auto s = make_shape(C<3>{}, C<4>{});              // (3,4) compile-time shape
    auto d = make_stride(C<4>{}, C<1>{});             // row-major stride for (3,4)
    auto L = make_layout(s, d);

    print("Hello from thread0()\n");
    print("x = "); print(x); print("\n");
    print("shape = "); print(shape(L)); print("\n");
    print("stride = "); print(stride(L)); print("\n");
    print("layout = "); print(L); print("\n");

    // Nice visualization for rank-2:
    print_layout(L);
  }
}

int main() {
  k_print_stuff<<<1, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}
