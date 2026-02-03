#pragma once
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

struct Args {
  int kernel = 0; // 0 = cublas
  int m = 4092, n = 4092, k = 4092;
  int reps = 50;
  int warmup = 5;
  bool verify = true;
};

inline bool starts_with(const char* s, const char* prefix) {
  return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

inline Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--kernel") && i + 1 < argc) a.kernel = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--m") && i + 1 < argc) a.m = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) a.n = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) a.k = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) a.reps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) a.warmup = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-verify")) a.verify = false;
  }
  return a;
}
