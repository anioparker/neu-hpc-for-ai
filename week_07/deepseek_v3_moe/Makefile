CC = gcc
CFLAGS = -O2 -Wall -Wextra -Wpedantic -std=c11 -Iinclude
LDFLAGS = -lm

SRC = src/tensor.c src/utils.c src/router.c src/expert.c src/moe.c src/main.c
OBJ = $(SRC:.c=.o)
OUT = deepseek_v3_moe

TEST_COMMON = src/tensor.c src/utils.c src/router.c src/expert.c src/moe.c

TEST_ROUTER = test_router
TEST_TOPK = test_topk
TEST_EXPERT = test_expert
TEST_MOE = test_moe

.PHONY: all clean test

all: $(OUT)

$(OUT): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_ROUTER): tests/test_router.c $(TEST_COMMON)
	$(CC) $(CFLAGS) -o $@ tests/test_router.c $(TEST_COMMON) $(LDFLAGS)

$(TEST_TOPK): tests/test_topk.c $(TEST_COMMON)
	$(CC) $(CFLAGS) -o $@ tests/test_topk.c $(TEST_COMMON) $(LDFLAGS)

$(TEST_EXPERT): tests/test_expert.c $(TEST_COMMON)
	$(CC) $(CFLAGS) -o $@ tests/test_expert.c $(TEST_COMMON) $(LDFLAGS)

$(TEST_MOE): tests/test_moe.c $(TEST_COMMON)
	$(CC) $(CFLAGS) -o $@ tests/test_moe.c $(TEST_COMMON) $(LDFLAGS)

test: $(TEST_ROUTER) $(TEST_TOPK) $(TEST_EXPERT) $(TEST_MOE)

clean:
	rm -f $(OUT) $(OBJ) $(TEST_ROUTER) $(TEST_TOPK) $(TEST_EXPERT) $(TEST_MOE)
