BUILD_DIR ?= build

.PHONY: build run clean sweep

build:
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j

run: build
	./$(BUILD_DIR)/sgemm --kernel 5 --m 4092 --n 4092 --k 4092 --reps 50

sweep: build
	python scripts/sweep.py --bin $(BUILD_DIR)/sgemm --m 4092 --n 4092 --k 4092 --reps 50

clean:
	rm -rf $(BUILD_DIR) results/raw/*.csv

