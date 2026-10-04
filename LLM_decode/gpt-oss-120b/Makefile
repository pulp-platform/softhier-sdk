HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(HERE)/../../..)
BUILD_DIR ?= $(ROOT)/build/gpt_oss_decode/full
SMOKE_DIR ?= $(ROOT)/build/gpt_oss_decode/smoke
TEST_DIR ?= $(ROOT)/build/gpt_oss_decode/regression
PYTHON ?= python3
BATCHES ?= 1 8 64

.PHONY: all data build run validate benchmark smoke regression
all: build

$(BUILD_DIR)/manifest.json: $(HERE)/tools/generate.py $(HERE)/config/model.json $(HERE)/config/arch.py
	$(PYTHON) $(HERE)/tools/generate.py --build-dir $(BUILD_DIR)

# Explicitly regenerate all synthetic weights and input files.
data:
	$(PYTHON) $(HERE)/tools/generate.py --build-dir $(BUILD_DIR)

build: $(BUILD_DIR)/manifest.json
	$(PYTHON) $(HERE)/tools/build.py --build-dir $(BUILD_DIR)

run: build
	$(PYTHON) $(HERE)/tools/run.py --build-dir $(BUILD_DIR) --batches $(BATCHES)

validate:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) $(HERE)/tools/reference.py --build-dir $(BUILD_DIR) --batches $(BATCHES)

benchmark: run
	OPENBLAS_NUM_THREADS=1 $(PYTHON) $(HERE)/tools/reference.py --build-dir $(BUILD_DIR) --batches $(BATCHES)
	$(PYTHON) $(HERE)/tools/report.py --build-dir $(BUILD_DIR) --batches $(BATCHES)

smoke:
	$(PYTHON) $(HERE)/tools/generate.py --build-dir $(SMOKE_DIR) --smoke --batches 1 4
	$(PYTHON) $(HERE)/tools/build.py --build-dir $(SMOKE_DIR)
	$(PYTHON) $(HERE)/tools/run.py --build-dir $(SMOKE_DIR) --batches 1 4
	OPENBLAS_NUM_THREADS=1 $(PYTHON) $(HERE)/tools/reference.py --build-dir $(SMOKE_DIR) --batches 1 4

regression:
	$(PYTHON) $(HERE)/tools/regression.py --build-dir $(TEST_DIR)
